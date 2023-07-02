r"""
Element of a Weil module.

AUTHORS:
   - Fredrik Strömberg
   - Nils-Peter Skoruppa
   - Stephan Ehlen

"""
from sage.all import ZZ, CC, RR, QQ
from sage.arith.misc import kronecker, gcd, divisors, odd_part, valuation, inverse_mod, \
    hilbert_symbol
from sage.functions.generalized import sgn
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method
from sage.misc.functional import is_odd, is_even, sqrt
from sage.misc.misc_c import prod
from sage.modular.arithgroup.congroup_sl2z import SL2Z
from sage.modules.free_module_element import vector
from sage.rings.integer import Integer
from sage.structure.formal_sum import FormalSum

from fqm_weil.modules.finite_quadratic_module.finite_quadratic_module_element import \
    FiniteQuadraticModuleElement
from fqm_weil.modules.utils import factor_matrix_in_sl2z
from fqm_weil.modules.weil_module.utils import _entries, hilbert_symbol_infinity


class WeilModuleElement(FormalSum):
    r"""
    An element of a Weil module $K[A]$.
    """

    def __init__(self, d, parent=None, check=True, verbose=0):
        r"""
        INPUT
            W -- a Weil module $K[A]$
            d -- a dictionary of pairs $a:k$, where $a$ is an element of
                 a finite quadratic module $A$ of level $l$ and $k$ an element
                 of the field $K$ of the $l$-th roots of unity.

        EXAMPLES::

           sage: from fqm_weil.all import FiniteQuadraticModule, WeilModule
           sage: F = FiniteQuadraticModule([3,3],[0,1/3,2/3])
           sage: W = WeilModule(F)
           sage: a,b = F.gens()
           sage: z = W(a+5*b); z
           e0 + 2*e1

        """
        from .weil_module import WeilModule
        self._coordinates = []
        self._verbose = verbose
        if isinstance(d, tuple):
            d = list(d)
        if not isinstance(d, list):
            if isinstance(d, FiniteQuadraticModuleElement):
                d = [(1, d)]
            elif d == 0:
                d = [(1, 0)]
            else:
                s = "d={0} Is not instance! Is:{1}".format(d, type(d))
                raise TypeError(s)
        if not isinstance(d[0], tuple):
            self._coordinates = d
            d = [(n, list(parent.finite_quadratic_module())[i]) for i, n in enumerate(d)]
        if not parent:
            parent = WeilModule(d[0][1].parent())
        if not isinstance(parent, WeilModule):
            raise TypeError(
                "Call as WeilModuleElement(W,d) where W=WeilModule. Got W={0}".format(parent))

        for t in d:
            if not isinstance(t, tuple) or len(t) != 2 or t[1] not in parent._QM:
                raise TypeError("argument must be a list of tuples of the form (n,x) "
                                "where n is an integer and x an element of the finite "
                                "quadratic module")
        FormalSum.__init__(self, d, parent, check, True)
        if not self._coordinates:
            self._coordinates = [0] * parent.rank()
            for i, x in self._data:
                ix = parent._el_index(x.list())
                self._coordinates[ix] = i

        self._W = parent
        self._parent = parent
        # BaseField including both FQM.level() roots and eight-roots of unity
        self._K = self._W._K  # Base field
        self._QM = parent._QM  # f.q.m.
        self._zl = parent._zl  # e(1/level)
        self._z8 = parent._z8  # e(1/8)
        self._n = parent._n
        self._sqn = parent._sqn
        #    self._minus_element.append(self._QM.list().index(-self._QM.list()[i]))
        self._level = self._QM.level()
        # Pre-compute invariants
        self._inv = parent._inv

    def _cache_key(self):
        return ('WeilModuleElement', tuple(self._data), self._parent)

    def as_finite_quadratic_module_element(self):
        """
        If the coefficients are all integers then this formal sum
        can be interpreted as a "real" sum.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, WeilModule
            sage: F = FiniteQuadraticModule([3,3],[0,1/3,2/3])
            sage: W = WeilModule(F)
            sage: w = W.basis()[1]
            sage: w
            e0
            sage: w.as_finite_quadratic_module_element()
            e0
            sage: (-w).as_finite_quadratic_module_element()
            2*e0
            sage: W(F.0 + F.1).as_finite_quadratic_module_element()
            e0 + e1
            sage: (-W(F.0 + F.1)).as_finite_quadratic_module_element() in F
            True
        """
        res = 0
        try:
            for i, x in self._data:
                res += ZZ(i)*x
        except TypeError:
            raise TypeError(f"Cannot interpret {self} as finite quadratic module element")
        return res

    ###################################
    ## Operations
    ###################################

    def __repr__(self):
        s = ""
        i = 0
        for j, x in self:
            if j == 1 and i == 0:
                s += "{0}".format(x)
            elif j == 1 and i > 0:
                s += " + {0}".format(x)
            elif j == -1 and i == 0:
                s += "-({0})".format(x)
            elif j == -1 and i > 0:
                s += " - ({0})".format(x)
            elif i == 0:
                s += "{0}*({1})".format(j, x)
            else:
                s += " + {0}*({1})".format(j, x)
            i += 1
        return s

    def parent(self, x=None):
        """
        Parent Weil module of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, WeilModule, WeilModuleElement
            sage: F = FiniteQuadraticModule([3,3],[0,1/3,2/3])
            sage: W = WeilModule(F)
            sage: w = WeilModuleElement([1,1], parent=W); w
            0 + e1
            sage: w.parent() == W
            True
        """
        return self._parent

    def coordinates(self):
        """
        Coordinates of self with respect to the basis of the parent Weil module.

        """
        return self._coordinates

    @cached_method
    def Bi(self, i, j):
        return self._QM.B(self._QM(list(self._W._elt(i)), can_coords=True),
                          self._QM(list(self._W._elt(j)), can_coords=True))

    def B(self, other):
        v1 = self._coordinates
        v2 = other._coordinates
        s = 0
        for i in len(v1):
            for j in len(v2):
                s += self._QM.B(list(self._QM)[i], list(self._QM)[j])
        return s

    @cached_method
    def Q(self, i):
        # return self._QM.Q(self._L[i])
        return self._QM.Q(self._QM(list(self._W._elt(i)), can_coords=True))

    @cached_method
    def _minus_element(self, ii):
        return self._W.neg_index(ii)

    def action(self, A):
        r"""
        Return the result of applying the element $A$ of $SL2Z$
        to this element.

        EXAMPLES
        """
        if A not in SL2Z:
            raise TypeError("{0} must be an element of SL2Z".format(A))

        # if A[1,0] % self._level == 0:
        #    return self._action_of_Gamma0(A)
        # else:
        return self.action_of_SL2Z(A)

        # def __rmul__(self,obj):
        # return self.__mul__( obj)

    ## Note that we want matrices from SL2Z to act from the left
    ## i.e. with an _rmul_

    def _rmul_(self, obj, act='l'):
        r"""
        Return the result of applying obj to this element from the left. I.e
        INPUT: obj
        OUPUT =
              obj*self = a*self  if obj=a in CyclotomicField(lcm(level,8))
              obj*self = rho_{Q}(A)*self if obj= A in SL2Z
        TODO: Implement the orthogonal group of $A$
        EXAMPLES

        """
        if self._K.has_coerce_map_from(obj.parent()):
            # print "case 1"
            d = list()
            for (j, x) in self:
                # print "(j,x)=",j,x
                d.append((j * obj, x))
            return WeilModuleElement(d, self.parent())
        # multiply with the group element instead
        if self._QM.base_ring().has_coerce_map_from(obj.parent()):
            # print "case 2"
            d = list()
            for (j, x) in self:
                d.append((j, obj * x))
            return WeilModuleElement(self.parent(), d)

        if obj in SL2Z:
            return self.action_of_SL2Z(obj, act)
        # #         # TODO: obj can also be an element of the orthogonal group of $A$
        raise TypeError("action of {0} on {1} not defined".format(obj, self))

    # def __eq__(self,other):
    #     r"""
    #     Check if self is equal to other
    #     TODO: check reduced words
    #     """
    #     if not hasattr(other,"parent"): return False
    #     if self.parent()!=other.parent(): return False
    #     v1 = self._coordinates
    #     v2 = other._coordinates
    #     if
    #     return self._data == other._data
    def __add__(self, other):
        r"""
        Add self and other. We use this since FormalSum does not reduce properly
        """
        if other.parent() != self.parent(): raise ValueError(
            "Can not add {0} to {1}".format(self, other))
        v1 = self._coordinates
        v2 = other._coordinates
        v_new = [v1[i] + v2[i] for i in range(len(v1))]
        return WeilModuleElement(v_new, self.parent())

    def __sub__(self, other):
        r"""
        Add self and other. We use this since FormalSum does not reduce properly
        """
        if other.parent() != self.parent(): raise ValueError(
            "Can not add {0} to {1}".format(self, other))
        v1 = self._coordinates
        v2 = other._coordinates
        v_new = [v1[i] - v2[i] for i in range(len(v1))]
        return WeilModuleElement(v_new, self.parent())

    def __mul__(self, obj):
        r"""
        Return the result of applying obj to this element from the right. I.e
        INPUT: obj
        OUPUT =
              obj*self = a*self  if obj=a in CyclotomicField(lcm(level,8))

        NOTE:
        We have only implemented elements in SL2Z to act from the left
        TODO: Implement the orthogonal group of $A$ and transposed PSL2Z action

        EXAMPLES

        """
        return self._rmul_(obj, act='r')
        # if obj.parent() == self.parent().base_ring():
        # Multiply the coefficient of x in the FormalSum

    def action_of_SL2Z(self, M, act='l', mode=0):
        r"""
          INPUT:
            M = element of SL2Z
            mode : determines how the Weil rpresentation is calculated
              0 => use formula (default)
              1 => use factorization
            act : determines if we act from the right or left
              'l' => rho(A)*x
              'r' => x^t * rho(A)
          OUTPUT: [s,fac]
             where s is a Formal Sum over
             K=CyclotomicField(lcm(8,level) and
             fact is a scalar and such that
               rho_{Q}(M)*self = fact*s
             where
          EXAMPLE:
            FQ = FiniteQuadraticModule
            A  = Element of SL2Z
            W = WeilModule(FQ)
            x=FQ.gens()[0]
            WE=W(x)
            [r,s] = x.action_of_SL2Z(A)
            fact*x = rho_{Q}(M)
        """
        # Since we only want to act on this specific element we first
        # figure out which matrix-entries we need to compute:
        if mode == 0:
            filter = matrix(ZZ, self._n)
            # print filter
            for (k, x) in self:
                jj = self._parent._el_index(x.list())
                # jj = self._L.index(x)
                for ii in range(0, self._n):
                    if (act == 'l'):
                        filter[ii, jj] = 1
                    else:
                        filter[jj, ii] = 1
            [r, fact] = self._action_of_SL2Z_formula(M, filter)
        else:
            # when we use factorization we must compute all elements
            [r, fact] = self._action_of_SL2Z_factor(M)
        # Compute rho_Q(A)*self
        res = FormalSum([(0, 0)], self._W)
        for (k, x) in self:
            # print "k,x=",k,x
            jj = self._parent._el_index(x.list())
            for ii in range(0, self._n):
                if (act == 'l'):
                    res = res + FormalSum(
                        [(r[ii, jj], self._QM(self._W._elt(ii), can_coords=True))], self._W)
                    # res=res+FormalSum([(r[ii,jj],self._L[ii])],self._W)
                else:
                    res = res + FormalSum(
                        [(r[jj, ii], self._QM(self._W._elt(ii), can_coords=True))], self._W)
                    # res=res+FormalSum([(r[jj,ii],self._L[ii])],self._W)
        return [res, fact]

    ## Action by special (simple) elements of SL(2,Z)
    def _action_of_T(self, b=1, sign=1, filter=None):
        r""" Action by the generator sign*T^pow=[[a,b],[0,d]]
        where a=d=sign
        """
        r = matrix(self._K, self._n, sparse=filter is not None)
        if sign == -1:
            si = self._QM.sigma_invariant() ** 2
        else:
            si = 1
        for ii in range(0, self._n):
            if filter != None and filter[ii, ii] != 1:
                continue
            if sign == 1:
                r[ii, ii] = self._zl ** (ZZ(b) * self._level * self.Q(ii))
            else:
                # r[self._n-1-ii,ii] = self._zl**(b*self._level*self._QM.Q(self._L[ii]))
                r[self._minus_element(ii), ii] = si * self._zl ** (b * self._level * self.Q(ii))
        return [r, 1]

    def _action_of_S(self, filter=None, sign=1, mult_by_fact=False):
        r"""
        Action by the generator S=[[0,-1],[1,0]]
        """
        r = matrix(self._K, self._n, sparse=filter is not None)
        if sign == -1:
            si = self._K(self._QM.sigma_invariant() ** 3)
            if is_odd(self.parent().signature()):
                si = -si  # sigma(Z,A)
        else:
            si = self._K(self._QM.sigma_invariant())
        for ii in range(0, self._n):
            for jj in range(0, self._n):
                if filter != None and filter[ii, jj] != 1:
                    continue
                arg = -sign * self._level * self.Bi(ii, jj)
                # arg = -self._level*self._QM.B(self._L[ii],self._L[jj])
                r[ii, jj] = si * self._zl ** arg
        # r = r*
        fac = self._parent._sqn ** -1
        return [r, fac]

    def _action_of_STn(self, pow=1, sign=1, filter=None):
        r""" Action by  ST^pow or -ST^pow
        NOTE: we do not divide by |D|
        """
        ## Have to find a basefield that also contains the sigma invariant
        if pow == 0:
            return self._action_of_S(filter, sign)
        r = matrix(self._K, self._n)
        if sign == -1:
            si = self._K(self._QM.sigma_invariant() ** 3)
            if is_odd(self.parent().signature()):
                si = -si  # sigma(Z,A)
        else:
            si = self._K(self._QM.sigma_invariant())
        for ii in range(self._n):
            for jj in range(self._n):
                argl = self._level * (pow * self.Q(jj) - sign * self.Bi(ii, jj))
                # ii = self._L.index(x); jj= self._L.index(j)
                if filter != None and filter[ii, jj] != 1:
                    continue
                r[ii, jj] = si * self._zl ** argl
        fac = self._parent._sqn ** -1
        return [r, fac]

    def _action_of_Z(self, filter=None):
        r""" Action by  Z=-Id
        NOTE: we do not divide by |D|
        """
        ## Have to find a basefield that also contains the sigma invariant
        r = matrix(self._K, self._n)
        for ii in range(0, self._n):
            if filter != None and filter[ii, ii] != 1:
                continue
            jj = self._W.neg_index(ii)
            r[ii, jj] = 1
        r = r * self._QM.sigma_invariant() ** 2
        return [r, 1]

    def _action_of_Id(self, filter=None):
        r""" Action by  Z=-Id
        NOTE: we do not divide by |D|
        """
        ## Have to find a basefield that also contains the sigma invariant
        r = matrix(self._K, self._n)
        for ii in range(0, self._n):
            if filter != None and filter[ii, ii] != 1:
                continue
            r[ii, ii] = 1
            # r = r*self._QM.sigma_invariant()**2
        return [r, 1]

    # Action by Gamma_0(N) through formula
    def _action_of_Gamma0(self, A, filter=None):
        r"""
        Action by A in Gamma_0(l)
        where l is the level of the FQM
        INPUT:
           A in SL2Z with A[1,0] == 0 mod l
           act ='r' or 'l' : do we act from left or right'
        filter = |D|x|D| integer matrix with entries 0 or 1
                         where 1 means that we compute this entry
                         of the matrix rho_{Q}(A)
        """
        a = A[0, 0];
        b = A[0, 1];
        c = A[1, 0];
        d = A[1, 1]  # [a,b,c,d]=elts(A)
        if c % self._level != 0:
            raise ValueError("Must be called with Gamma0(l) matrix! not A=" % (A))
        r = matrix(self._K, self._n, sparse=True)
        for ii in range(0, self._n):
            for jj in range(0, self._n):
                if (self._QM(self._W._elt(ii), can_coords=True) == d * self._QM(self._W._elt(jj),
                                                                                can_coords=True) and (
                        filter == None or filter[ii, jj] == 1)):
                    argl = self._level * b * d * self._QM.Q(
                        self._QM(self._W._elt(jj), can_coords=True))
                    r[ii, jj] = self._zl ** argl
        # Compute the character
        signature = self._inv['signature']
        if self._level % 4 == 0:
            test = (signature + kronecker(-1, self._n)) % 4
            if is_even(test):
                if test == 0:
                    power = 1
                elif test == 2:
                    power = -1
                if d % 4 == 1:
                    chi = 1
                else:
                    chi = self._z8 ** (power * 2)
                chi = chi * kronecker(c, d)
            else:
                if test == 3:
                    chi = kronecker(-1, d)
                else:
                    chi = 1
            chi = chi * kronecker(d, self._n * 2 ** signature)
        else:
            chi = kronecker(d, self._n)
        r = r * chi
        return [r, 1]

    # Now we want the general action

    def _action_of_SL2Z_formula(self, A, filter=None, **kwds):
        r"""
        The Action of A in SL2(Z) given by a matrix rho_{Q}(A)
        as given by the formula
        filter = |D|x|D| integer matrix with entries 0 or 1
                         where 1 means that we compute this entry
                         of the matrix rho_{Q}(A)
        """
        ## extract eleements from A
        [a, b, c, d] = _entries(A)
        # check that A is in SL(2,Z)
        # if(A not in SL2Z):
        # ()    raise  TypeError,"Matrix must be in SL(2,Z)!"
        ## Check if we have a generator
        sign = 1
        if c == 0:
            if b == 0:
                if a < 0:
                    return self._action_of_Z(filter)
                else:
                    return self._action_of_Id(filter)
            if a < 1:
                sign = -1
            else:
                sign = 1
            return self._action_of_T(b, sign, filter)
        if c % self._level == 0:
            return self._action_of_Gamma0(A)
        if abs(c) == 1 and a == 0:
            if self._verbose > 0:
                print("call STn with pos={0} and sign={1}".format(abs(d), sgn(c)))
            sic = sgn(c)
            return self._action_of_STn(pow=d * sic, sign=sic, filter=filter)
            # These are all known easy cases
        # recall we assumed the formula
        if c < 0 or (c == 0 and d < 0):  # change to -A
            a = -a;
            b = -b;
            c = -c;
            d = -d
            A = SL2Z(matrix(ZZ, 2, 2, [a, b, c, d]))
            sign = -1
        else:
            sign = 1
        xis = self._get_xis(A)
        xi = 1
        for q in xis.keys():
            xi = xi * xis[q]
        norms_c = self._get_all_norm_alpha_cs(c)
        # norms_c_old=self._get_all_norm_alpha_cs_old(c)
        if self._verbose > 0:
            print("c={0}".format(c))
            print("xis={0}".format(xis))
            print("xi={0},{1}".format(xi, CC(xi)))
            print("norms={0}".format(norms_c))
        # print "11"
        r = matrix(self._K, self._n)
        # if sign==-1:
        #     #r=r*self._QM.sigma_invariant()**2
        #     si = -1*self._QM.sigma_invariant()**2
        #     #if is_odd(self.parent().signature()):
        #     #    if c>0 or (c==0 and d<0):
        #     #        si = -si ## sigma(Z,A)
        # else:
        #     si=1
        # if self._verbose>0:
        #     print("si={0}".format(si))
        #     print("sign={0}".format(sign))
        for na in range(0, self._n):
            # print "na=",na
            # alpha=self._L[na]
            # if sign==-1:
            #    na=self._minus_element[na] #-alpha
            # na=self._L.index(-alpha)
            for nb in range(0, self._n):
                if filter is not None and filter[na, nb] == 0:
                    continue
                if sign == -1:
                    # print type(nb)
                    nbm = self._minus_element(nb)  # -alpha
                else:
                    nbm = nb
                # beta=self._L[nb]
                # gamma=alpha-d*beta
                # c*alpha' =
                gi = self.lin_comb(na, -d, nbm)
                try:
                    ngamma_c = norms_c[gi]
                    # ngamma_c_old=norms_c_old[gamma]
                except KeyError:
                    # print alpha," not in D^c*"
                    continue
                # ngamma_c_old=self._norm_alpha_c(gamma,c)
                # arg_old=a*ngamma_c_old+b*self._QM.B(gamma,beta)+b*d*self._QM.Q(beta)
                # gi = self._L.index(gamma)
                # CHECK: + or - ? arg=a*ngamma_c+b*self.B(gi,nbm)+b*d*self.Q(nbm)
                arg = a * ngamma_c + b * self.Bi(na, nbm) - b * d * self.Q(nbm)
                larg = arg * self._level
                if self._verbose > 0 and na == 0 and nb == 1:
                    print("na,nb,nbm={0},{1},{2}".format(na, nb, nbm))
                    print("gi={0}".format(gi))
                    print("ngamma_c[{0}]={1}".format(gi, ngamma_c))
                    print("b*B(na,nbm)={0}".format(b * self.Bi(na, nbm)))
                    print(
                        "arg={0}*{1}+{2}*{3}-{4}*{5}*{6}".format(a, ngamma_c, b, self.Bi(na, nbm),
                                                                 b, d, self.Q(nbm)))
                    print("arg={0}".format(arg))
                    print("e(arg)={0}".format(CC(0, arg * RR.pi() * 2).exp()))
                    print("e_L(arg)={0}".format(CC(self._zl ** (larg))))
                # if na==nb:
                #    print "arg[",na,"]=",a*ngamma_c,'+',b*self.B(gi,nbm),'-',b*d*self.Q(nbm),'=',arg

                r[na, nb] = xi * self._zl ** (larg)
                if self._verbose > 0 and na == 0 and nb == 1:
                    print("r[{0},{1}]={2}".format(na, nb, r[na, nb]))
                    print("r[{0},{1}]={2}".format(na, nb, r[na, nb].complex_embedding(53)))
                # print "xi=",xi
                # print "zl=",self._zl
        fac = self._get_lenDc(c)
        # print "12"
        return [r, (QQ(fac) / QQ(self._n)).sqrt()]

    def _get_lenDc(self, c):
        r"""
        compute the number of elements in the group of elements of order c
        """
        # Check to see if everything is precomputed if so we fetch it
        g = gcd(c, self._n)
        try:
            n = self._All_len_Dcs[g]
        except AttributeError:
            # if the dictionary doesn't exist, create it and store the correct value
            n = self._get_one_lenDc(c)
            self._All_len_Dcs = dict()
            self._All_len_Dcs[g] = n
        except KeyError:
            # The dictionary exist but this value does not
            n = self._get_one_lenDc(c)
            self._All_len_Dcs[g] = n
        return n

    @cached_method
    def _get_one_lenDc(self, c):
        r"""
        compute the number of elements in the group of elements of order c
        """
        n = 0
        for ii in range(0, self._n):
            x = self._QM(self._W._elt(ii), can_coords=True)
            if (c * x == self._QM(0)):
                n = n + 1
        return n

    def _get_all_lenDc(self):
        r"""
        Compute the number of elements in the group of elements of order c
        for all c dividing |D|
        """
        res = dict()
        divs = divisors(self._n)
        for c in divs:
            res[c] = self._get_one_lenDc(c)
        # set all lengths
        self._All_len_Dcs = res

    # Setup the functions for computing the Weil representation

    def get_xis(self, *args, pset=None):
        return self._get_xis(*args, pset=pset)

    def xi(self, *args, met='old'):
        """
        Return xi(A).
        """
        # if met == 'old':
        #     return prod(x for x in self._get_xis(*args).values())
        # else:
        return prod(x for x in self._get_xis(*args).values())

    def _get_xis(self, *args, pset=None):
        r"""
        compute the p-adic factors: \xi_0, \xi_p, p | |D|

        if pset = p we only return the p-factor of xi.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, WeilModule
            sage: F=FiniteQuadraticModule('3^2')
            sage: w = WeilModule(F)(F.0)
            sage: A=matrix([[-61,92],[59,-89]])
            sage: w.xi(A) == -1
            True
            sage: F=FiniteQuadraticModule('7^-2')
            sage: w = WeilModule(F)(F.0)
            sage: A=matrix([[-17,-79],[14,65]])
            sage: w.xi(A) == 1
            True
            sage: R = matrix([[1,0],[3,5]])
            sage: G = matrix([[2/5,1/5],[1/5,3/5]])
            sage: F = FiniteQuadraticModule(R,G)
            sage: W = WeilModule(F)
            sage: w = W(F.0)
        """
        JD = self._QM.jordan_decomposition()
        a, b, c, d = _entries(*args)
        if self._verbose > 0:
            print("pset={0}".format(pset))
            print("JD={0}".format(JD))
        sign = self._inv['signature'][-1]
        oddity = self._inv['total oddity']
        z8 = self._z8
        if c == 0 and d == 1:
            return {-1: 1}
        if c == 0 and d == -1:
            return {-1: z8 ** (-2 * sign)}
        xis = {p: 1 for p in ZZ(self._level).prime_factors()}
        xis[-1] = z8 ** (-2 * sign)
        if sign % 2 == 0:
            xis[0] = 1
        else:
            xis[0] = kronecker(-a, c) * hilbert_symbol_infinity(-a, c)
            xis[-1] *= hilbert_symbol_infinity(-1, c)
        if c % 2 != 0:
            xis[2] = 1
        else:
            xis[2] = z8 ** (oddity * (a + 1))
            if sign % 2 == 1:
                c2 = odd_part(c)
                xis[2] *= z8 ** ( (c2 - 1) * (a + 1))
        # print("xis=",xis)
        for comp in JD:
            q = comp.q
            p = comp.p
            n = comp.n
            k = comp.k
            # q = p ** k
            constituent = JD.constituent(q).as_finite_quadratic_module()
            xi_comp = 1
            if c % p != 0:
                xi_comp *= constituent.char_invariant(c, p)[0]
            else:
                xi_comp *= kronecker(-a, q) ** n
                if not comp.is_type_I() or k != valuation(c, 2):
                    # if the component is not type I or xc does not belong to this component.
                    # print("inv=",constituent.char_invariant(-a*c, p))
                    xi_comp *= constituent.char_invariant(-a * c, p)[0]
                    # print(f"xis[{p}]=",xi_comp)
            xis[p] *= xi_comp  # **n
        return xis

    @cached_method
    def _get_all_norm_alpha_cs(self, c):
        r"""
        Computes a vector of all Q(alpha_c)
        for alpha in D  (==0 unless alpha_c is in D^c*)
        """
        res = dict()
        for ai in range(self._n):
            # for alpha in self._L:
            nc = self._get_norm_alpha_c(ai, c)
            if nc is not None:
                res[ai] = nc
        return res

    @cached_method
    def _get_norm_alpha_c(self, ai, c):
        r"""
        FQM = Finite Quadratic Module
        Test before that alpha is in D^c*!!
        """
        alpha = self._QM(self._W._elt(ai), can_coords=True)
        xc = self._QM.xc(c)
        if xc != 0:
            gammatmp = alpha - xc
        else:
            gammatmp = alpha
        # We need to find its inverse mod c
        # i.e. gammatmp/c
        # print alpha,gammatmp
        if gcd(c, self._level) == 1:
            cc = inverse_mod(c, self._level)
            gamma = (cc * gammatmp).list()
        else:
            gamma = []
            for jj, g in enumerate(self._QM.gens()):
                for x in range(g.order()):
                    if (c * x - gammatmp.list()[jj]) % g.order() == 0:
                        gamma.append(x)
                        break
            if len(gamma) < len(self._QM.gens()):
                if self._verbose > 1:
                    print("c={0}".format(c))
                    print("x_c={0}".format(xc))
                    print("gammatmp={0}".format(gammatmp))
                    print("y=gamma/c={0}".format(gamma))
                return None
        # gamma=y #vector(y)
        #    # raise ValueError, "Found no inverse (alpha-xc)/c: alpha=%s, xc=%s, c=%s !" %(alpha,xc,c)
        if self._verbose > 0:
            print("xc={0}".format(xc))
            # print "orders=",self._W._gen_orders
            # print "gamma=",gamma
        if len(gamma) != len(self._W._gen_orders):
            print("W={0}".format(self._W))
            print("F={0}".format(list(self._W._QM)))
            print("F.gens={0}".format(self._W._QM.gens()))
            print("F.gram{0}".format(self._W._QM.gram()))
            print("is_nondeg={0}".format(self._W._QM.is_nondegenerate()))
            print("ai={0}".format(ai))
            print("c={0}".format(c))
        gi = self._W._el_index(gamma)
        if self._verbose > 0:
            print("gi={0}".format(gi))
        # res=c*self._QM.Q(gamma)
        res = c * self.Q(gi)
        if xc != 0:
            # res=res+self._QM.B(xc,gamma)
            if self._verbose > 0:
                print("xc={0}".format(xc))
                print("xc.list={0}".format(xc.list()))
                print("orders={0}".format(self._W._gen_orders))
            xci = self._W._el_index(xc.list())
            res = res + self.Bi(xci, gi)
        return res

    def _get_all_norm_alpha_cs_old(self, c):
        r"""
        Computes a vector of all Q(alpha_c)
        for alpha in D  (==0 unless alpha_c is in D^c*)
        """
        res = dict()
        for alpha in self._L:
            nc = self._get_norm_alpha_c_old(alpha, c)
            if nc is not None:
                res[alpha] = nc
        return res

    def _get_norm_alpha_c_old(self, alpha, c):
        r"""
        FQM = Finite Quadratic Module
        Test before that alpha is in D^c*!!
        """
        ## Make an extra test. This should be removed in an efficient version
        # Dcs=D_upper_c_star(FQM,c)
        # if(Dcs.count(alpha)==0):
        #    raise ValueError, "alpha=%s is not in D^c* for c=%s, D^c*=%s" %(alpha,c,Dcs)
        xc = self._get_xc(c)
        # print "xc=",xc
        if xc != 0:
            gammatmp = alpha - xc
        else:
            gammatmp = alpha
        # first a simple test to see if gammatmp is not in D^c
        test = gammatmp * Integer(self._n / gcd(self._n, c))
        if test != self._QM(0):
            return None
        # We need to find its inverse mod c
        # i.e. gammatmp/c
        # print alpha,gammatmp
        try:
            for y in self._L:
                yy = c * y
                # print "yy=",yy
                if yy == gammatmp:
                    gamma = y  # (alpha-xc)/c
                    raise StopIteration
        except StopIteration:
            pass
        else:
            return None
            # raise ValueError, "Found no inverse (alpha-xc)/c: alpha=%s, xc=%s, c=%s !" %(alpha,xc,c)
        res = c * self._QM.Q(gamma)
        if xc != 0:
            res = res + self._QM.Bi(xc, gamma)
        return res

    # Defines the action of the Weil representation using factorization into product of ST^k_j
    def _action_of_SL2Z_factor(self, A):
        r""" Computes the action of A in SL2Z on self
             Using the factorization of A into S and T's
             and the definition of rho on these elements.
             This method works for any WeilModule but is (obviously) very slow
             INPUT : A in SL2Z
             OUTPUT: [r,fact]
                     r = matrix over CyclotomicField(lcm(level,8))
                     fact = sqrt(integer)
                     rho(A) = fact*r
        """
        if A not in SL2Z:
            raise TypeError("{0} must be an element of SL2Z".format(A))
        # [fak,sgn]=self._factor_sl2z(A)
        sgn, n, fak = factor_matrix_in_sl2z(A)
        # [fak,sgn]=self._factor_sl2z(A)
        fak.insert(0, n)
        [r, fact, M] = self._weil_matrix_from_list(fak, sgn)
        for i in range(2):
            for j in range(2):
                if A[i, j] != M[i, j] and A[i, j] != -M[i, j]:
                    raise ValueError("\n A=\n{0} != M=\n{1}, \n factor={2} ".format(A, M, fak))
        return [r, fact]

    def _weil_matrix_from_list(self, fak, sgn):
        r"""
        INPUT: fak = output from _factorsl2z
                   = [a0,a1,...,ak]
               sgn = +-1
        OUTPUT [r,fact,M]
               M       = sgn*T^a0*S*T^a1*...*S*T^ak
               r*fact  = rho(M)
               r       = matrix in CyclotomicField(lcm(level,8))

        """
        [S, T] = SL2Z.gens()
        M = SL2Z([1, 0, 0, 1])
        Z = SL2Z([-1, 0, 0, -1])
        r = matrix(self._K, self._n)
        ss = self._QM.sigma_invariant()
        ## Do the first T^a
        if sgn == -1:
            r, fac = self._action_of_Z()
            rt, fact = self._action_of_T(fak[0])
            r = r * rt
            fac = fac * fact
            M = Z * T ** fak[0]
        else:
            r, fac = self._action_of_T(fak[0])
            M = T ** fak[0]
        if self._verbose > 2:
            print("M={0}".format(M))
            print("r={0}".format(r))
        A = SL2Z([1, 0, 0, 1])  # Id
        for j in range(1, len(fak)):
            A = A * S * T ** fak[j]
        if sgn == -1 and sigma_cocycle(Z, A) == -1:
            sfak = ss ** 4
        else:
            sfak = 1
            # Now A should be the starting matrix except the first T factor
        fact = 1
        # print "A=\n",A
        for j in range(1, len(fak)):
            rN, fac = self._action_of_STn(fak[j])
            Mtmp = S * T ** fak[j]
            M = M * Mtmp
            A = (Mtmp ** -1) * A
            if (j < len(fak) - 1):
                si = sigma_cocycle(Mtmp, A)
                if si == -1:
                    # sig=Integer(si)
                    sfak = sfak * ss ** 4
            r = r * rN  # *sfak
            #   print "fact=",fact
            fact = fact * self._n
        r = r * sfak
        if self._verbose > 0:
            print("sfak={0}".format(sfak))
        # print "|D|=",self._n
        # print "fact=",fact
        if hasattr(fact, "sqrt"):
            fact = fact.sqrt() ** -1
        else:
            fact = sqrt(fact) ** -1
        return [r, fact, M]

    def _get_oddity(self, p, n, r, ep, t):
        r"""
        return the oddity of the Jordan block q_t^(r*ep) where q = p^n
        """
        if n == 0 or p == 1:
            return 0
        k = 0
        if n % 2 != 0 and ep == -1:  # q not a square and sign=-1
            k = 1
        else:
            k = 0
        if t:
            odt = (t + k * 4) % 8
        else:
            odt = (k * 4) % 8
        return odt

    def _get_pexcess(self, p, n, r, ep):
        r"""
        return the oddity of the corresponding Jordan block
        q = p^n  and the module is q^(r*ep)
        """
        if (n == 0 or p == 1):
            return 0
            # if( n % 2 !=0  and ep==-1 ):  # q is a square
        if (is_odd(n) and ep == -1):  # q is a square
            k = 1
        else:
            k = 0
        exc = (r * (p ** n - 1) + 4 * k) % 8
        return exc

    @cached_method
    def lin_comb(self, a, d, b):
        x = self._QM(self._W._elt(a), can_coords=True) + d * self._QM(self._W._elt(b),
                                                                      can_coords=True)
        x = vector(x.list())
        x.set_immutable()
        return self._W._el_index(x)


def sigma_cocycle(A, B):
    r"""
    Computing the cocycle sigma(A,B) using the Theorem and Hilbert symbols

    INPUT:

    -''A'' -- matrix in SL(2,Z)
    -''B'' -- matrix in SL(2,Z)

    OUTPUT:

    -''s'' -- sigma(A,B) \in {1,-1}

    EXAMPLES::


    sage: S,T=SL2Z.gens()


    """
    [a1, b1, c1, d1] = _entries(A)
    [a2, b2, c2, d2] = _entries(B)
    if c2 * c1 != 0:
        C = A * B
        [a3, b3, c3, d3] = _entries(C)
        if c3 != 0:
            # print "here",c3*c1,c3*c2
            return hilbert_symbol(c3 * c1, c3 * c2, -1)
        else:
            return hilbert_symbol(c2, d3, -1)
    elif c1 != 0:
        return hilbert_symbol(-c1, d2, -1)
    elif c2 != 0:
        return hilbert_symbol(-c2, d1, -1)
    else:
        return hilbert_symbol(d1, d2, -1)

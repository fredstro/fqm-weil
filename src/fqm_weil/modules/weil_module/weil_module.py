# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2009 Nils-Peter Skoruppa <nils.skoruppa@uni-siegen.de>
#                          Fredrik Stroemberg <stroemberg@mathematik.tu-darmstadt.de>
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
# *****************************************************************************

r"""
Weil representations for finite quadratic modules.
Implements a class for working with Weil representations and also the explicit formula for general elements of SL(2,Z).


Note: We did not want to implement the metaplectic group so the Weil representation for odd signature is given by the canonical section rho(A)=rho(A,j_A(z))  where j_A(z)=(cz+d)**-1/2 if A = ( a b // c d )
This means that the representation will not be multiplicative, but
$\rho(A)\rho(B)=\sigma(A,B)\rho(AB)$ where the cocycle $\sigma(A,B)$ is implemented as the function sigma_cocycle().





REFERENCES:
 - [St] Fredrik Strömberg, "Weil representations associated with finite quadratic modules", arXiv:1108.0202
 

 AUTHORS:
   - Fredrik Strömberg
   - Nils-Peter Skoruppa
   - Stephan Ehlen


EXAMPLES::

    sage: from fqm_weil.all import FiniteQuadraticModule
    sage: F=FiniteQuadraticModule('5^1')


   
"""
from builtins import range

from sage.all import Integer, RR, CC, QQ, ZZ, cached_method, CyclotomicField, lcm, \
    SL2Z, floor, ceil, hilbert_symbol, xgcd, latex
from sage.matrix.matrix0 import Matrix
from sage.modular.arithgroup.arithgroup_element import ArithmeticSubgroupElement
from sage.structure.formal_sum import FormalSums
from .weil_module_alg import *
from .finite_quadratic_module.finite_quadratic_module_base import FiniteQuadraticModule_base
from .finite_quadratic_module.finite_quadratic_module_element import FiniteQuadraticModuleElement
from .finite_quadratic_module.finite_quadratic_module_ambient import FiniteQuadraticModuleRandom
from .weil_module_element import WeilModuleElement


class WeilModule(FormalSums):
    r"""
    Implements the Weil representation of the metaplectic
    cover $Mp(2,Z)$ or $SL(2,Z)$ of associated to a finite
    quadratic module $A$.
    More precisely, it implements the $K$-vector space $K[A]$
    as $Mp(2,Z)$-module, where $K$ is the $lcm(l,8)$-th cyclotomic field
    if $l$ denotes the level of $A$.
    """

    def __init__(self, A, **kwds):
        r"""
        Initialize the Weil representation associated to the
        nondegenerate finite quadratic Module A.

        INPUT
            A -- an instance of class  FiniteQuadraticModule_base.

        """
        if not isinstance(A, FiniteQuadraticModule_base):
            raise TypeError("Argument must be a nondegenerate Finite Quadratic module. ")
        if not A.is_nondegenerate():
            raise TypeError(" Argument is a degenerate Finite Quadratic module.")
        # Recall that we have elements in CyclotomicField (A.level())
        # times the sigma invariant
        self._verbose = 0
        self._K = CyclotomicField(lcm(A.level(), 8))
        self._QM = A
        FormalSums.__init__(self, base=self._K)
        self._is_WeilModule = True

        self._zl = self._K(CyclotomicField(self._QM.level()).gen())  # e(1/level)
        self._z8 = CyclotomicField(8).gen()  # e(1/8)

        self._n = self._QM.order()
        self._sqn = self._n.sqrt()
        self._gens = self._QM.fgens()
        self._gen_orders = []
        for jj, g in enumerate(self._gens):
            self._gen_orders.append(Integer(g.order()))
        # self._minus_element=[]
        # self._minus_element=self._get_negative_indices()

        self._level = self._QM.level()
        # Pre-compute invariants
        self._inv = self._get_invariants()
        self._zero = WeilModuleElement([(0, self._QM(0))], parent=self)
        self._even_submodule = None
        self._odd_submodule = None
        self._even_submodule_ix = None
        self._odd_submodule_ix = None
        self._basis = []
        self._dim_param = {}
        self._dim_param_numeric = {}
        # self._L = list()
        # e=WeilModuleElement(self,self._QM.list()[0])
        # for ii in range(0,self._n):
        #    self._L.append(WeilModuleElement(self,e))

    def basis(self):
        r"""
        Gives a basis of self as a vector space of dimension |D|
        It is the ordered basis used in self.matrix(A).
        """
        if self._basis == []:
            for i in range(self._QM.order()):
                x = self._QM(self._elt(i), coords='fundamental')
                self._basis.append(WeilModuleElement([(1, x)], parent=self))
        return self._basis

    def _get_negative_indices(self):
        r"""
        Return a list of all indices of negatives of elements (using a method implemented in cython).
        """
        return cython_neg_indices(self._n, self._gen_orders)

    def _get_negative_indices_python(self):
        r"""
        Return a list of all indices of negatives of elements (using a method implemented in python).
        """
        l = []
        for ii in range(self._n):
            l.append(self._neg_index_python(ii))
        return l

    # @cached_method
    def _el_index(self, c):
        r"""
        Return the index of the element c in self.
        """
        if not isinstance(c, (list, Vector_integer_dense)):
            raise ValueError("Need element of list form! Got c={0} of type={1}".format(c, type(c)))
        if not len(c) == len(self._gen_orders):
            raise ValueError(
                "Need element of list of the same length as orders={0}! Got c={1}".format(
                    self._gen_orders, c))
        return cython_el_index(c, self._gen_orders)

    @cached_method
    def _neg_index(self, ii):
        r"""
        Return the index of -1 times the ii-th element of self.
        """
        return cython_neg_index(ii, self._gen_orders)

    def neg_index(self, ii):
        return self._neg_index(ii)

    @cached_method
    def _elt(self, ii):
        r"""
        Return the ii-th element of self.
        """
        return cython_elt(ii, self._gen_orders)

    def zero(self):
        r""" 
        Return the zero element of self.
        """
        return self._zero

    def an_element(self):
        r"""
        Return an element of self.
        """
        return WeilModuleElement(self._QM.an_element(), parent=self)

    def random_element(self):
        r"""
        Return a random element of self.
        """
        return WeilModuleElement(self._QM.random_element(), parent=self)

    ###################################
    ## Introduce myself ...
    ###################################

    def _latex_(self):
        r"""
        EXAMPLES
        """
        return 'Weil module associated to %s' % latex(self._QM)

    def _repr_(self):
        r"""
        EXAMPLES
        """
        return "Weil module associated to {0}".format(self._QM)

    ###################################
    ## Coercion
    ###################################

    def __call__(self, x):
        r"""
        Coerce object into an appopriate child object
        of self if possible.

        We ccoerce
        - an element of $A$ into an elemnt of $K[A]$.

        EXAMPLES
        """
        # print "x=",x
        # print "par=",x.parent()
        if isinstance(x, FiniteQuadraticModuleElement):
            if x.parent() is self._QM:
                return WeilModuleElement([(1, x)], parent=self)

        # otherwise we assume that x is a list of pairs (a,k),
        # where a is in self.__QM and k is in self.__K
        # print "x=",x
        return WeilModuleElement(x, parent=self)

        raise TypeError("argument must be an element of {0}".format(self._QM))

    ###################################
    ## Basic Methods ...
    ###################################

    def matrix(self, A, filter=None, by_factoring=False, **kwds):
        r"""
        Return the matrix rho(A) giving the action of self on $K[A]$.

        INPUT:
        
        -''A''
        -''filter'' -- integer matrix of the same size as the Weil representation. Describes which elements to compute. Only relevant if using formula
        -''by_factoring'' -- logical (default False). If set to True we use the definition of the Weil representation together with factoring the matrix A.  
        - 'kwds' set for example:

            -'numeric' -- integer. set to 1 to return a matrix_complex_dense

        OUTPUT:
        --''[r,f]'' -- r =  n x n matrix over self._K describing
                       f =  sqrt(len(Dc))/sqrt(len(D))
                       r*f = rho(A)

        EXAMPLES::

        sage: from fqm_weil.all import FiniteQuadraticModule, WeilModule
        sage: F = FiniteQuadraticModule([3,3],[0,1/3,2/3])
        sage: W = WeilModule(F)
        sage: A = SL2Z([1,2,1,3])
        sage: [r,f]=W.matrix(A)
        """
        prec = kwds.get('prec', 0)
        if prec > 0:
            return weil_rep_matrix_mpc(self, A[0, 0], A[0, 1], A[1, 0], A[1, 1], filter=filter,
                                       prec=prec, verbose=self._verbose)
        # We only need the diagonal elements of rho(A)
        n = len(list(self._QM))
        # Need a WeilModuleElement to compute the matrix
        e = WeilModuleElement([(1, self._QM.gens()[0])], self, verbose=self._verbose)
        # print "e=",e
        if (by_factoring == False):
            if filter != None:
                [r, fac] = e._action_of_SL2Z_formula(A, filter, **kwds)
            else:
                [r, fac] = e._action_of_SL2Z_formula(A, **kwds)
        else:
            [r, fac] = e._action_of_SL2Z_factor(A)
        return [r, fac]

    def trace(self, A):
        r"""
        Return the trace of the matrix A in Mp(2,Z) or SL(2,Z)
        as endomorphism of $K[A]$.

        EXAMPLES::

        sage: from fqm_weil.all import FiniteQuadraticModule, WeilModule
        sage: F = FiniteQuadraticModule([3,3],[0,1/3,2/3])
        sage: W = WeilModule(F)
        sage: A = SL2Z([1,2,1,3])
        sage: W.trace(A)
         [3, 1/3]
        """
        # We only need the diagonal elements of rho(A)
        n = len(list(self._QM))
        filter = MatrixSpace(ZZ, n).identity_matrix()
        # Need a WeilModuleElement to compute the matrix
        e = self._zero
        [r, fac] = e._action_of_SL2Z_formula(A, filter)
        s = 0
        for j in range(0, n):
            s = s + r[j, j]
        # print "s=",s,fac
        return [s, fac]

    def trace_all(self):
        r"""
        Return a list of "all" traces
        I.e. W.trace([a,b,c,d]) where 0<=c<=Q.level() and d mod c
        """
        l = list()
        for c in range(self._QM.level()):
            for d in range(c):
                # we only want d=0 if c=1
                if d == 0 and c != 1:
                    continue
                elif d == 0 and c == 1:
                    a = 0;
                    b = 1
                else:
                    [g, b, a] = xgcd(c, d)
                    if g != 1:
                        continue
                [t, f] = self.trace([a, -b, c, d])
                # print "a,b,c,d:trace=",a,-b,c,d,':',t
                l.append(t)
        return [l, f]

    def _get_invariants(self):
        r"""
        Compute total invariants of a Jordan decomposition

        OUTPUT: dictionary res with entries:
            res['total p-excess']= sum of all p-excesses
            res['total oddity']  = sum of all oddities
            res['oddity']        = list of oddities of all components
            res['p-excess']=     = list of p-excesses of all components
            res['signature']     = signature := (odt-pexc_tot) % 8
            res['type']          = dictionary q=>'I' or 'II' telling whether the 2-adic component q is odd or even

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, WeilModule
            sage: F = FiniteQuadraticModule([3,3],[0,1/3,2/3])
            sage: W = WeilModule(F)
            sage: W._get_invariants()
            {'oddity': {},
             'p-excess': {3: 0},
             'sigma': {0: 1, 3: 1},
             'signature': 0,
             'total oddity': 0,
             'total p-excess': 0,
             'type': {}}

            sage: R = matrix([[1,0],[3,5]])
            sage: G = matrix([[2/5,1/5],[1/5,3/5]])
            sage: F = FiniteQuadraticModule(R,G)
            sage: W = WeilModule(F)
            sage: W._get_invariants()
            {'oddity': {},
             'p-excess': {5: 4},
             'sigma': {0: -1, 5: -1},
             'signature': 4,
             'total oddity': 0,
             'total p-excess': 4,
             'type': {}}

            sage: A = FiniteQuadraticModule('2^2.4_1^1.3^2')
            sage: W = WeilModule(A)
            sage: W._get_invariants()
            {'oddity': {2: 0, 4: 1},
             'p-excess': {3: 4},
             'sigma': {0: zeta8^3, 2: -zeta8^3, 3: -1},
             'signature': 5,
             'total oddity': 1,
             'total p-excess': 4,
             'type': {2: 'II', 4: 'I'}}
            """
        pexc = dict()
        odts = dict()
        res = dict()
        types = dict()
        sigma_inv = dict()
        sigma_inv[0] = self._QM.sigma_invariant()
        for comp in self._QM.jordan_decomposition():
            q = comp.q
            if comp.p > 2:
                pexc[q] = comp.signature()
            else:
                odts[q] = comp.signature()
            if comp.is_type_I():
                types[q] = "I"  # odd
            if comp.is_type_II():
                types[q] = "II"  # even
            sigma_inv[comp.p] = self._QM.sigma_invariant(comp.p)
        pexc_tot = sum(pexc.values())
        odt = sum(odts.values())
        res['total p-excess'] = pexc_tot % 8
        res['total oddity'] = odt % 8
        res['oddity'] = odts
        res['p-excess'] = pexc
        res['signature'] = (odt - pexc_tot) % 8
        res['type'] = types
        res['sigma'] = sigma_inv
        return res

    def signature(self):
        return self._inv['signature']

    def invariant(self, s):
        if s in self._inv:
            return self._inv[s]
        raise ValueError("Invariant {0} is not defined! Got:{1}".format(s, list(self._inv.keys())))

    def finite_quadratic_module(self):
        return self._QM

    def rank(self):
        return self._n

    def oddity(self):
        return self._inv['oddity']

    def level(self):
        return self._level

    def pexcess(self, p):
        return self._inv['p-excess'].get(p, 0)

    def odd_submodule(self, indices=0):
        if indices == 0:
            return self._symmetric_submodule(sign=-1)
        else:
            return self._symmetric_submodule_ix(sign=-1)

    def even_submodule(self, indices=0):
        if indices == 0:
            return self._symmetric_submodule(sign=1)
        else:
            return self._symmetric_submodule_ix(sign=1)

    def _symmetric_submodule(self, sign=1):
        r"""
        Compute the submodule of self which is C-spanned by
        e_{gamma} + sign*e_{-gamma} for gamma in self._QM
        """
        if sign == 1 and self._even_submodule != None:
            return self._even_submodule
        if sign == -1 and self._odd_submodule != None:
            return self._odd_submodule
        if sign == 1:
            basis = [self(self.finite_quadratic_module()(0))]
        else:
            basis = []
        elts = [self.finite_quadratic_module()(0)]
        for x in self.finite_quadratic_module():
            if -x in elts or x in elts: continue
            elts.append(x)
            w1 = self(x);
            w2 = self(-x)
            if sign == 1:
                if w1 == w2:
                    f = w1
                else:
                    f = w1 + w2
            else:
                f = w1 - w2
            # print "f0=",f,type(f),f==self.zero()
            if f == self.zero(): continue
            # print "f1=",f,type(f),f==self.zero()
            if f not in basis:
                basis.append(f)
        if sign == 1:
            self._even_submodule = basis
        else:
            self._odd_submodule = basis
        return basis

    def _symmetric_submodule_ix(self, sign=1):
        r"""
        Compute the submodule of self which is C-spanned by
        e_{gamma} + sign*e_{-gamma} for gamma in self._QM
        """
        if sign == 1 and self._even_submodule_ix != None:
            return self._even_submodule_ix
        if sign == -1 and self._odd_submodule_ix != None:
            return self._odd_submodule_ix
        if sign == 1:
            basis = [0]
        else:
            basis = []
        el = list(self.finite_quadratic_module())
        elts = [el[0]]
        for x in self.finite_quadratic_module():
            if -x in elts or x in elts: continue
            elts.append(x)
            w1 = self(x);
            w2 = self(-x)
            if sign == 1:
                if w1 == w2:
                    f = w1
                else:
                    f = w1 + w2
            else:
                f = w1 - w2
            # print "f0=",f,type(f),f==self.zero()
            if f == self.zero(): continue
            # print "f1=",f,type(f),f==self.zero()
            fi = el.index(x)
            if fi not in basis:
                basis.append(fi)
        if sign == 1:
            self._even_submodule_ix = basis
        else:
            self._odd_submodule_ix = basis
        return basis

    def dim_parameters(self):
        r"""
        Compute and store parameters which are used in the dimension formula.
        """
        if self._dim_param != {}:
            return self._dim_param
        res = {}
        F = self.finite_quadratic_module()
        K = CyclotomicField(24)
        s2, d2 = F.char_invariant(2)
        s1, d1 = F.char_invariant(-1)
        s3, d3 = F.char_invariant(3)
        # d1 = K(d1**2).sqrt()
        d2 = K(d2 ** 2).sqrt()
        d3 = K(d3 ** 2).sqrt()
        sq3 = K(3).sqrt()
        sqd = K(self.rank()).sqrt()
        # if CC(d1).real()<0: d1 = -d1
        if CC(d2).real() < 0: d2 = -d2
        if CC(d3).real() < 0: d3 = -d3
        if CC(sq3).real() < 0: sq3 = -sq3
        if CC(sqd).real() < 0: sqd = -sqd
        # s1 = s1
        s2 = s2  # *d2
        s3 = s3  # *d3
        ## The factors which are not contained in the ground field are added later
        res['f1'] = 1
        res['f2'] = 1
        res['f3'] = 1
        if d2 in K:
            s2 = s2 * d2
        else:
            res['f2'] = res['f2'] * d2
        if d3 in K:
            s3 = s3 * d3
        else:
            res['f3'] = res['f3'] * d3
        if sq3 in K:
            s1 = s1 / sq3
            s3 = s3 / sq3
        else:
            res['f1'] = res['f1'] / sq3
            res['f3'] = res['f3'] / sq3
        if sqd in K:
            s2 = s2 * sqd
            s3 = s3 * sqd
        else:
            res['f2'] = res['f2'] * sqd
            res['f3'] = res['f3'] * sqd
        res['sq3'] = sq3
        res['s1'] = s1  # /sq3
        res['s2'] = s2  # *sqd
        res['s3'] = s3  # *sqd/sq3

        for ep in [-1, 1]:
            res[ep] = {}
            W = self._symmetric_submodule(ep)
            dim = len(W)
            res[ep]['dim'] = dim
            Qv = [];
            Qv2 = []
            for x in W:
                qx = F.Q(x[0][1])  ## We only take one representative
                if qx >= 1 or qx < 0:
                    qx = qx - floor(qx)
                qx2 = 1 - qx  ## For the dual rep.
                if qx2 >= 1 or qx2 < 0:
                    qx2 = qx2 - floor(qx2)
                Qv.append(qx)
                Qv2.append(qx2)
            K0 = Qv.count(0)
            res[ep]['K0'] = K0
            res[ep]['Qv'] = {1: Qv, -1: Qv2}
            res[ep]['parabolic'] = {1: K0 + sum(Qv), -1: K0 + sum(Qv2)}
        self._dim_param = res
        return res

    def dim_parameters_numeric(self):
        r"""
        Compute and store parameters which are used in the dimension formula.
        """
        if self._dim_param_numeric != {}:
            return self._dim_param_numeric
        res = {}
        F = self.finite_quadratic_module()
        # K = CyclotomicField(24)
        s2, d2 = F.char_invariant(2)
        s1, d1 = F.char_invariant(-1)
        s3, d3 = F.char_invariant(3)
        s1 = CC(s1)
        s2 = CC(s2)
        s3 = CC(s3)
        # d1 = K(d1**2).sqrt()
        d2 = RR(d2);
        d3 = RR(d3);
        sq3 = RR(3).sqrt()
        sqd = RR(self.rank()).sqrt()
        # s2 = s2 #*d2
        # s3 = s3 #*d3

        res['f1'] = 1;
        res['f2'] = 1;
        res['f3'] = 1
        res['s1'] = s1 / sq3
        res['s2'] = s2 * d2 * sqd
        res['s3'] = s3 * d3 / sq3 * sqd
        res['sq3'] = sq3

        for ep in [-1, 1]:
            res[ep] = {}
            W = self._symmetric_submodule(ep)
            dim = len(W)
            res[ep]['dim'] = dim
            Qv = [];
            Qv2 = []
            for x in W:
                qx = F.Q(x[0][1])  ## We only take one representative
                if qx >= 1 or qx < 0:
                    qx = qx - floor(qx)
                qx2 = 1 - qx  ## For the dual rep.
                if qx2 >= 1 or qx2 < 0:
                    qx2 = qx2 - floor(qx2)
                Qv.append(qx)
                Qv2.append(qx2)
            K0 = Qv.count(0)
            res[ep]['Qv'] = {1: Qv, -1: Qv2}
            res[ep]['K0'] = K0
            res[ep]['parabolic'] = {1: K0 + sum(Qv), -1: K0 + sum(Qv2)}
        self._dim_param_numeric = res
        return res

    def dimension_mod_forms(self, k, sgn=0, verbose=0, numeric=False):
        d, ep = self.dimension_cusp_forms(k=k, sgn=sgn, verbose=verbose, numeric=numeric)
        if numeric:
            K0 = self.dim_parameters_numeric()[ep]['K0']
        else:
            K0 = self.dim_parameters()[ep]['K0']
        return d + K0, ep

    def dimension_cusp_forms(self, k, sgn=0, verbose=0, numeric=False):
        r"""
        Compute the dimension of cusp forms weight k with representation self (if sgn=1) or the dual of self (if sgn=-1)
        If numeric = False we work with algebraic numbers.

        INPUT:

        - `k`      -- integer. the weight
        - `sgn`     -- integer. sgn=1 for the Weil representation and -1 for the dual
        - `verbose` -- integer
        - `numeric` -- bool. set to True to use floating-point numbers (much faster) instead of algebraic numbers.

        OUTPUT:

        -`d,ep` -- tuple. dimension of S^{ep}(rho,k) where rho is the Weil representation or its dual and ep is the symmetry of the space.

        """
        if k <= 2:
            raise ValueError("Only weight k>2 implemented! Got k={0}".format(k))
        s = (sgn * self.signature()) % 8
        t = (2 * k - s) % 4
        if verbose > 0:
            print("k = {0}, (2k-{1}) % 4 = {2}".format(k, s, t))
        ep = 0
        if t == 0: ep = 1
        if t == 2: ep = -1
        if ep == 0: return 0, 0
        s = self.signature()
        if numeric:
            par = self.dim_parameters_numeric()
        else:
            par = self.dim_parameters()

        s1 = par['s1'];
        s2 = par['s2'];
        s3 = par['s3']
        if sgn == -1:
            s1 = s1.conjugate();
            s2 = s2.conjugate();
            s3 = s3.conjugate();
            s = -s
        F = self.finite_quadratic_module()
        if not numeric:
            K = CyclotomicField(24)
            z24 = K.gens()[0]
            z8 = z24 ** 3
            identity_term = QQ(par[ep]['dim'] * (k + 5)) / QQ(12)
            six = QQ(6);
            eight = QQ(8)
        else:
            twopi_24 = RR.pi() / RR(12)
            z24 = CC(0, twopi_24).exp()
            twopi_8 = RR.pi() / RR(4)
            z8 = CC(0, twopi_8).exp()
            identity_term = RR(par[ep]['dim'] * (k + 5)) / RR(12)
            six = RR(6);
            eight = RR(8)
        arg = (2 * k - s) % 8
        elliptic_term_e2 = (s2 * ep + s2.conjugate()) * z8 ** arg / eight
        elliptic_term_e2 = elliptic_term_e2 * par['f2']
        arg = (4 * k + 2 - 3 * s) % 24
        if verbose > 1:
            print("z8={0}".format(z8, CC(z8)))
            print("s1={0}".format(CC(s1)))
            print("s2={0}".format(CC(s2)))
            print("s3={0}".format(CC(s3)))
        zf = z24 ** arg
        e31 = s1 * zf
        e32 = s3 * zf
        e311 = e31 + e31.conjugate()
        e322 = e32 + e32.conjugate()
        e311 *= par['f1']
        e322 *= par['f3']
        e3 = (e311 + ep * e322) / six
        # e3 = (s1+ep*s3) * zf
        elliptic_term_e3 = e3  # (e3+e3.conjugate())/6

        parabolic_term = - par[ep]['parabolic'][sgn]
        if verbose > 0:
            print("ep={0}".format(ep))
            print("sgn={0}".format(sgn))
            print("signature={0}".format(s))
            print("Id = {0}".format(identity_term))
            print("E(2)={0}".format(elliptic_term_e2, CC(elliptic_term_e2)))
            print("E(3)={0}".format(elliptic_term_e3, CC(elliptic_term_e3)))
            print("P={0}".format(parabolic_term))
        res = identity_term + elliptic_term_e2 + elliptic_term_e3 + parabolic_term
        ## We want to simplify as much as possible but "res.simplify is too slow"
        if hasattr(res, "coefficients") and not numeric:
            co = res.coefficients()
            if len(co) > 1 or co[0][1] != 0:
                raise ArithmeticError(
                    "Got non-real dimension: d={0} = {1}".format(res, res.simplify()))
            return co[0][0], ep
        else:
            if not numeric:
                return res, ep
            if abs(res.imag()) > 1e-10:
                raise ArithmeticError("Got non-real dimension: {0}".format(res))
            d1 = ceil(res.real() - 1e-5);
            d2 = floor(res.real() + 1e-5)
            if d1 != d2:
                raise ArithmeticError("Got non-integral dimension: d={0}".format(res))
            return d1, ep

    def Zemel_basis_vector(self, H, eta, xi):
        r"""
        Return the vector a_{eta,xi}^{H}=1/sqrt{|H|}\sum_{gamma in H} e(B(gamma,eta))e_{xi+gamma}

        These vectors form an ON basis according to Zemel, "Integral Bases and Invariant Vectors for Weil Representations"

        INPUT:

        - `H` -- subgroup of self
        - `eta` -- element of self
        - `xi` -- element of self


        """




# def WeilRepresentation(FQM):
#     r"""
#     Construct a dummy element of WeilModule. This is useful to extract informaiton about the Weil representation without constructing an actual element.
#     """


#### End of WeilModule Element
def _entries(*args):
    r"""
    Returns the entries of A
    where A is one of:
    1) Element of SL2Z
    2) 2x2 integer matrix with determinant 1
    3) list [a,b,c,d] with ad-bc=1

    EXAMPLES::

        sage: from fqm_weil.modules.weil_module import _entries
        sage: _entries(1, 2, 3, 4)
        (1, 2, 3, 4)
        sage: _entries(matrix([[1, 2], [3, 4]]))
        (1, 2, 3, 4)
        sage: _entries(SL2Z.gens()[0])
        (0, -1, 1, 0)

    """
    if len(args) == 4:
        return args
    if isinstance(args[0], (tuple, list)) and len(args[0]) == 4:
        return args[0]

    if isinstance(args[0], ArithmeticSubgroupElement):
        return tuple(args[0])
    if isinstance(args[0], Matrix):
        return args[0][0, 0], args[0][0, 1], args[0][1, 0], args[0][1, 1]
    raise ValueError(f"Can not create SL(2,Z) entries from {args}")


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


def hilbert_symbol_infinity(a, b):
    if (a < 0 and b < 0):  # or (a == 0 and b < 0) or (a < 0 and b == 0):
        return -1
    return 1


def kubota_cocycle(A, B):
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
    C = A * B
    [a3, b3, c3, d3] = _entries(C)
    sA = kubota_sigma_symbol(c1, d1)
    sB = kubota_sigma_symbol(c2, d2)
    sC = kubota_sigma_symbol(c3, d3)
    res = hilbert_symbol(sA, sB, -1) * hilbert_symbol(-sA * sB, sC, -1)
    return res


def kubota_sigma_symbol(c, d):
    r"""
    Compute sigma_A=sigma(c,d) for A = (a b // c d)
    given by sigma_A = c if c!=0 else = d
    """
    if c != 0:
        return c
    else:
        return d


def test_dimensions_1(fqbound=100, nbound=10, cbound=10, size_bd=50, kmin=2, kmax=10, verbose=0,
                      test=1, numeric=False):
    r"""
    Run tests on a set of random quadratic modules
    test >= 0: make sure only exactly one of dim(r) and dim(r*) arenon-zero
    Test >= 1: check that dimensions are integers 

    
    """
    for i in range(nbound):
        l = size_bd + 1
        while l > size_bd:
            FQ = FiniteQuadraticModuleRandom(fqbound, nbound, verbose - 1)
            l = len(list(FQ))
            W = WeilModule(FQ)
            g = FQ.jordan_decomposition().genus_symbol()
            s = W.signature()
            if verbose > 0:
                print("Check genus={0} sign={1}".format(g, s))
            for twok in range(2 * kmin + 1, 2 * kmax + 1):
                k = QQ(twok) / QQ(2)
                if verbose > 1:
                    print("k={0}".format(k))
                ## There is a built-in integrality test already
                try:
                    d1, ep1 = W.dimension_cusp_forms(k, sgn=1, verbose=verbose - 2,
                                                     numeric=numeric)
                    d2, ep2 = W.dimension_cusp_forms(k, sgn=-1, verbose=verbose - 2,
                                                     numeric=numeric)
                except ArithmeticError:
                    print("Fail 2")
                    print("k={0}".format(k))
                    print("genus:{0}".format(g))
                    print("FQ={0}".format(FQ))
                    print("dimS(rho,{0})={1}".format(k, CC(d1)))
                    print("dimS(rho*,{0})={1}".format(k, CC(d2)))
                    return False, W, d1, d2
    return True


### testing functions
def test_formula(fqbound=100, nbound=10, cbound=10, size_bd=50, kmin=2, kmax=10, verbose=0, test=1,
                 numeric=False, gamma0_test=0):
    r"""
    Run tests on a set of random quadratic modules
    test >= 0: make sure only exactly one of dim(r) and dim(r*) arenon-zero
    Test >= 1: check that dimensions are integers 

    
    """
    for i in range(nbound):
        l = size_bd + 1
        while l > size_bd:
            FQ = FiniteQuadraticModuleRandom(fqbound, nbound, verbose - 1)
            if list(FQ) == [FQ(0)]:
                continue  ## This is just the 0 module
            l = len(list(FQ))
            W = WeilModule(FQ)
            g = FQ.jordan_decomposition().genus_symbol()
            s = W.signature()
        if verbose > 0:
            print("signature={0}".format(s))
            print("genus={0}".format(g))
            print("is_nondeg={0}".format(FQ.is_nondegenerate()))
            print("gram={0}".format(FQ.gram()))
        w = W.an_element()
        i = 0
        j = 0
        while (i < cbound):
            A = SL2Z.random_element()
            if gamma0_test == 1 and A[1, 0] % W.level() != 0:
                j += 1
                if j > 2000:
                    raise ArithmeticError("Error in random elements of SL2Z!")
                continue
            j = 0
            i = i + 1
            t = compare_formula_for_one_matrix(W, A, verbose)
            if not t:
                if verbose > 0:
                    print("A={0}".format(A))
                    return W, A

                    # raise ArithmeticError,"Got different matrix! r1={0} and r2={1}".format(r1,r2)
    return True


def compare_formula_for_one_matrix(W, A, verbose=0):
    r"""
    Compare the action of A on W using the formula and factoring into generators.
    """
    r1 = W.matrix(A, by_factoring=False)
    r2 = W.matrix(A, by_factoring=True)
    f1 = CC(r1[1])
    f2 = CC(r2[1])
    for i in range(r1[0].nrows()):
        for j in range(r1[0].ncols()):
            t1 = f1 * r1[0][i, j].complex_embedding()
            t2 = f2 * r2[0][i, j].complex_embedding()
            if abs(t1 - t2) > 1e-10:
                if verbose > 0:
                    print("i,j={0},{1}".format(i, j))
                    print("t1={0}".format(t1))
                    print("t2={0}".format(t2))
                return False
    return True


def compare_formula_for_one_module(W, nmats=10, verbose=0):
    r"""
    Compare the action of A on W using the formula and factoring into generators.
    """
    for i in range(nmats):
        A = SL2Z.random_element()
        t = compare_formula_for_one_matrix(W, A, verbose)
        if not t:
            return False
    return True

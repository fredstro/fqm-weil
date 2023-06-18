from sage.arith.functions import lcm
from sage.arith.misc import gcd
from sage.groups.additive_abelian.additive_abelian_group import AdditiveAbelianGroupElement
from sage.misc.latex import latex
from sage.modules.fg_pid.fgp_element import FGP_Element
from sage.modules.free_module_element import vector
from sage.modules.vector_integer_dense import Vector_integer_dense
from sage.rings.integer import Integer
from sage.structure.element import AdditiveGroupElement, Vector


# class FiniteQuadraticModuleElement(AdditiveGroupElement, FGP_Element):
class FiniteQuadraticModuleElement(FGP_Element):
    r"""
    Describes an element of a quadratic module.

    EXAMPLES NONE
##         sage: p = FiniteQuadraticModule([3,3])
##         sage: g0,g1 = p.gens()
##         sage: x = g0 + g1*5; x
##         e0 + 5*e1
##         sage: -x
##         5*e0 + e1
##         sage: x*1001 + g1
##         5*e0 + 2*e1
##         sage: p.Q(x)
##         1/6

    NOTES::

        Code partly grabbed from sage.structure.element.AbelianGroupElement.
    """

    def __init__(self, A, x, check=False, **kwds):
        r"""
        Create the element of the FiniteQuadraticModule_base A whose coordinates
        with respect to the generators ('fundamental' or 'canonical',
        dependent on the parameter 'coords') of $A$
        are given by the list x; create the zero element if $x=0$.

        INPUT:

            ``A`` -- quadratic module
            ``x`` -- list or 0
            ``coords`` -- string ('canonical' or 'fundamental', default: 'fundamental')

        OUTPUT:

            an element of the quadratic module A

        EXAMPLES NONE
##             sage: q = FiniteQuadraticModule([2,4]);
##             sage: x = FiniteQuadraticModuleElement(q, [1,1])
##             sage: x
##             e0 + e1
        """
        from .finite_quadratic_module_base import FiniteQuadraticModule_base
        if not isinstance(A, FiniteQuadraticModule_base):
            raise TypeError("Argument 'A' must be a finite quadratic module")
        self.__repr = None
        # The incoming coordinates can be either canonical or fundamental coordinates
        # but the internal representation uses only the fundamental coordinates.
        # coords = kwds.get('coords') or 'canonical'
        # if coords not in ['canonical', 'fundamental']:
        #     raise ValueError(f"Invalid option for 'coords': {coords}")
        # self._default_coords = coords
        n_fundamental = A.ngens(coords='fundamental')
        n_canonical = A.ngens(coords='canonical')
        if isinstance(x, (int, Integer)) and x == 0:
            self.__intl_rep = [0] * n_fundamental
        elif isinstance(x, (list, tuple, Vector)):
            # We need fundamental coordinates for the intl_rep
            # The input vector is either a vector of canonical coordinates
            # or in case
            # print("kwds=",kwds)
            # print("A def coords=",A._default_coords)
            # print("lenx=",len(x))
            # print("n fund=",n_fundamental)
            # print("n can=",n_canonical)
            # print("X0=",x)

            # if len(x) == n_fundamental and (A._default_coords == 'fundamental' or
            #                                 kwds.get('coords', 'canonical') == 'canonical'):
            #     v = x
            # elif len(x) == n_canonical:
            #     v = A._c2f(x)
            # elif len(x) != n_fundamental:
            #     raise ValueError(f"Input vector has wrong length {len(x)} != {n_fundamental}")
            # else:
            #     v = x

            # if A._default_coords == 'canonical' and len(x) == A.ngens(coords='canonical'):
            #     x = A._c2f(x)
            # if len(x) != n_fundamental:
            #      raise ValueError(f"Input vector has wrong length {len(x)} != {n_fundamental}")

            # We either have
            # 1. kwds['coords'] = fundamental: then x is supposed to have correct length
            # 2. kwds['coords'] = None or canonical: If len(x)=n_canonical we assume canonical coords.
            coords = kwds.get('coords', 'canonical')
            if coords is None or coords == 'canonical' and len(x) == n_canonical:
                x = A._c2f(x)
            # print("X1=",x)
            self.__intl_rep = [x[i] % f for i, f in enumerate(A.elementary_divisors())]
        else:
            raise TypeError("Argument x (= {0}) is of wrong type.".format(x))
        # Canonical internal representation for parent
        # print("A._R=",A._R())
        # print("A._E=",A._E())
        # print("A._gens_orders=",A._gens_orders)
        # print("A def coord=",A._default_coords)
        # print("intl rep_f=",self.__intl_rep)
        self.__intl_rep_canonical = A._f2c(self.__intl_rep)
        # print("intl rep_c=", self.__intl_rep_canonical)
        if A._default_coords == 'fundamental':
            rep = self.__intl_rep
        else:
            rep = self.__intl_rep_canonical
        self.__vector = vector(rep)
        self.__vector.set_immutable()
        FGP_Element.__init__(self, A, A._V(rep))


    # def _cache_key(self):
    #     return ('FiniteQuadraticModuleElement', tuple(self.__intl_rep), self.parent())
    #
    # def __hash__(self):
    #     return hash(self._cache_key())
    ###################################
    # Introduce myself ...
    ###################################

    def _latex_(self):
        r"""
        EXAMPLES NONE
        """
        s = ""
        A = self.parent()
        x = A.variable_names()
        n = len(A.variable_names())
        v = self.list(coords=A._default_coords)
        for i in range(n):
            if v[i] == 0:
                continue
            elif v[i] == 1:
                if len(s) > 0:
                    s += " + "
                s += "%s" % x[i]
            else:
                if len(s) > 0:
                    s += " + "
                s += r"%s \cdot %s" % (latex(v[i]), x[i])
        if len(s) == 0:
            s = "0"
        return s

    def _repr_(self):
        r"""
        EXAMPLES NONE
        """
        s = ""
        A = self.parent()
        x = A.variable_names()
        n = len(A.variable_names())
        v = self.list(coords=A._default_coords)
        # print("variable names=",x)
        # print("n=",n)
        # print("v=",v)
        for i in range(n):
            if v[i] == 0:
                continue
            elif v[i] == 1:
                if len(s) > 0:
                    s += " + "
                s += "%s" % x[i]
            else:
                if len(s) > 0:
                    s += " + "
                s += "%s*%s" % (v[i], x[i])
        if not s:
            s = "0"
        return s

    ###################################
    # Providing struct. defining items
    ###################################

    def list(self, coords='fundamental'):
        r"""
        Return the coordinates of self w.r.t. the fundamental
        generators of self.parent() as a list.

        EXAMPLES NONE
        """
        if coords == 'canonical':
            return self.parent()._f2c(self.__intl_rep)
        return self.__intl_rep

    def c_list(self):
        r"""
        Return the coordinates of self w.r.t. the 'standard' generators of self.parent() as a list.

        EXAMPLES NONE
        """
        return self.parent()._f2c(self.__intl_rep)

    def __hash__(self):
        return super(FiniteQuadraticModuleElement,self).__hash__()

    def _vector_(self):
        r"""
        Return the coordinates of self w.r.t. the fundamental
        generators of self.parent() as a vector.

        EXAMPLES NONE
        """
        return vector(self.list(coords='fundamental'), immutable=False)

    ###################################
    # Operations
    ###################################

    # def _add_(self, y):
    #     r"""
    #     EXAMPLES NONE
    #     """
    #     # Same as _mul_ in FreeAbelianMonoidElement except that the
    #     # exponents get reduced mod the invariant.
    #
    #     invs = self.parent().elementary_divisors()
    #     n = len(invs)
    #     z = self.parent()(0)
    #     xelt = self.__intl_rep
    #     yelt = y.__intl_rep
    #     zelt = [xelt[i] + yelt[i] for i in range(len(xelt))]
    #     if len(invs) >= n:
    #         L = []
    #         for i in range(len(xelt)):
    #             if invs[i] != 0:
    #                 L.append(zelt[i] % invs[i])
    #             if invs[i] == 0:
    #                 L.append(zelt[i])
    #         z.__intl_rep = L
    #     if len(invs) < n:
    #         L1 = []
    #         for i in range(len(invs)):
    #             if invs[i] != 0:
    #                 L1.append(zelt[i] % invs[i])
    #             if invs[i] == 0:
    #                 L1.append(zelt[i])
    #         L2 = [zelt[i] for i in range(len(invs), len(xelt))]
    #         z.__intl_rep = L1 + L2
    #     return z

    def _rmul_(self, c):
        """
        Multiply this FQM element by c.

        """
        return self.__mul__(c)

    def __truediv__(self, c):
        """
        Return an element y (not necessarily unique) in parent so that c*y = self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F=FiniteQuadraticModule([3])
            sage: x = F.gens()[0]
            sage: x
            e


        """
        coords = []
        for i, x in enumerate(self.list()):
            if x == 0:
                coords.append(0)
                continue
            modulus = self.parent().gens_orders()[i]
            t = gcd(c, modulus)
            if t != 1 and x % t != 0:
                raise ValueError(f"This element is not divisuble by '{c}'")
            x = x / t
            s = c // t
            s1 = s.inverse_mod(modulus // t)
            coords.append(x*s1)
        return self.parent()(coords)

    def _richcmp_(self, right, op):
        """
        Compare self and right.

        Note: We reverse the comparison in the FGPElement since we want reverse lexicographical
        ordering to make sure that generators are ordered as e0 < e1 < e2 etc.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F.<a,b,c> = FiniteQuadraticModule([2,12,34])
            sage: a < b < c
            True
            sage: x,y,z = FiniteQuadraticModule('2_1.4^2').gens()
            sage: x < y < z
            True
        """
        from sage.structure.richcmp import op_LT, op_GE, op_GT, op_LE, op_EQ, op_NE, richcmp
        op = {
            op_LT: op_GT,
            op_GT: op_LT,
            op_GE: op_LE,
            op_LE: op_GT,
            op_EQ: op_EQ,
            op_NE: op_NE
        }.get(op)
        # Note that _x is the representation with respect to the initialising coordintate system.
        return richcmp(self._x, right._x, op)

    # def __cmp__(self, other, ):
    #     r"""
    #     EXAMPLES NONE
    #     """
    #     if not isinstance(other, type(self)):
    #         return False
    #     return self.__intl_rep == other.__intl_rep

    def __eq__(self, other):
        r"""
        Test if this FQM Element is equal to other.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F.<a,b,c> = FiniteQuadraticModule([2,12,34]); F
            Finite quadratic module in 3 generators:
             gens: a, b, c
             form: 1/8*x0^2 + 1/48*x1^2 + 1/136*x2^2
            sage: b + a == a + b
            True
            sage: b == a
            False
            sage: b == 'b'
            False
            sage: 'b' == b
            False
        """
        if other == 0:
            return self.__intl_rep == [0]*len(self.__intl_rep)
        if not isinstance(other, type(self)):
            return False
        if self.parent() != other.parent():
            return False
        return self.__intl_rep == other.__intl_rep

    # def __lt__(self, other):
    #     r"""
    #     Test if this FQM Element is less than other.
    #
    #     The comparison is based on lexicographical ordering on the fundamental coordinates.
    #
    #     EXAMPLES::
    #
    #     """
    #     if not isinstance(other, type(self)):
    #         return False
    #     if self.parent() != other.parent():
    #         return False
    #     return self.__intl_rep < other.__intl_rep

    def __ne__(self, other):
        r"""
        Test if this FQM Element is different from other.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F.<a,b,c> = FiniteQuadraticModule([2,12,34]); F
            Finite quadratic module in 3 generators:
             gens: a, b, c
             form: 1/8*x0^2 + 1/48*x1^2 + 1/136*x2^2
            sage: b + a != a + b
            False
            sage: b != a
            True
            sage: b != 'b'
            True
            sage: 'b' != b
            True

        """
        return not self.__eq__(other)

    ###################################
    # Associated quantities
    ###################################

    def order(self):
        r"""
        Returns the order of this element.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F.<a,b,c> = FiniteQuadraticModule([2,12,34]); F
            Finite quadratic module in 3 generators:
             gens: a, b, c
             form: 1/8*x0^2 + 1/48*x1^2 + 1/136*x2^2
            sage: x = a - b
            sage: x.order()
            24
        """
        A = self.parent()
        if self == FiniteQuadraticModuleElement(A, 0):
            return Integer(1)
        invs = A.elementary_divisors()
        L = list(self.__intl_rep)
        N = lcm([invs[i] / gcd(invs[i], L[i]) for i in range(len(invs)) if L[i] != 0])
        return N

    def norm(self):
        r"""
        If this element is $a$ and belongs to the module $(M,Q)$ then
        return $Q(a)$.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F.<a,b,c> = FiniteQuadraticModule([2,12,34]); F
            Finite quadratic module in 3 generators:
             gens: a, b, c
             form: 1/8*x0^2 + 1/48*x1^2 + 1/136*x2^2
            sage: x = a - b
            sage: x.norm()
            7/48
        """
        return self.parent().Q(self)

    def dot(self, b):
        r"""
        If this element is $a$ and belongs a module with associated
        bilinear form $B$ then return $B(a,b)$.

        INPUT:

        - ``b`` -- finite quadratic module element.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F.<a,b,c> = FiniteQuadraticModule([2,12,34]); F
            Finite quadratic module in 3 generators:
             gens: a, b, c
             form: 1/8*x0^2 + 1/48*x1^2 + 1/136*x2^2
            sage: (a+b).dot(a)
            1/4
            sage: (a+b).dot(c)
            0

        """
        return self.parent().B(self, b)

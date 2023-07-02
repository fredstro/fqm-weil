import logging
from sage.all import QQ, ZZ
from sage.arith.misc import is_prime, valuation, kronecker, primitive_root, gcd, is_prime_power
from sage.categories.sets_cat import cartesian_product
from sage.functions.other import floor
from sage.matrix.args import MatrixSpace
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method
from sage.misc.functional import is_odd, is_even
from sage.misc.misc_c import prod, copy
from sage.modules.free_module_element import vector
from sage.rings.integer import Integer
from sage.rings.number_field.number_field import QuadraticField
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.ring_extension_element import PolynomialRing
from sage.arith.misc import divisors

from .finite_quadratic_module_base import FiniteQuadraticModule_base
from .finite_quadratic_module_element import FiniteQuadraticModuleElement
from .finite_quadratic_module_subgroup import FiniteQuadraticModule_subgroup
from .jordan_decomposition import JordanDecomposition, JordanComponent

log = logging.getLogger(__name__)

class FiniteQuadraticModule_ambient(FiniteQuadraticModule_base):
    r"""
    Describes a finite quadratic module $(A,Q)$. The abelian group $A$
    is given by generators \code{(e_0,e_1,\dots)} and a matrix $R$ of relations
    satisfied by the generators (hence $(a,b,\dots)\cdot R = 0$). The
    quadratic form $Q$ is given by the Gram matrix w.r.t. the generators;
    more precisely, the $(i,j)$-th entry $g_{i,j}$ is a rational number
    such that $Q(e_i+e_j)-Q(r_i)-Q(e_j) = g_{i,j} + \ZZ$..

    NOTES::

        The abelian group may also be thought of as $\ZZ^n/R\ZZ^n$,
        and the generators as $e_i = c_i + R\ZZ^n$, where $c_i$ is
        the canonical basis for $\ZZ^n$.

        In our implementation we think of elements of $\ZZ^n$
        as column vectors.

        Use FiniteQuadraticModule() for more flexibility when creating
        FiniteQuadraticModule_base objects.

    EXAMPLES ::

        sage: from fqm_weil.all import FiniteQuadraticModule
        sage: R = matrix(2,2,[2,1,1,2])
        sage: G = 1/2 * R^(-1)
        sage: A.<a,b> = FiniteQuadraticModule(R, G); A
        Finite quadratic module in 2 generators:
         gens: b, b
         form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2
        sage: a == b
        True
        sage: A.<a> = FiniteQuadraticModule(R, G, default_coords='fundamental'); A
        Finite quadratic module in 1 generator:
         gen: a
         form: 1/3*x^2
    """

    Subgroup = FiniteQuadraticModule_subgroup

    def __init__(self, R, G, check=True, names=None, default_coords='canonical'):
        r"""
        Initialize a quadratic module from R and G.

        INPUT:
        - ``R``     -- an integral non-degenerate square matrix of size $n$
                     representing the finite abelian group $\ZZ^n/R\ZZ^n$.

        - ``G``     -- a symmetric rational matrix of same size as $R$ and such
                     that $R^tGR$ is half integral and $2*R*M$ is an
                     integer matrix, representing the quadratic
                     form $x + R\ZZ^n \mapsto G[x] + \ZZ$.


        - ``check`` -- True or False indicating whether R and G should be
                     checked to be a valid set of data for the definition of
                     a quadratic module.

        - ``names`` -- a string used to name the generators of
                     the underlying abelian group.

        - ``default_coords`` -- string 'canonical' or 'fundamental' (default: 'fundamental')
                        decides which coordinates should be used for the string representation.


        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: from fqm_weil.all import FiniteQuadraticModule_ambient
            sage: F = FiniteQuadraticModule('3^-1')
            sage: isinstance(F,FiniteQuadraticModule_ambient)
            True
            sage: F1 = FiniteQuadraticModule_ambient(F.relations(),F.gram())
            sage: F1 == F
            True
            sage: TestSuite(F1).run()

        """
        FiniteQuadraticModule_base.__init__(self, R, G, names=names, default_coords=default_coords)

    #
    # Functions related to subgroups
    #

    def orthogonal_basis(self, p=None):
        r"""
        Return an orthogonal system of generators for the
        underlying group of this quadratic module, if $p$ is None,
        respectively for the $p$-Sylow subgroup if $p$ is a prime.

        NOTES:

            See FiniteQuadraticModule_subgroup.orthogonal_basis()
            for detailed explanation.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b,c,d,e,f,g> = FiniteQuadraticModule('11^-7.2^-2', \
            ....: default_coords='fundamental')
            sage: A.orthogonal_basis (11)
            [a, b, c, d, e, 10*f, 10*g]
            sage: A.orthogonal_basis (2)
            [11*f, 11*g]
            sage: A.<a,b,c,d,e,f,g,h,j> = FiniteQuadraticModule('11^-7.2^-2')
            sage: A.orthogonal_basis (11)
            [a, b, c, d, e, f, g]
            sage: A.orthogonal_basis (2)
            [h, j]
            sage: R.<X>= ZZ['X']
            sage: K.<x> = NumberField(X^10 + X^9 - X^7  - X^6 - X^5 - X^4  - X^3 + X + 1)
            sage: L = FiniteQuadraticModule((1-x)/1001); L
            Finite quadratic module in 10 generators:
             gens: e0, e1, e2, e3, e4, e5, e6, e7, e8, e9
             form: 1/91*x0^2 + 997/1001*x0*x1 + 1000/1001*x1^2 + 999/1001*x0*x2 + 2/1001*x1*x2 + 998/1001*x2^2 + 2/1001*x0*x3 + 995/1001*x1*x3 + 999/1001*x3^2 + 995/1001*x0*x4 + 997/1001*x2*x4 + 10/1001*x3*x4 + 1000/1001*x4^2 + 997/1001*x1*x5 + 10/1001*x2*x5 + 999/1001*x3*x5 + 993/1001*x4*x5 + 997/1001*x5^2 + 997/1001*x0*x6 + 10/1001*x1*x6 + 999/1001*x2*x6 + 993/1001*x3*x6 + 993/1001*x4*x6 + 12/1001*x5*x6 + 993/1001*x6^2 + 10/1001*x0*x7 + 999/1001*x1*x7 + 993/1001*x2*x7 + 993/1001*x3*x7 + 12/1001*x4*x7 + 985/1001*x5*x7 + 8/1001*x6*x7 + 1/1001*x7^2 + 999/1001*x0*x8 + 993/1001*x1*x8 + 993/1001*x2*x8 + 12/1001*x3*x8 + 985/1001*x4*x8 + 8/1001*x5*x8 + 2/1001*x6*x8 + 981/1001*x7*x8 + 1/1001*x8^2 + 993/1001*x0*x9 + 993/1001*x1*x9 + 12/1001*x2*x9 + 985/1001*x3*x9 + 8/1001*x4*x9 + 2/1001*x5*x9 + 981/1001*x6*x9 + 2/1001*x7*x9 + 989/1001*x8*x9 + 4/1001*x9^2
            sage: og_b = L.orthogonal_basis(); og_b #long time
            [77*e0,
             77*e0 + 462*e9,
             77*e0 + 77*e8 + 693*e9,
             77*e0 + 693*e7 + 385*e8 + 231*e9,
             77*e0 + 693*e5 + 154*e7 + 308*e8 + 77*e9,
             77*e0 + 693*e5 + 616*e6 + 231*e7 + 462*e8 + 77*e9,
             77*e0 + 308*e4 + 462*e5 + 539*e6 + 616*e7 + 77*e8 + 462*e9,
             77*e0 + 308*e3 + 154*e4 + 154*e5 + 924*e6 + 770*e7 + 462*e8 + 308*e9,
             77*e0 + 770*e2 + 770*e3 + 231*e4 + 77*e5 + 693*e6 + 924*e7 + 77*e8 + 77*e9,
             77*e0 + 77*e1 + 231*e2 + 308*e3 + 539*e4 + 847*e5 + 231*e6 + 539*e7 + 847*e8 + 385*e9,
             91*e1,
             91*e0 + 455*e9,
             91*e0 + 819*e8 + 637*e9,
             91*e0 + 819*e7 + 455*e8 + 546*e9,
             91*e0 + 273*e6 + 637*e7 + 182*e8 + 455*e9,
             91*e0 + 364*e5 + 364*e6 + 273*e7 + 455*e8 + 455*e9,
             91*e0 + 364*e4 + 91*e5 + 364*e6 + 910*e8 + 455*e9,
             91*e0 + 637*e3 + 182*e5 + 364*e6 + 728*e7 + 728*e8 + 182*e9,
             91*e0 + 91*e2 + 455*e3 + 910*e4 + 728*e5 + 91*e6 + 455*e7 + 455*e8 + 819*e9,
             91*e0 + 455*e1 + 455*e3 + 637*e4 + 364*e5 + 273*e6 + 455*e7 + 728*e8 + 819*e9,
             143*e0,
             143*e1 + 429*e9,
             143*e0 + 715*e8 + 715*e9,
             143*e0 + 858*e7 + 143*e8 + 429*e9,
             143*e0 + 858*e4 + 286*e7 + 572*e8 + 715*e9,
             143*e0 + 858*e4 + 858*e6 + 572*e7 + 429*e8 + 429*e9,
             143*e0 + 429*e3 + 858*e4 + 572*e6 + 143*e7 + 572*e8 + 858*e9,
             143*e0 + 429*e3 + 858*e4 + 143*e5 + 858*e6 + 572*e7 + 572*e8,
             143*e0 + 858*e2 + 429*e3 + 858*e4 + 858*e5 + 429*e6 + 286*e7 + 286*e9,
             143*e0 + 143*e1 + 286*e2 + 143*e3 + 572*e5 + 715*e6 + 143*e7 + 858*e8 + 143*e9]

        TODO:
        - If a basis is already orthogonal then return it.
        - The og basis should correspond to a 'normalized' genus_symbol

        """
        if not self.is_nondegenerate():
            raise ValueError
        if p is None:
            U = self.subgroup(self.gens())
        elif is_prime(p):
            U = self.subgroup(p)
        else:
            raise TypeError
        return U.orthogonal_basis()

    def jordan_decomposition(self):
        r"""
        Jordan Decomposition of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('11^-1')
            sage: list(A.jordan_decomposition())
            [11^-1]
            sage: A = FiniteQuadraticModule([11,33]);
            sage: list(A.jordan_decomposition())
            [2_4^-2, 3, 11^2]
            sage: list(FiniteQuadraticModule('2_1.4^2').jordan_decomposition())
            [2_1, 4^2]
            sage: list(FiniteQuadraticModule('4^2.2_1').jordan_decomposition())
            [2_1, 4^2]

        """
        try:
            return self.__jd
        except AttributeError:
            self.__jd = JordanDecomposition(self)
        return self.__jd

    def is_indecomposable(self):
        """
        Check if self is an indecomposable module.

        Note: Indecomposable modules are either of the form Z/p^kZ for prime p,
        or Z/2^kZ x Z/2^kZ with specific gram matrices.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('11^-1')
            sage: A.is_indecomposable()
            True
            sage: A = FiniteQuadraticModule('11^-1.3^2')
            sage: A.is_indecomposable()
            False


        """
        if len(self._gens_orders) > 2:
           return False
        if len(self._gens_orders) == 1:
            return is_prime_power(self._gens_orders[0])
        q1, q2 = self._gens_orders

        if not is_prime_power(q1) or not is_prime_power(q2):
            return False
        if q1 != q2 or (q1 % 2 == 1 and q1 != 1):
            return False
        # In this situation we have q1 = q2 = 2^k for some k>1
        mat_B = matrix(QQ, 2, [1 / q1, 1 / (2 * q1), 1 / (2 * q1), 1 / q1])
        mat_C = matrix(QQ, 2, [0, 1 / (2 * q1), 1 / (2 * q1), 0])
        return self.gram_matrix() in [mat_B, mat_C]


    def spawn(self, gens, **kwargs):
        r"""
        Spawn the subgroup generated by the elements of the list
        gens equipped with the quadratic form induced by this module as finite
        quadratic module.

        INPUT:
        - `gens` -- generators
        - `names` -- names

        OUTPUT:
        - A tuple (B,f) where B is a finite quadratic module and f: B -> self a homomorphism.

        EXAMPLES::


            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: B,f = A.spawn([A.gens()[0]]); B,f
            (Finite quadratic module in 1 generator:
              gen: e
              form: 1/44*x^2,
             Homomorphism : Finite quadratic module in 1 generator:
              gen: e
              form: 1/44*x^2 --> Finite quadratic module in 2 generators:
              gens: e0, e1
              form: 1/44*x0^2 + 1/132*x1^2
              e |--> e0)
              sage: f(B.gens()[0]) == A.gens()[0]
              True
            sage: f.domain() == B
            True
            sage: f.codomain() == A
            True
        """
        names = kwargs.pop('names', None)
        return self.subgroup(gens).as_ambient(names)

    def quotient(self, U):
        r"""
        Return the quotient module $self/U$ for the isotropic subgroup
        $U$ of self. If $U$ is not isotropic an exception is thrown.

        INPUT:
        - `U` -- a subgroup of self

        OUTPUT:
        - a finite quadratic module

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A1=A.quotient(list(A.isotropic_subgroups())[0]); A1
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2
            sage: A1 == A
            True
            sage: A2=A.quotient(list(A.isotropic_subgroups())[1]); A2
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/33*x0^2 + 1/33*x0*x1 + 1/33*x1^2
            sage: K.<a>=QuadraticField(-7)
            sage: om=19/37*a - 10/37
            sage: A = FiniteQuadraticModule(om / om.parent().gen())
            sage: N = A.kernel()
            sage: A.quotient(N)
            Finite quadratic module in 2 generators:
             gens: 31*e1, e1
             form: 7/37*x0^2 + 10/37*x0*x1 + 30/37*x1^2

        NOTES:

            Let $U^\sharp = K\ZZ^n/M\ZZ^n$ the dual of the subgroup $U =
            H\ZZ^n/M\ZZ^n$ of $M$. The quotient module
            $(U^\sharp/U, x + U \mapsto Q(x) + \ZZ)$
            is then isomorphic to $(\ZZ^n/K^{-1}H\ZZ^n, G[Kx])$.

        TODO: A.quotient(U) should return B, f, g where  B is V/U (V = dual of U),
                f:V-->A and g:V-->B are  the natural morphisms.

        """
        if not isinstance(U, FiniteQuadraticModule_subgroup) or U.ambience() != self \
                or not U.is_isotropic():
            raise ValueError("{0}: not an isotropic subgroup".format(U))
        if U.is_trivial():
            return self
        V = U.dual()
        K = V.gens_matrix()
        R = K ** -1 * U.gens_matrix()
        G = K.transpose() * self.gram() * K
        return FiniteQuadraticModule(R, G)

    def __truediv__(self, U):
        r"""
        Return the quotient of $A/U$.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A1=A / list(A.isotropic_subgroups())[0]; A1
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2
            sage: A1 == A
            True
            sage: A2=A / list(A.isotropic_subgroups())[1]; A2
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/33*x0^2 + 1/33*x0*x1 + 1/33*x1^2
        """
        return self.quotient(U)

    def anisotropic_kernel(self):
        r"""
        Return the anisotropic quotient of this quadratic module,
        i.e. if this module is $A$ then return $A/U$, where $U$
        is a maximal isotropic subgroup.

        OUTPUT:

            (A/U, f, g), where $U$ is a maximal isotropic subgroup,
            and, where $f:(U^#,Q) \rightarrow A$ and $g:(U^#,Q) \rightarrow A/U$
            ($Q$ denotes the quadratic form of $A$) are the natural morphisms of quadratic modules.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A.anisotropic_kernel()
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2

        """
        maximal_isotropic = self.isotropic_subgroups()[0]
        return self.quotient(maximal_isotropic)

    ###################################
    # Deriving subgroups
    ###################################

    def subgroup(self, arg=None, names="f"):
        r"""
        Return a subgroup of the underlying abelian group $U$ of this quadratic module.
        Return the subgroup of $U$ generated by the elements in arg if arg is a list or tuple.
        Return the $p$-Sylow subgroup if $arg$ is a prime $p$.

        INPUT:

           - `arg` -- a list of elements of this quadratic module or a prime number
           - `names` -- names of the generators of the subgroup.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b,c,d,e,f,g> = FiniteQuadraticModule('11^-3.2_2^4')
            sage: A2 = A.subgroup (2); A2
            < d, e, f, g >
            sage: A11 = A.subgroup (11); A11
            < a, b, c >
            sage: A.<a,b,c,d> = FiniteQuadraticModule('11^-3.2_2^4', default_coords='fundamental')
            sage: A2 = A.subgroup (2); A2
            < a, 11*b, 11*c, 11*d >
            sage: A3 = A.subgroup (3); A3
            < 0 >
            sage: A11 = A.subgroup (11); A11
            < 10*b, 10*c, 10*d >
            sage: A11.order()
            1331
            sage: A2.order()
            16
        """
        if not arg:
            arg = [self(0)]
        if isinstance(arg, (list, tuple)):
            return FiniteQuadraticModule_subgroup(self, tuple(arg))
        p = Integer(arg)
        if is_prime(p):
            U = FiniteQuadraticModule_subgroup(self, tuple(self.gens()))
            U = U.split(p)[0] if 0 == U.order() % p else self.subgroup([self(0)])
            return U
        raise ValueError

    def kernel(self):
        r"""
        Return the dual subgroup of the underlying group of this module,
        i.e. return $\{y \in A : B(y,x) = 0 \text{ for all } x \in A  \}$,
        for this module $(A,Q)$.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3]); A
            Finite quadratic module in 2 generators:
             gens: a, b
             form: 1/3*x0^2 + 1/3*x0*x1 + 1/3*x1^2
            sage: U = A.kernel(); U
            < a + b >
            sage: B = A.quotient(U); B
            Finite quadratic module in 2 generators:
             gens: 2*e1, e1
             form: 1/3*x0^2 + 1/3*x0*x1 + 1/3*x1^2
            sage: B.kernel()
            < 0 >
            sage: B.jordan_decomposition().genus_symbol()
            '3^-1'
        """
        return self.dual_subgroup(self.subgroup(self.gens()))

    def dual_group(self, names="e", base_ring=None):
        """
        Return the dual group of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: A.dual_group()
            < a + b >

        """
        return self.dual_subgroup(self.subgroup(self.gens()))

    def dual_subgroup(self, U):
        r"""
        Return the dual subgroup
        $U^\sharp = \{y \in A : B(y,x) = 0 \text{ for all } x \in U  \}$
        of the subgroup $U$ of $self = (A,Q)$

        INPUT:

        - ``U`` -- a subgroup of this quadratic module

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: U=A.subgroup([A.gens()[1]])
            sage: A.dual_subgroup(U)
            < a + b >

        NOTES:
            If  the  dual  group  (w.r.t. the fundamental system) is  given
            as $K\ZZ^r/E\ZZ^n$ then the
            columns of $K$ form a basis for the integral solutions  of
            $2H^tJx \in \ZZ^n$. We solve this  by the trick of augmenting
            $2H^tJx$ by the unit matrix and solving the corresponding
            system of linear equations over $\ZZ$
        """

        H = U.gens_matrix()  # matrix(U)
        X = 2 * H.transpose() * self.gram(coords='canonical')
        n = X.nrows()
        Y = X.augment(MatrixSpace(QQ, n).identity_matrix())
        # print("Y=",Y)
        K0 = matrix(ZZ, Y.transpose().integer_kernel().matrix().transpose())
        # print("K0=",K0)
        K = K0.matrix_from_rows(list(range(n)))
        # print("K=",K)
        gens = [FiniteQuadraticModuleElement(self, x.list(), coords=U._default_coords)
                for x in K.columns()]
        # print("gens=",gens)
        return self.subgroup(gens)

    def kernel_subgroup(self, c):
        r"""
        Return the subgroup D_c={ x in D | cx=0}

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: A.kernel_subgroup(1)
            < 0 >
            sage: A.kernel_subgroup(3)
            < a, b >
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A.kernel_subgroup(2)
            < 11*e0, 33*e1 >

        """
        if c not in ZZ:
            raise ValueError("c has to be an integer.")
        if gcd(c, self.order()) == 1:
            return self.subgroup([])
        gens = []
        for x in self:
            y = c * x
            if y == self(0):
                gens.append(x)
        return self.subgroup(gens)

    def power_subgroup(self, c):
        r"""
        Compute the subgroup D^c={c*x | x in D}

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: A.power_subgroup(1)
            < a, b >
            sage: A.power_subgroup(3)
            < 0 >
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A.power_subgroup(1)
            < e0, e1 >
            sage: A.power_subgroup(2)
            < 2*e0, 2*e1 >
            sage: A.power_subgroup(12)
            < 2*e0, 6*e1 >

        """
        return self.subgroup(list(set([c * x for x in self])))

    def power_subset_star(self, c, check=False):
        r"""
        Compute the subset D^c*={x in D | cQ(y)+(x,y)=0, for all y in D_c}
        Using D^c* = x_c + D^c

        INPUT:

        - `c` -- integer
        - `check` -- boolean (default: False) if True then we cchek the definition

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: A.power_subset_star(1, check=True)
            [0, b, 2*b, a, a + b, a + 2*b, 2*a, 2*a + b, 2*a + 2*b]
            sage: F.<a,b,c> = FiniteQuadraticModule('2_1^1.2^2')
            sage: F.power_subset_star(2, check=True)
            [a]
            """
        if c % self.level() == 0:
            return [self(0)]
        xc = self.xc(c)
        Dc = self.power_subgroup(c)
        res = [x + xc for x in Dc]
        if check:
            res_by_definition = [
                x for x in self if all(c * self.Q(y) + self.B(x, y) in ZZ
                                       for y in self.kernel_subgroup(c))
            ]
            if set(res) != set(res_by_definition):
                raise ArithmeticError("Formula using x_c disagrees with definition!"
                                      f"By def: {res_by_definition}")
        return res

    def power_subset_kernel_star(self, c, S0=None):
        r"""
        Compute the subset D_S0^c*={x in D | cQ(y)+(x,y)=0, for all y in c^-1*S0 \cap S0^{\perp}}

        Note: This just uses a brute-force check.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: A.power_subset_kernel_star(1)
            [0, b, 2*b, a, a + b, a + 2*b, 2*a, 2*a + b, 2*a + 2*b]

        """
        if not S0 or S0.order() == 1:
            return self.power_subset_star(c)

        xc = self.xc(c)
        Dc = self.power_subgroup(c)
        res = []
        if xc == self._zero:
            return list(Dc)
        for x in Dc:
            res.append(x + xc)
        return res

    def Q_c(self, c: Integer, omega: FiniteQuadraticModuleElement) -> QQ:
        r"""
        Return Qc(omega) = cQ(y) + B(x_c,y) + ZZ for omega = x_c + yc in D^c*

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: A.Q_c(1, a)
            1/3
            sage: A.Q_c(1, 1)
            Traceback (most recent call last):
            ...
            TypeError: unsupported operand parent(s) for -: 'Integer Ring' and 'Finite quadratic...
        """
        y = omega - self.xc(c)
        if y not in self.power_subgroup(c):
            raise ValueError("Q_c(omega) is only defined for omega in D^c")
        y = y / c
        return c*self.Q(y) + self.B(self.xc(c), y)

    def xc(self, c):
        r"""
        Compute all non-zero values of the element x_c in the group D

        INPUT:

        - 'c'~- integer

        OUTPUT:

        - 'x_c' element of self such that D^{c*} = x_c + D^{c}

        EXAMPLES::
            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule([3,3], [1/3,1/3,1/3])
            sage: A.xc(1)
            0
            sage: A.xc(2)
            0
            sage: F = FiniteQuadraticModule('2_1^1')
            sage: F.xc(1)
            0
            sage: F.xc(2)
            e
            sage: F = FiniteQuadraticModule('4^2.2_1')
            sage: F.xc(1)
            0
            sage: F.xc(2)
            e2
            sage: F.<a,b,c> = FiniteQuadraticModule('2_1^1.2^2')
            sage: F.xc(2)
            a
            sage: F.xc(6)
            a

        """
        if is_odd(c) or c % self.level() == 0:
            return self._zero
        k = valuation(c, 2)
        if self.level() % 2**(k+1) != 0:
            return self._zero
        return self._xc(k)

    def _xc(self, k: Integer):
        r"""
        Compute $x_c$ such that $D^{c*} = x_c + D^{c}$ for $c$ s.t. $2^k || c$

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2_1^1')
            sage: A._xc(0)
            0
            sage: A._xc(1)
            e
        """
        if self._xcs == {}:
            self._compute_all_xcs()
        if k in self._xcs:
            return self._xcs[k]
        return self._zero

    def _compute_all_xcs(self):
        r"""
        Compute all non-zero values of the element `x_c` in self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2_1^1')
            sage: A._compute_all_xcs()
            sage: A._xcs
            {0: 0, 1: e}
        """
        J = self.jordan_decomposition()
        res = dict()
        res[0] = 0
        for comp in J:
            # p, k, r, d = c.[1][:4]
            # t = None if 4 == len(c[1]) else c[1][4]
            if comp.p != 2 or comp.t is None:
                continue
            res[comp.k] = sum(2 ** (comp.k - 1) * ind.basis()[0]
                              for ind in comp.decompose() if ind.is_type_I())
        self._xcs = res

    def subgroups(self, d=None):
        r"""
        Return a list of all subgroups of $M$ of order $d$, where $M$
        is the underlying group of self, or of all subgroups if d is not set.

        INPUT
            d -- integer

        OUTPUT
            generator for a list of FiniteQuadraticModule_subgroup of order d

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([1,3]); A
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/4*x0^2 + 1/12*x1^2
            sage: list(A.subgroups())
             [< e0, e1 >,
             < e0, 2*e1 >,
             < e0 + e1, 2*e1 >,
             < e0, 3*e1 >,
             < e0 >,
             < e0 + 3*e1 >,
             < e1 >,
             < 2*e1 >,
             < 3*e1 >,
             < 0 >]
             sage: B.<a> = FiniteQuadraticModule('25^-1'); B
             Finite quadratic module in 1 generator:
              gen: a
              form: 1/25*x^2
             sage: I = [U for U in B.subgroups() if U.is_isotropic()]; I
             [< 5*a >, < 0 >]

        ....NOTES:

            Subgroups are  internally represented by lower triangular
            matrices in Hermite normal form dividing $M$.

            Any   subgroup  $U$   of  a finite  abelian group $\ZZ^n/R\ZZ^n$   is  of   the  form
            $N\ZZ^n/R\ZZ^n$,  where  $N$  is  a regular  integral  square
            matrix of  size $n$ such  that $N$ left divides $M$,  (i.e. such
            that  $N^{-1}M$  is  an   integer  matrix).  The  coset  $N
            GL(n,\ZZ)$ is  uniquely determined by $U$.   Recall that any
            such  coset contains  exactly one lower triangular matrix
            $H = (h_[ij})$  in Hermite  normal form  (i.e.  $h_{ii}>0$ and
            $0\le h_{ij}$ < h_{ii}$ for $j < i$).

            Accordingly, this function sets up an iterator over all
            matrices in lower Hermite normal form left dividing the diagonal
            matrix whose diagonal entries are the elementary divisors of
            the underlying abelian group of this module
            (and whose determinants equal $self.order()/d$).

            Note that some groups might appear twice.

        ...TODO:
            Find a more effective implementation.

            Introduce optional arguments which allow to iterate in addition effectively
            over all subgroups contained in or containing a certain subgroup, being isotropic etc.

            One can use short cuts using e.g. that $N$ defines
            an isotropic subgroups if and only if $N^tJN$ is half integral (where $J$ is the Gram matrix
            w.r.t the fundamental generators of this module.
        """
        elementary_divisors = self.elementary_divisors()
        N = len(elementary_divisors)

        Mat = MatrixSpace(ZZ, N)

        def __enumerate_echelon_forms():
            """
            Return an iterator over all matrices in HNF
            left dividing self.__E (the relations matrix for the fundamental system).

            NOTE:
            If e1, e_2, ...  denote the elemetary divisors of self,
            we define a string like
            ([[d_0,0], [n_10,d_1], ...]
                for d_0 in divisors(e1)
                for d_1 in divisors(e2)
                for n_10 in range(d_1), ... )
            and evaluate it to obtain our generator.
            """
            genString = '['
            forStringN = ''
            forStringD = ''

            for r in range(N):
                genString += ' ['
                for c in range(N):
                    if c < r:
                        v = 'n_' + str(r) + str(c)
                        genString += v + ','
                        forStringN += 'for ' + v + ' in ' + 'range(d_' + str(r) + ') '
                    elif c == r:
                        v = 'd_' + str(r)
                        genString += v + ','
                        forStringD += 'for ' + v + ' in ' + 'divisors(' + str(
                            elementary_divisors[r]) + ') '
                    else:
                        genString += '0,'

                genString = genString[:-1] + '] ,'
            genString = genString[:-1] + '] '

            genExpression = '(' + genString + forStringD + forStringN + ') '
            return eval(genExpression)

        for h in __enumerate_echelon_forms():
            h1 = Mat(h)
            if FiniteQuadraticModule_subgroup._divides(h1, self.relations(coords='fundamental')):
                gens = [FiniteQuadraticModuleElement(self, list(x),
                                                     coords='fundamental')
                        for x in h1.transpose()]
                subgroup = FiniteQuadraticModule_subgroup(self, tuple(gens))
                if d:
                    if subgroup.order() == d:
                        yield subgroup
                else:
                    yield subgroup

    def isotropic_subgroups(self):
        """
        Return a list of isotropic subgroups of self.

        TODO: A more efficient implementation

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A.isotropic_subgroups()
            [< 0 >, < 11*e0 + 33*e1 >]

        """

        isotropic_subgroups = []
        for subgroup in self.subgroups():
            if set([self.Q(g) for g in subgroup]) == {0}:
                isotropic_subgroups.append(subgroup)
        isotropic_subgroups.sort(key=lambda x: x.order())
        return isotropic_subgroups



###################################
# Indecomposable modules
###################################

def _A(q, s=1, **args):
    r"""
    Return the quadratic module $A_q^s$ as defined in [Sko].

    INPUT:

    - ``q`` -- prime power
    - ``s`` -- integer (coprime to q)

    EXAMPLES::

        sage: from fqm_weil.all import _A
        sage: A = _A(3); A
        Finite quadratic module in 1 generator:
         gen: e
         form: 1/3*x^2
        sage: B = _A(5,-2); B
        Finite quadratic module in 1 generator:
         gen: e
         form: 3/5*x^2
        sage: C.<a> = _A(2^5); C
        Finite quadratic module in 1 generator:
         gen: a
         form: 1/64*x^2
        sage: a.order()
        32
        sage: _A(2, 7)
        Traceback (most recent call last):
        ...
        NotImplementedError: Current implementation does not work for A_2^7...
    """
    q = Integer(q)
    s = Integer(s)
    if q == 2 and s == 7:
        raise NotImplementedError("Current implementation does not work for A_2^7 "
                                  "which should correspond to the genus symbol '2_7^1'")
    if q > 1 and q.is_prime_power() and 1 == q.gcd(s):
        if is_odd(q):
            return FiniteQuadraticModule([q], [s / q], **args)
        else:
            return FiniteQuadraticModule([q], [s / (2 * q)], **args)
    raise ValueError(
        'q (={0}) must be prime power and s (={1}) relatively prime to q'.format(q, s))


def _B(q, **args):
    r"""
    Return the quadratic module $C_q$ as defined in [Sko].

    INPUT:

    - ``q`` -- prime power

    EXAMPLES::

        sage: from fqm_weil.all import _B
        sage: B.<a,b> = _B(2^3); B
        Finite quadratic module in 2 generators:
         gens: a, b
         form: 1/8*x0^2 + 1/8*x0*x1 + 1/8*x1^2
        sage: (a+b).order()
        8
    """
    q = Integer(q)
    if is_even(q) and q.is_prime_power():
        return FiniteQuadraticModule([q, q],
                                     matrix(QQ, 2, [1 / q, 1 / (2 * q), 1 / (2 * q), 1 / q]),
                                     **args)
    raise ValueError('q = ({0}) must be q power of 2'.format(q))


def _C(q, **args):
    r"""
    Return the quadratic module $C_q$ as defined in [Sko].

    INPUT:

    - ``q`` -- prime power

    EXAMPLES::

        sage: from fqm_weil.all import _C
        sage: C.<a,b> = _C(2); C
        Finite quadratic module in 2 generators:
         gens: a, b
         form: 1/2*x0*x1
        sage: C.exponent()
        2
    """
    q = Integer(q)
    if is_even(q) and q.is_prime_power():
        return FiniteQuadraticModule([q, q], matrix(QQ, 2, [0, 1 / (2 * q), 1 / (2 * q), 0]),
                                     **args)
    raise ValueError('q = ({0}) must be q power of 2'.format(q))


def _FiniteQuadraticModule_from_string_old(S, **args):
    r"""
    Return the quadratic module described by a genus symbol in the string S.

    INPUT:
    - ``S`` -- a string representing Jordan constituents of a finite quadratic modules

    NOTES:
        The strings which will be accepted have the form
        $$
        'a^{\pm k}.b^{\pm l}.c^{\pm m}. \dots',
        $$
        where $a$, $b$, $c$, \dots are prime powers, and where $k$, $l$, etc.
        are positive integers (if an exponent $\pm k$ equals 1 it can be omitted).
        If the $a$, $b$, \dots are powers of $2$ we admit also subscripts $t$, i.e. symbols
        of the form $a_t^{\pm k}$, where $0\le t < 8$ is an integer.

        The dot corresponds to the direct sum of quadratic modules, and the symbols $a^{\pm k}$, \dots
        have the following meaning:

        For a $2$-power $a$, the symbol $a^{+k}$, indicates the $k/2$-fold direct sum of the module $B = (\ZZ^2/a\ZZ^2, xy/a)$,
        whereas the symbol $a^{-k}$ denotes the module $(\ZZ^2/a\ZZ^2, (x^2+xy+y^2)/a)$ plus the $k/2-1$-fold sum of $(\ZZ^2/a\ZZ^2, xy/a)$.

        A symbol $a^{\pm k}$, for an odd prime power $a$, indicates the quadratic module
        $A_a^e+(k-1)A_a$, where $e$ is an integer such that the Legendre symbol $2^k e$ over $p$ equals $\pm 1$.

        A symbol $a_t^{\pm k}$, for a $2$-power $a$, indicates the quadratic module
        $A_a^{c_1}+\cdots + A_a^{c_k}$, where the $c_i$ are odd integers such that
        $D := c_1 \cdots c_k$ is a quadratic residue or non-residue modulo $8$ according to
        the sign $\pm$, and where $c_1 + \cdots + c_k \equiv t \bmod 8$. Here, for even $a$, we
        use $A_a = (\ZZ/a\ZZ, x^2/2a)$.

        Note that, for a symbol $2^{\pm k}$, the $k$ must be even.
        Furthermore, a solution $(c_1,\dots,c_k)$ of the equations $\sum c_i \equiv t \bmod 8$ and
        Legendre symbol of 8 over $\prod c_i$ equal to $\pm 1$ exists if and only if
        $t \equiv k \bmod 2$, legendre symbol of $8$ over $t$ equal to $\pm 1$ for $k=1$,
        if $t\equiv 0 \bmod 8$ then $\pm 1 = +1$ and if $t\equiv 4 \bmod 8$ then \pm 1 = -1$
        for $k=2$. If any of these conditions is not fulfilled an error is raised.

    ALGORITHM::

        The genus symbol is first split into components separated by "." and then each component
        is used to create an instance of JordanComponent. This individual components are then
        decomposed into indecomposable modules of the form A_q^t, B_q and C_q and each of these is
        then realised as finite quadratic modules before summing them at the end.

    EXAMPLES::

        sage: from fqm_weil.modules.finite_quadratic_module.finite_quadratic_module_ambient \
        ....: import _FiniteQuadraticModule_from_string
        sage: A.<a,b,c,d> =_FiniteQuadraticModule_from_string ('3^-1.3.5^2', \
        ....: default_coords='canonical'); A
        Finite quadratic module in 4 generators:
         gens: a, b, c, d
         form: 1/3*x0^2 + 2/3*x1^2 + 2/5*x2^2 + 2/5*x3^2
        sage: A =_FiniteQuadraticModule_from_string ('3^-1.3.5^2'); A
        Finite quadratic module in 4 generators:
         gens: e0, e1, e2, e3
         form: 1/3*x0^2 + 2/3*x1^2 + 2/5*x2^2 + 2/5*x3^2
        sage: A =_FiniteQuadraticModule_from_string ('8^+6'); A
        Finite quadratic module in 6 generators:
         gens: e0, e1, e2, e3, e4, e5
         form: 1/8*x0*x1 + 1/8*x2*x3 + 1/8*x4*x5
        sage: A.elementary_divisors ()
        (8, 8, 8, 8, 8, 8)
        sage: A =_FiniteQuadraticModule_from_string ('8_1^+3'); A
        Finite quadratic module in 3 generators:
         gens: e0, e1, e2
         form: 1/16*x0^2 + 1/8*x1*x2
        sage: A.elementary_divisors ()
        (8, 8, 8)
        sage: D.<a,b,c,d,e> = _FiniteQuadraticModule_from_string ('8_1^3.4^-2'); D
        Finite quadratic module in 5 generators:
         gens: a, b, c, d, e
         form: 1/16*x0^2 + 1/8*x1*x2 + 1/4*x3^2 + 1/4*x3*x4 + 1/4*x4^2
        sage: D.<a,b,c,d,e> = _FiniteQuadraticModule_from_string ('8_1^3.4^-2', \
        ....: default_coords='fundamental'); D
        Finite quadratic module in 5 generators:
         gens: a, b, c, d, e
         form: 1/4*x0^2 + 1/4*x0*x1 + 1/4*x1^2 + 1/4*x0*x3 + 1/2*x1*x3 + 7/8*x2*x3 + 1/4*x3^2 + 1/2*x0*x4 + 1/4*x1*x4 + 1/4*x3*x4 + 5/16*x4^2
        sage: D.level(), D.exponent(), D.order(), D.elementary_divisors ()
        (16, 8, 8192, (4, 4, 8, 8, 8))
        sage: E.<a,b,c,d,e,f,g> = _FiniteQuadraticModule_from_string ('8_1^3.4^-2.3^-1.11^-1'); E
        Finite quadratic module in 7 generators:
         gens: a, b, c, d, e, f, g
         form: 1/16*x0^2 + 1/8*x1*x2 + 1/4*x3^2 + 1/4*x3*x4 + 1/4*x4^2 + 1/3*x5^2 + 1/11*x6^2
        sage: E.elementary_divisors()
         (4, 4, 8, 8, 264)

    TECHNICAL NOTES:

        The accepted strings can be described in BNF as follows:

        \begin{verbatim}
        <S>            ::= <symbol_list>
        <symbol_list>  ::= <symbol_list> "." <symbol> | <symbol>
        <symbol>       ::= <p-power> | <p-power>"^"<exponent> |
                           <2-power> "_" type | <2-power> "_" <type> "^" <exponent>
        <p-power>      ::= number
        <2-power>      ::= number
        <type>         ::= number
        <exponent>     ::= number | "+"number | "-"number
        \end{verbatim}

        Of course, we impose  the additional requirements that
        number is a positive integer, and number and type satisfy the above requirements.
    """

    S = S.replace(' ', '')
    List = S.split('.')

    ElementList = []
    for item in List:
        L1 = item.split("^")
        if len(L1) > 2:  # more than one ^ in item
            raise ValueError
        elif len(L1) == 2:
            k = Integer(L1[1])
        else:
            k = 1
        L1 = L1[0].split("_")
        a = Integer(L1[0])
        if len(L1) > 2:  # more than one _ in item
            raise ValueError
        elif len(L1) == 2:
            if Integer(L1[1]) in range(8):
                t = Integer(L1[1])
            else:
                raise ValueError("Type given, which is not in 0..7: {0}".format(L1[1]))
        else:
            t = None
        if not (k != 0 and a != 1 and a.is_prime_power()
                and (t is None or (is_even(a) and t % 2 == k % 2))
                and (not (t is None and is_even(a)) or 0 == k % 2)
        ):
            raise ValueError("{0} is not a valid signature!".format(S))
        c = None
        if is_odd(a):
            c = [1] * abs(k)
            p = a.factor()[0][0]
            s = kronecker(2, p) ** k
            if s * k < 0:
                c[0] = -1 if 3 == p % 4 else primitive_root(p)
        if is_even(a) and t is not None:
            if 1 == abs(k):
                if k == kronecker(t, 2):
                    c = [t]
                else:
                    raise ValueError("{0} is not a valid signature!".format(S))
            if abs(k) > 1:
                CP = cartesian_product([[1, 3, 5, 7] for x in range(abs(k) - 1)])
                # CP = eval("cartesian_product([" + "[1,3,5,7]," * (abs(k) - 1) + "])")
                # TODO: find better algorithm
                e = 1 if k > 0 else -1
                for x in CP:
                    s = sum(x) % 8
                    if kronecker(prod(x) * (t - s), 2) == e:
                        c = list(x)
                        c.append(t - s)
                        break
                if not c:
                    raise ValueError("{0} is not a valid signature!".format(S))
        entry = {'a': a, 'k': k, 't': t, 'c': c}
        ElementList.append(entry)
    names = args.pop('names', None)
    # TODO: Once the 0-module is cached replace the next 6 lines by: A = FiniteQuadraticModule()
    print("ElementList:", ElementList)
    sym = ElementList[0]
    q = sym['a']
    t = sym['t']
    k = sym['k']
    A = None
    if is_odd(q):
        A = sum(_A(q, s, **args) for s in sym['c'])
    elif t is not None:
        A = sum(sym['c'])
    if is_even(q) and t is None:
        A = _C(q, **args) * (k // 2) if k > 0 else _B(q, **args)
        if (-k) // 2 > 1:
            A += _C(q, **args) * ((-k) // 2 - 1)
    for sym in ElementList[1:]:
        q = sym['a']
        t = sym['t']
        k = sym['k']
        if is_odd(q) or t is not None:
            A += sum(_A(q, s, **args) for s in sym['c'])
        if is_even(q) and t is None:
            A += _C(q, **args) * (k // 2) if k > 0 else _B(q, **args)
            if (-k) // 2 > 1:
                A += _C(q, **args) * ((-k) // 2 - 1)
    A = FiniteQuadraticModule_ambient(A.relations(),
                                      A.gram(),
                                      names=names, **args)
    return A

def _FiniteQuadraticModule_from_string(S, **kwargs):
    r"""
    Return the quadratic module described by the string S.

    INPUT:

    - ``S`` -- a string representing Jordan constituents of a finite quadratic modules

    NOTES:
        The strings which will be accepted have the form
        $$
        'a^{\pm k}.b^{\pm l}.c^{\pm m}. \dots',
        $$
        where $a$, $b$, $c$, \dots are prime powers, and where $k$, $l$, etc.
        are positive integers (if an exponent $\pm k$ equals 1 it can be omitted).
        If the $a$, $b$, \dots are powers of $2$ we admit also subscripts $t$, i.e. symbols
        of the form $a_t^{\pm k}$, where $0\le t < 8$ is an integer.

        The dot corresponds to the direct sum of quadratic modules, and the symbols $a^{\pm k}$, \dots
        have the following meaning:

        For a $2$-power $a$, the symbol $a^{+k}$, indicates the $k/2$-fold direct sum of the module $B = (\ZZ^2/a\ZZ^2, xy/a)$,
        whereas the symbol $a^{-k}$ denotes the module $(\ZZ^2/a\ZZ^2, (x^2+xy+y^2)/a)$ plus the $k/2-1$-fold sum of $(\ZZ^2/a\ZZ^2, xy/a)$.

        A symbol $a^{\pm k}$, for an odd prime power $a$, indicates the quadratic module
        $A_a^e+(k-1)A_a$, where $e$ is an integer such that the Legendre symbol $2^k e$ over $p$ equals $\pm 1$.

        A symbol $a_t^{\pm k}$, for a $2$-power $a$, indicates the quadratic module
        $A_a^{c_1}+\cdots + A_a^{c_k}$, where the $c_i$ are odd integers such that
        $D := c_1 \cdots c_k$ is a quadratic residue or non-residue modulo $8$ according to
        the sign $\pm$, and where $c_1 + \cdots + c_k \equiv t \bmod 8$. Here, for even $a$, we
        use $A_a = (\ZZ/a\ZZ, x^2/2a)$.

        Note that, for a symbol $2^{\pm k}$, the $k$ must be even.
        Furthermore, a solution $(c_1,\dots,c_k)$ of the equations $\sum c_i \equiv t \bmod 8$ and
        Legendre symbol of 8 over $\prod c_i$ equal to $\pm 1$ exists if and only if
        $t \equiv k \bmod 2$, legendre symbol of $8$ over $t$ equal to $\pm 1$ for $k=1$,
        if $t\equiv 0 \bmod 8$ then $\pm 1 = +1$ and if $t\equiv 4 \bmod 8$ then \pm 1 = -1$
        for $k=2$. If any of these conditions is not fulfilled an error is raised.

    EXAMPLES::

        sage: from fqm_weil.modules.finite_quadratic_module.finite_quadratic_module_ambient \
        ....: import _FiniteQuadraticModule_from_string
        sage: A.<a,b,c,d> =_FiniteQuadraticModule_from_string ('3^-1.3.5^2', \
        ....: default_coords='canonical'); A
        Finite quadratic module in 4 generators:
         gens: a, b, c, d
         form: 1/3*x0^2 + 2/3*x1^2 + 2/5*x2^2 + 2/5*x3^2
        sage: A =_FiniteQuadraticModule_from_string ('3^-1.3.5^2'); A
        Finite quadratic module in 4 generators:
         gens: e0, e1, e2, e3
         form: 1/3*x0^2 + 2/3*x1^2 + 2/5*x2^2 + 2/5*x3^2
        sage: A =_FiniteQuadraticModule_from_string ('8^+6'); A
        Finite quadratic module in 6 generators:
         gens: e0, e1, e2, e3, e4, e5
         form: 1/8*x0*x1 + 1/8*x2*x3 + 1/8*x4*x5
        sage: A.elementary_divisors ()
        (8, 8, 8, 8, 8, 8)
        sage: A =_FiniteQuadraticModule_from_string ('8_1^+3'); A
        Finite quadratic module in 3 generators:
         gens: e0, e1, e2
         form: 1/16*x0^2 + 1/8*x1*x2
        sage: A.elementary_divisors ()
        (8, 8, 8)
        sage: D.<a,b,c,d,e> = _FiniteQuadraticModule_from_string ('8_1^3.4^-2'); D
        Finite quadratic module in 5 generators:
         gens: a, b, c, d, e
         form: 1/16*x0^2 + 1/8*x1*x2 + 1/4*x3^2 + 1/4*x3*x4 + 1/4*x4^2
        sage: D.<a,b,c,d,e> = _FiniteQuadraticModule_from_string ('8_1^3.4^-2', \
        ....: default_coords='fundamental'); D
        Finite quadratic module in 5 generators:
         gens: a, b, c, d, e
         form: 1/4*x0^2 + 1/4*x0*x1 + 1/4*x1^2 + 1/4*x0*x3 + 1/2*x1*x3 + 7/8*x2*x3 + 1/4*x3^2 + 1/2*x0*x4 + 1/4*x1*x4 + 1/4*x3*x4 + 5/16*x4^2
        sage: D.level(), D.exponent(), D.order(), D.elementary_divisors ()
        (16, 8, 8192, (4, 4, 8, 8, 8))
        sage: E.<a,b,c,d,e,f,g> = _FiniteQuadraticModule_from_string ('8_1^3.4^-2.3^-1.11^-1'); E
        Finite quadratic module in 7 generators:
         gens: a, b, c, d, e, f, g
         form: 1/16*x0^2 + 1/8*x1*x2 + 1/4*x3^2 + 1/4*x3*x4 + 1/4*x4^2 + 1/3*x5^2 + 1/11*x6^2
        sage: E.elementary_divisors()
         (4, 4, 8, 8, 264)

    TECHNICAL NOTES:

        The accepted strings can be described in BNF as follows:

        \begin{verbatim}
        <S>            ::= <symbol_list>
        <symbol_list>  ::= <symbol_list> "." <symbol> | <symbol>
        <symbol>       ::= <p-power> | <p-power>"^"<exponent> |
                           <2-power> "_" type | <2-power> "_" <type> "^" <exponent>
        <p-power>      ::= number
        <2-power>      ::= number
        <type>         ::= number
        <exponent>     ::= number | "+"number | "-"number
        \end{verbatim}

        Of course, we impose  the additional requirements that
        number is a positive integer, and number and type satisfy the above requirements.
    """
    names = kwargs.pop('names', None)
    A = sum(JordanComponent((), s).as_finite_quadratic_module(**kwargs) for s in S.split('.'))
    return FiniteQuadraticModule_ambient(A.relations(), A.gram(), names=names, **kwargs)


def FiniteQuadraticModule(arg0=None, arg1=None, **args):
    r"""
    Create an instance of the class FiniteQuadraticModule_base.

    INPUT:

    - ``arg0`` -- one of the supported formats from below.
    - ``arg1`` -- one of the supported formats from below.

    Note:
        Supported formats:

        N.  FiniteQuadraticModule():
                the trivial quadratic module

        S.  FiniteQuadraticModule(string):
                the quadratic module $(L^#/L, B(x,x)/2)$, where $(L,B)$
                is a $\Z_p$-lattice encoded by the string as described
                in Conway-Sloane, p.???. TODO: look up ???

        L.  FiniteQuadraticModule(list):
                discriminant module constructed from the diagonal matrix
                with $2*x$ and $x$ running through list on its diagonal.

        M.  FiniteQuadraticModule(matrix):
                discriminant   module   constructed   from  a   regular
                symmetric even integral matrix.

        F.  FiniteQuadraticModule(number_field_element):
                For a nonzero $\omega$ in a numberfield $K$,
                the quadratic module $(\ZZ_K/A, x+A \mapsto tr(\omega x^2) + \ZZ)$,
                where $A$ is determined by
                $\omega D = B/A$ with relatively prime ideals $A$, $B$,
                and with $D$ denoting the different of $K$.

        LL. FiniteQuadraticModule(list_of_orders, list_of_coeffs):
                for a list of orders $[e_i]$  of size $n$ and a list of
                coefficients $[a_{ij}]$, the quadratic module
                $(\ZZ/e_1\times\cdots\times\ZZ/e_n,class(x)\mapsto\sum_{i\le j} a_{ij} x_i x_j)$.

        LM. FiniteQuadraticModule(list_of_orders, Gram_matrix):
                for  a  list  of  orders  $[e_i]$ of  size  $n$  and  a
                symmetric matric $G$, the quadratic module
                $(\ZZ/e_1 \times \cdots \times \ZZ/e_n, class(x) \mapsto G[x] + \ZZ)$.

        ML. FiniteQuadraticModule(matrix, list_of_coeffs):
                for a matrix $R$ of  size $n$ and a list of coefficients
                $[a_{ij}]$, the quadratic module
                $(\ZZ^n/R\ZZ^n, x+R\ZZ^n \mapsto \sum_{i\le j} a_{ij} x_i x_j)$.

        MM. FiniteQuadraticModule(matrix, Gram_matrix):
                for  a  matrix $R$  and  a  symmetric  matric $G$,  the
                quadratic module $(\ZZ^n/R\ZZ^n, x+R\ZZ^n \mapsto G[x] + \ZZ)$.


    EXAMPLES::

        sage: from fqm_weil.all import FiniteQuadraticModule
        sage: N.<n> = FiniteQuadraticModule(); N
        Trivial finite quadratic module.
        sage: n.order()
        1
        sage: N.is_trivial()
        True
        sage: S.<x,y,z> = FiniteQuadraticModule('7^-1.3.2_3^-1'); S
        Finite quadratic module in 3 generators:
         gens: x, y, z
         form: 3/7*x0^2 + 2/3*x1^2 + 3/4*x2^2
        sage: S.<x> = FiniteQuadraticModule('7^-1.3.2_3^-1', default_coords='fundamental'); S
        Finite quadratic module in 1 generator:
         gen: x
         form: 71/84*x^2
        sage: L.<w> = FiniteQuadraticModule([13]); L
        Finite quadratic module in 1 generator:
         gen: w
         form: 1/52*x^2

        sage: E8 = matrix(ZZ, 8, [4,-2,0,0,0,0,0,1,-2,2,-1,0,0,0,0,0,0,-1,2,-1,0,0,0,0,0,0,-1,2,-1,0,0,0,0,0,0,-1,2,-1,0,0,0,0,0,0,-1,2,-1,0,0,0,0,0,0,-1,2,0,1,0,0,0,0,0,0,2]); E8
        [ 4 -2  0  0  0  0  0  1]
        [-2  2 -1  0  0  0  0  0]
        [ 0 -1  2 -1  0  0  0  0]
        [ 0  0 -1  2 -1  0  0  0]
        [ 0  0  0 -1  2 -1  0  0]
        [ 0  0  0  0 -1  2 -1  0]
        [ 0  0  0  0  0 -1  2  0]
        [ 1  0  0  0  0  0  0  2]
        sage: M.<a,b,c,d,e,f,g,h> = FiniteQuadraticModule(3*E8); M
        Finite quadratic module in 8 generators:
         gens: a, b, c, d, e, f, g, h
         form: 1/3*x0^2 + 2/3*x0*x2 + 2/3*x1*x2 + 1/3*x0*x3 + 1/3*x1*x3 + 1/3*x3^2 + 2/3*x0*x5 + 2/3*x1*x5 + 1/3*x3*x5 + 2/3*x4*x5 + 1/3*x0*x6 + 1/3*x1*x6 + 2/3*x3*x6 + 1/3*x4*x6 + 1/3*x6^2 + 2/3*x0*x7 + 2/3*x2*x7 + 1/3*x3*x7 + 2/3*x5*x7 + 1/3*x6*x7 + 2/3*x7^2

        sage: X = QQ['X'].0
        sage: K.<x> = NumberField(X^4-8*X^3+1); K
        Number Field in x with defining polynomial X^4 - 8*X^3 + 1
        sage: F.<a,b,c,d> = FiniteQuadraticModule((x^2-4)/7); F
        Finite quadratic module in 4 generators:
         gens: a, b, c, d
         form: 6/7*x0^2 + 1/7*x0*x1 + 5/7*x1*x2 + 5/7*x0*x3 + 6/7*x2*x3 + 3/7*x3^2

        sage: LL = FiniteQuadraticModule([3,4,30],[1/3,0,1/3,1/8,5/2,7/60])
        sage: LL
        Finite quadratic module in 3 generators:
         gens: e0, e1, e2
         form: 1/3*x0^2 + 1/8*x1^2 + 1/3*x0*x2 + 1/2*x1*x2 + 7/60*x2^2
        sage: LL.elementary_divisors()
        (6, 60)
        sage: LL = FiniteQuadraticModule([3,4,30],[1/3,0,1/3,1/8,5/2,7/60], \
        ....: default_coords='fundamental'); LL
        Finite quadratic module in 2 generators:
          gens: e0, e1
          form: 11/12*x0^2 + 1/3*x0*x1 + 49/120*x1^2
        sage: LL.elementary_divisors()
        (6, 60)
        sage: LL2.<u,v> = FiniteQuadraticModule([5,5], [3/5,1/5,4/5]); LL2
        Finite quadratic module in 2 generators:
         gens: u, v
         form: 3/5*x0^2 + 1/5*x0*x1 + 4/5*x1^2
        sage: LL2.is_nondegenerate()
        True

        sage: G = matrix(3,3,[1,1/2,3/2,1/2,2,1/9,3/2,1/9,1]); G
        [ 1 1/2 3/2]
        [1/2   2 1/9]
        [3/2 1/9   1]
        sage: LM.<x,y,z> = FiniteQuadraticModule([4,9,18],G); LM
        Finite quadratic module in 3 generators:
         gens: x, y, z
         form: 2/9*x1*x2
        sage: LM.<x,y> = FiniteQuadraticModule([4,9,18],G, default_coords='fundamental'); LM
        Finite quadratic module in 2 generators:
         gens:  x, y
         form: 2/9*x0*x1
        sage: LM.is_nondegenerate ()
        False

        sage: M = matrix(2, [4,1,1,6]); M
        [4 1]
        [1 6]
        sage: ML.<s,t> = FiniteQuadraticModule(M, [3/23,-1/23,2/23])
        sage: ML
        Finite quadratic module in 2 generators:
         gens: 17*t, t
         form: 3/23*x0^2 + 22/23*x0*x1 + 2/23*x1^2
        sage: ML.<t> = FiniteQuadraticModule(M, [3/23,-1/23,2/23], default_coords='fundamental');ML
        Finite quadratic module in 1 generator:
         gen: t
         form: 2/23*x^2
        sage: E = matrix(2, [8,3,3,10]); E
        [ 8  3]
        [ 3 10]
        sage: MM.<x,y> = FiniteQuadraticModule(E, 1/2 * E^-1); MM
        Finite quadratic module in 2 generators:
         gens: 44*y, y
         form: 5/71*x0^2 + 68/71*x0*x1 + 4/71*x1^2
        sage: MM.<x> = FiniteQuadraticModule(E, 1/2 * E^-1, default_coords='fundamental'); MM
        Finite quadratic module in 1 generator:
         gen: x
         form: 4/71*x^2
        """
    if arg0 is None:
        if 'check' not in args:
            args['check'] = False
        return FiniteQuadraticModule_ambient(matrix(1, [1]), matrix(1, [0]), **args)
    elif isinstance(arg0, FiniteQuadraticModule_base):
        return copy(arg0)

    if arg1 is None:
        if isinstance(arg0, str):
            # S. FiniteQuadraticModule(string)
            if 'check' not in args:
                args['check'] = False
            return _FiniteQuadraticModule_from_string(arg0, **args)

        elif isinstance(arg0, list):
            # L. FiniteQuadraticModule(list_of_orders):
            M = matrix(ZZ, len(arg0), len(arg0),
                       dict([((i, i), 2 * arg0[i]) for i in range(len(arg0))]))
            G = QQ(1) / QQ(2) * M ** -1
        elif isinstance(arg0, (int, Integer)):
            M = matrix(ZZ, 1, 1, [arg0])
            G = QQ(1) / QQ(2) * M ** -1

        elif hasattr(arg0, '_matrix_'):
            # M. FiniteQuadraticModule(matrix):
            M = matrix(ZZ, arg0)
            G = QQ(1) / QQ(2) * M ** -1

        elif isinstance(arg0, NumberFieldElement):
            # F. FiniteQuadraticModule(number_field_element):
            if arg0.is_zero():
                raise ValueError("{0}: must be nonzero".format(arg0))
            K = arg0.parent()
            d = K.different()
            n = K.degree()
            basis = K.integral_basis()
            # Find G:
            G = matrix(QQ, n, n)
            for i in range(n):
                for j in range(n):
                    G[i, j] = (arg0 * basis[j] * basis[i]).trace()

            # Compute the denominator ideal of omega*different:
            p = arg0 * d
            s = p.factor()
            A = K.ideal([1])
            for i in range(len(s)):
                s_factor = s[i]
                if s_factor[1] < 0:
                    A = A * s_factor[0] ** -s_factor[1]
            # Compute M as the product of two matrices:
            L = matrix(QQ, n, n)
            for j in range(n):
                for i in range(n):
                    L[j, i] = A.integral_basis()[i].list()[j]
            E = matrix(QQ, n, n)
            for j in range(n):
                for i in range(n):
                    E[j, i] = basis[i].list()[j]
            M = E ** -1 * L
        else:
            raise ValueError("Can not construct finite quadratic module from {0}".format(arg0))
    else:

        if isinstance(arg0, list):
            M = matrix(ZZ, len(arg0), len(arg0),
                       dict([((i, i), arg0[i]) for i in range(len(arg0))]))
        elif hasattr(arg0, '_matrix_'):
            M = arg0._matrix_().change_ring(ZZ)
        else:
            raise TypeError("{0}: should be None, a matrix or a list".format(arg0))

        if isinstance(arg1, list):
            # LL./ML. FiniteQuadraticModule(list_of_orders/matrix, list_of_coeffs):
            n = M.nrows()
            G = matrix(QQ, n, n)
            i = j = 0
            for g in arg1:
                if i == j:
                    G[i, j] = g - floor(g)
                else:
                    G[i, j] = G[j, i] = QQ(g - floor(g)) / QQ(2)
                j += 1
                if n == j:
                    i += 1
                    j = i
        elif hasattr(arg1, '_matrix_'):
            # LM./.MM FiniteQuadraticModule(list_of_orders/matrix, Gram matrix):
            G = arg1._matrix_().change_ring(QQ)
        else:
            raise TypeError("{0}: should be None, a matrix or a list".format(arg1))
    # print("M=",M)
    # print("G=",G)
    return FiniteQuadraticModule_ambient(M, G, **args)


def FiniteQuadraticModuleRandom(discbound=100, normbound=100, verbose=0):
    """
    Returns a random finite quadratic module with discriminant within the discriminant bound.
    It is generated by an element omega with denominator ideal bounded by "normbound" and in a
    number field with discriminant bounded by the "normbound".

    INPUT:

    - ``discbound`` -- integer, bound for the discriminant
    - ``normbound`` -- integer, bound for the norm
    - ``verbose`` -- integer

    EXAMPLES::

        sage: from fqm_weil.all import FiniteQuadraticModuleRandom
        sage: from fqm_weil.all import FiniteQuadraticModule_ambient
        sage: F = FiniteQuadraticModuleRandom(discbound=10,normbound=100)
        sage: isinstance(F, FiniteQuadraticModule_ambient)
        True

    """
    D = ZZ(0)
    while not D.is_fundamental_discriminant():
        D = ZZ.random_element(-discbound, discbound)
    K = QuadraticField(D, names='a')
    beta = 0
    while 0 == beta:
        beta = K.random_element()
    alpha = 0
    while 0 == alpha:
        alpha = K.random_element()
    om = beta / alpha
    log.debug(f"D={D}")
    log.debug(f"K={K}")
    log.debug(f"om={om}")

    class Redo(Exception):
        pass
    try:
        if om.denominator_ideal().absolute_norm() > normbound:
            raise Redo
        A = FiniteQuadraticModule(om / om.parent().gen())
        if A.is_trivial():
            raise Redo
        N = A.kernel()
        log.debug(f"A={A}")
        log.debug(f"|A.list()|={A}")
        log.debug(f"N={N}")
        if not N.is_isotropic():
            raise Redo
        if max(map(max, A.gram().rows())) == 0 and min(map(min, A.gram().rows())) == 0:
            raise Redo
        if not A.is_nondegenerate():
            raise Redo
        B = A.quotient(N)
        if not B.is_nondegenerate():
            msg = "Quotient by Kernel is nondegenerate!"
            log.critical(msg + f"A={A} Kernel(A)={N} A/N={B}")
            raise ArithmeticError(msg)
        return B
    except Redo:
        return FiniteQuadraticModuleRandom(discbound, normbound, verbose)
    except ArithmeticError:
        raise ArithmeticError("Quotient by Kernel is nondegenerate!")

#*****************************************************************************
#       Copyright (C) 2008 Nils-Peter Skoruppa <nils.skoruppa@uni-siegen.de>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************

r"""
Implementation of the category of finite quadratic modules.

A quadratic module is a pair $(M,Q)$, where $M$ is a finite abelian
group and where $Q:M\rightarrow \Q/Z$ is a quadratic form. The latter
means that $Q(tx)=t^2Q(x)$ for all integers $t$ and all $x$ in $M$,
and that $B(x,y):=Q(x+y)=Q(x)-Q(y)$ defines a bilinear map
$B: M \times M \rightarrow \Q/Z$. A morphism $f:A \rightarrow B$
between quadratic modules $A=(M,Q) and $B=(N,R)$ is a homomorphism
of abelian groups $F:M \rightarrow N$ such that $R \circ f = Q$. 

Quadratic modules show up naturally in the theory of Weil
representations, which in turn are needed e.g. in the theory of Jacobi
forms or elliptic modular forms of integral or half integral
weight. This is due to the fact, that every irreducible representation of $SL(2,\ZZ)$
whose kernel is a congruence subgroup is contained in a Weil
representation.


TECHNICAL NOTE

A finite abelian group $M$ is given as a set of $n$
generators $a,b,c..,$ and an $n \times n$-matrix $R$ encoding the
relations among the generators: $(a,b,c,...)R = 0$. A (finite) quadratic form
on $M$ is then given by its Gram matrix
$$
    G \equiv \frac 12
    \begin{pmatrix}
    B(a,a) & B(a,b) & B(a,c) & \dots
    \\ B(b,a) & B(b,b) & B(b,c) \dots
    \\
    \dots \end{pmatrix}
    \bmod \ZZ^{n \times n}.
$$
The finite quadratic module is thus isomorphic to
$$
    (\ZZ^n/R\ZZ^n, x+R\ZZ^n \mapsto x^tGx)
$$
via the map$(x_a, x_b, \dots) + R\ZZ^n \mapsto x_a a + x_b b + \cdots.$
Accordingly a typical initialization of a finite quadratic module would
be to provide the integer matrix $R$ and the rational matrix $G$.


REMARK

Many of the mathematically more meaningful methods of the class FiniteQuadraticModule_base assume that the
represented finite quadratic module is nondegenerate (i.e. $B(x,y)=0$
for all $y$ in $M$ is only possible for $x=0$). Applying such a method
to a degenerate module will raise an exception (TODO: what exception?).


REFERENCES

    [Sko] Nils-Peter Skoruppa, Finite quadratic modules and Weil representations,
          in preparation 2022


    TODO: find other references


    
AUTHORS:
    -- Hatice Boylan,      <boylan@mathematik.uni-siegen.de>
    -- Stephan Ehlen       <ehlen@mathematik.tu-darmstadt.de>
    -- Martin Frick,       <frick@mathematik.uni-siegen.de>
    -- Lars Fischer,       <lars.fischer@student.uni-siegen.de>
    -- Shuichi Hayashida,  <hayashida@mathematik.uni-siegen.de>
    -- Sebastian Opitz <opitz@mathematik.tu-darmstadt.de>
    -- Nils-Peter Skoruppa <nils.skoruppa@uni-siegen.de>
    -- Fredrik Stroemberg <fredrik314@gmail.com>

    
    The CN-Group started the development of this package as a seminar
    project at the university of Siegen. Its initial members have been:
    Hatice Boylan, Martin Frick, Lars Fischer, Shuichi Hayashida,
    Saber Mbarek, Nils-Peter Skoruppa

"""
from builtins import range
from builtins import str
import logging
from math import lcm

from sage.all import copy, cached_method, is_even, Sequence, \
    prod, valuation, randrange, xmrange, latex
from sage.arith.all import is_prime, kronecker, prime_divisors
from sage.arith.misc import is_prime_power, gcd
from sage.categories.commutative_additive_groups import CommutativeAdditiveGroups
from sage.categories.homset import HomsetWithBase
from sage.categories.morphism import    Morphism
from sage.graphs.graph import DiGraph
from sage.groups.abelian_gps.abelian_group import AbelianGroup_class
from sage.matrix.constructor import matrix, identity_matrix
from sage.matrix.matrix0 import Matrix
from sage.matrix.matrix_space import MatrixSpace
from sage.matrix.special import zero_matrix
from sage.modules.fg_pid.fgp_module import FGP_Module_class
from sage.modules.free_module_element import vector
from sage.modules.free_quadratic_module import FreeQuadraticModule
from sage.modules.module import Module
from sage.modules.torsion_quadratic_module import TorsionQuadraticModule
from sage.modules.vector_integer_dense import Vector_integer_dense
from sage.rings.all import ZZ, QQ, Integer, PolynomialRing
from sage.rings.number_field.number_field import CyclotomicField
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.structure.category_object import normalize_names
from sage.structure.element import Vector
from sage.structure.sequence import Sequence_generic

from .finite_quadratic_module_element import FiniteQuadraticModuleElement

log = logging.getLogger(__name__)

CF8 = CyclotomicField(8)
z8 = CF8.gen()

class FiniteQuadraticModule_base(FGP_Module_class, AbelianGroup_class):
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

        sage: from fqm_weil.modules.finite_quadratic_module.all import FiniteQuadraticModule_base
        sage: R = matrix(2,2,[2,1,1,2])
        sage: G = 1/2 * R^(-1)
        sage: A.<a,b> = FiniteQuadraticModule_base(R, G); A
        Finite quadratic module in 2 generators:
         gens: b, b
         form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2
        sage: a == b
        True
        sage: a is b
        False
        sage: a + b in A
        True
        sage: A = FiniteQuadraticModule_base(R, G, default_coords='fundamental'); A
        Finite quadratic module in 1 generator:
         gen: e
         form: 1/3*x^2

        # Check that elements and modules can be hashed

        sage: hash(a) is not None
        True
        sage: hash(A) is not None
        True
        sage: TestSuite(A).run()
    """

    Element = FiniteQuadraticModuleElement

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

        - ``default_coords`` -- string 'canonical' or 'fundamental' (default: 'canonical')
                    decides which format is used for the string and LaTeX representation

        EXAMPLES::

            sage: from fqm_weil.modules.finite_quadratic_module.all import \
            FiniteQuadraticModule_base
            sage: R = matrix(2,2,[2,1,1,2])
            sage: G = 1/2 * R^(-1)
            sage: A.<a,b> = FiniteQuadraticModule_base(R, G); A
            Finite quadratic module in 2 generators:
             gens: b, b
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2
            sage: A.<a> = FiniteQuadraticModule_base(R, G, default_coords='fundamental'); A
            Finite quadratic module in 1 generator:
             gen: a
             form: 1/3*x^2
            sage: G = 1/3 * R^(-1)
            sage: FiniteQuadraticModule_base(R, G)
            Traceback (most recent call last):
            ...
            ValueError: ([2 1]
            [1 2],[ 2/9 -1/9]
            [-1/9  2/9]): not a compatible pair


        .. TODO: check if R, G are matrices over ZZ, QQ, admit nonsquare R with rank == size G


        """
        if check:
            if not all((isinstance(R, Matrix), isinstance(G, Matrix))):
                raise TypeError(f"Input must be two matrices.")
            if not all((R.base_ring() in [QQ, ZZ], G.base_ring() in [QQ, ZZ])):
                raise TypeError(f"Input must be two matrices over ZZ or QQ.")
            if not R.is_square() or 0 == R.det() or R.denominator() != 1:
                raise ValueError(f"{R}: list not a regular integral square matrix")
            if not G.is_square() or not G.is_symmetric():
                raise ValueError(f"{G}: list not a symmetric square matrix")
            if R.ncols() != G.ncols():
                raise ValueError(f"({R},{G}): not a compatible pair")
            C0 = G*R
            C = R.transpose()*C0
            v = vector([C[i, i] for i in range(C.nrows())])
            if C0.denominator() > 2 or C.denominator() > 2 or v.denominator() > 1:
                raise ValueError(f"({R},{G}): not a compatible pair")
        self._default_coords = default_coords
        self._check = check
        if default_coords not in ['canonical', 'fundamental']:
            raise ValueError(f"Unknown value for 'default_coords': {default_coords}")
        # Convert R to an integer matrix
        R = R.change_ring(ZZ)
        # Keep the initializing pair $(R,G)$
        self.__iM = R
        self.__iG = self._reduce_mat(G)
        self.__iM.set_immutable()
        self.__iG.set_immutable()
        self._names = None

        # Replace $__iM$ by the unique $__R$ in $__iM * GL(n,\ZZ)$ which is
        # in lower Hermite normal form (i.e. is lower triangular and the rows
        # are reduced modulo their rightmost nonzero element).
        self.__R = matrix(ZZ, self.__iM).transpose().hermite_form().transpose()
        self.__G = self.__iG

        # If R is diagonal and not just a scalar remove row/columns of 1s
        if self.__R.is_diagonal() and self.__R.nrows() > 1:
            mask = [n for n, a in enumerate(self.__R.diagonal()) if a != 1]
            self.__R = self.__R.matrix_from_rows_and_columns(mask, mask)
            self.__G = self.__iG.matrix_from_rows_and_columns(mask, mask)
        # In case R is representing the trivial group
        if not self.__R or self.__R == matrix(ZZ, [[1]]):
            self._init_trivial_module()
            return

        # For simplicity and effectiveness in various internal computations
        # we use an equivalent form $(__E,__J)$ of our quadratic module,
        # where $__E$ is the diagonal matrix formed from the elementary divisors of $__R$
        # in descending order, and where superfluous $1$'s are thrown out.
        # The system of generators $e_i + __E\ZZ^m$, where $e_i$ is the standard basis of $\ZZ^m$
        # are in the sequel called 'the fundamental system of generators'. 
        # TODO: In addition, $J$ should be put in Jordan form
        D, U, V = matrix(ZZ, self.__R).dense_matrix().smith_form()
        # print("D=",D)
        # print("U=",U)
        # print("V=",V)
        # # Here $D = U * __R * V$
        mask = [n for n, a in enumerate(D.diagonal()) if a != 1] or []
        self.__E = D.matrix_from_rows_and_columns(mask, mask).sparse_matrix()
        T = U**(-1)
        # print("T=",T)
        # print("iG=", self.__iG)
        self.__J = self._reduce_mat(
            (T.transpose()
             * self.__G
             * T).matrix_from_rows_and_columns(mask, mask))
        # Transformation matrices:
        # can_sys = fun_gen * C2F, fun_sys = can_sys * F2C
        self.__C2F = U.matrix_from_rows(mask)
        self.__F2C = T.matrix_from_columns(mask)
        # print("F2C=",self.__F2C)
        # print("C2F=",self.__C2F)
        # Set the relations, Gram matrix and ngens to be used for the output
        # Note:
        #  __G is the full gram matrix with respect to "standard" (canonical) coordinates.
        #  __J is the Gram matrix wrt the fundamental system of generators.
        #
        self.__elementary_divisors = tuple(self.__E.diagonal())
        # Describe the underlying group as ZZ^n / R for both fundamental and
        # canonical coordinates.

        self.__relations = self.__R
        self.__ngens_canonical = self.__relations.ncols()
        self.__ngens_fundamental = len(self.__elementary_divisors)

        # print("names=",names)
        if default_coords == 'canonical':
            ngens = self.__ngens_canonical
            self._gens_orders = tuple(self.__R.diagonal())
        else:
            ngens = self.__ngens_fundamental
            self._gens_orders = self.__elementary_divisors
        if names is None:
            names = "e"
        if isinstance(names, tuple) and len(names) > ngens:
            raise ValueError(f"The number of variables {names} is different from the number of " +
                             f"generators: {ngens}")
        # print("default coords=",default_coords)
        # print("ngsns=",ngens)
        # print("names=",names)
        names = normalize_names(ngens, names)
        self._assign_names(names)
        # print("D=", D)
        # print("R=",self.__R)
        # print("J=", self.__J)
        # print("E=",self.__E)
        # print("iG=", self.__iG)
        # print("G=", self.__G)
        # print("C2F=",self.__C2F)
        # print("divisors=",self.__elementary_divisors)
        # The abelian group parent object is created using the canonical generators
        # not the "fundamental system".
        V = ZZ**ngens
        if self._default_coords == 'canonical':
            W = V.submodule(self.__R)
        else:
            W = V.submodule(self.__E)
        FGP_Module_class.__init__(self, V, W)
        # self.__relations = self.W().basis_matrix()
        # self.__ngens_canonical = self.__relations.ncols()
        # print("relations=",self.__relations)
        # print("relations=", self.__ncols)
        # print("ngens canonical=",self.__ngens_canonical)
        # print("ngens fundamental=",self.__ngens_fundamental)
        # zero of self
        self._zero = self.element_class(self, 0)
        # list of possible x_c's
        self._xcs = {}

    def _init_trivial_module(self):
        """
        Set all properties of a trivial group.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: T = FiniteQuadraticModule()
            sage: T.is_trivial()
            True
            sage: T.is_nondegenerate()
            True
            sage: T.relations()
            [1]
            sage: T.ngens()
            1
            sage: T.gram()
            [0]
            sage: T.gens()
            (0,)
            sage: T1 = FiniteQuadraticModule()
            sage: T1._init_trivial_module()
            sage: T1.is_trivial()
            True
            sage: T1 == T
            True

        """
        self.__J = identity_matrix(ZZ, 1)
        self.__E = identity_matrix(ZZ, 1)
        self.__R = identity_matrix(ZZ, 1)
        self.__C2F = identity_matrix(ZZ, 1)
        self.__F2C = identity_matrix(ZZ, 1)
        self.__G = zero_matrix(ZZ, 1)
        self.__iG = zero_matrix(ZZ, 1)
        self.__iM = zero_matrix(ZZ, 1)
        self.__elementary_divisors = (ZZ(1),)
        self.__ngens_fundamental = 1
        self.__ngens_canonical = 1
        self._gens_orders = (ZZ(1),)
        self.__relations = identity_matrix(ZZ, 1)
        self._assign_names(("e",))
        FGP_Module_class.__init__(self, ZZ**1, ZZ**1)

    # Private methods to access the internal properties
    def _C2F(self):
        r"""
        Return the matrix which transforms canonical to fundamental coordinates by
        multiplication from the left.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: A._C2F()
            [-2  1]
            sage: x = A.0 + A.1
            sage: A._C2F()*vector(x.c_list()) == vector(x.list())
            True

        """
        return self.__C2F

    def _F2C(self):
        r"""
        Return the matrix which transforms canonical to fundamental coordinates.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: A._F2C()
            [0]
            [1]
            sage: x = A.0 + A.1
            sage: A._F2C()*vector(x.list()) == vector(x.c_list())
            True
        """
        return self.__F2C

    def __hash__(self):
        """ Return the Hash of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F = FiniteQuadraticModule([2,2])
            sage: G = FiniteQuadraticModule([2,2])
            sage: hash(F) == hash(G)
            True
        """
        return hash(self._cache_key())

    def _cache_key(self):
        """
        Return the cache key of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F = FiniteQuadraticModule([2,2])
            sage: F._cache_key()
            (
                                                           [4 0]  [1/8   0]
            'FiniteQuadraticModule_ambient_with_category', [0 4], [  0 1/8], True, ('e0', 'e1')
            )
        """
        return self.__class__.__name__, self.__iM, self.__iG, self._check, self._names

    def _latex_(self):
        r""" LaTeX representation of this finite quadratic module.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2.27^-3')
            sage: latex(A) # doctest: +NORMALIZE_WHITESPACE
             \left(\left\langle \mathbb{Z}^{5}/\left(\begin{array}{rrrrr}
                3 & 0 & 0 & 0 & 0 \\
                0 & 3 & 0 & 0 & 0 \\
                0 & 0 & 27 & 0 & 0 \\
                0 & 0 & 0 & 27 & 0 \\
                0 & 0 & 0 & 0 & 27
             \end{array}\right)\mathbb{Z}^{5} \right\rangle,...
        """
        n = self.ngens(coords=self._default_coords)
        v = vector(PolynomialRing(QQ, 'x', n).gens())
        form = v.dot_product(self.gram(coords=self._default_coords) * v)
        if self._default_coords == 'canonical':
            mat = self.__R
        else:
            mat = self.__R
        Zn = f"\\mathbb{{Z}}^{{{n}}}"
        return f'\\left(\\left\\langle {Zn}/{latex(mat)}{Zn} \\right\\rangle,' \
               f'{latex(form)}\\right)'

    def _repr_(self):
        r""" String representation of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2'); A
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/2*x0*x1
            sage: FiniteQuadraticModule()
            Trivial finite quadratic module.
        """
        if self.is_trivial():
            return "Trivial finite quadratic module."
        n = self.ngens(coords=self._default_coords)
        v = vector(PolynomialRing(QQ, 'x', n).gens())
        gens = ', '.join([str(x) for x in self.gens(coords=self._default_coords)])
        form = v.dot_product(self.gram(coords=self._default_coords) * v)
        s = "s" if n > 1 else ""
        return f"Finite quadratic module in {n} generator{s}:\n gen{s}: {gens}\n form: {form}"\

    ###################################
    # Providing struct. defining items
    ###################################

    @cached_method()
    def ngens(self, coords=None):
        r"""
        Return the number of generators of the underlying abelian group.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F = matrix(QQ, 3, 3, [2, 1, 5, 1, 34, 19, 5, 19, 6]); F
            [ 2  1  5]
            [ 1 34 19]
            [ 5 19  6]
            sage: A = FiniteQuadraticModule(F); A
            Finite quadratic module in 3 generators:
                 gens: 27*e2, 921*e2, e2
                 form: 157/1960*x0^2 + 891/980*x0*x1 + 13/1960*x1^2 + 151/980*x0*x2 + 33/980*x1*x2 + 1893/1960*x2^2
            sage: A.ngens()
            3
            sage: A.ngens(coords='fundamental')
            1
            sage: A = FiniteQuadraticModule(F, default_coords='canonical'); A
            Finite quadratic module in 3 generators:
             gens: 27*e2, 921*e2, e2
             form: 157/1960*x0^2 + 891/980*x0*x1 + 13/1960*x1^2 + 151/980*x0*x2 + 33/980*x1*x2 + 1893/1960*x2^2
            sage: A.ngens()
            3
            sage: R = matrix(2,2,[2,1,1,2])
            sage: G = 1/2 * R^(-1)
            sage: F = FiniteQuadraticModule(R, G, default_coords='fundamental')
            sage: F.ngens()
            1
            sage: F.ngens(coords='canonical')
            2
        """
        if coords == 'fundamental' or (self._default_coords == 'fundamental' and not coords):
            return self.__ngens_fundamental
        return self.__ngens_canonical

    def gen(self, i=0, coords=None):
        r"""
        Return the $i$-th generator of the underlying abelian group.
        
        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([2, 4, 8]); A
            Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/8*x0^2 + 1/16*x1^2 + 1/32*x2^2
            sage: A.gen(0)
            e0
            sage: A.gen(1)
            e1
            sage: A.gen(2)
            e2
            sage: A.1
            e1
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: A.gen(0)
            e1
            sage: A.gen(1)
            e1
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]), default_coords='fundamental')
            sage: A.gen(0)
            e
            sage: A.gen(1)
            Traceback (most recent call last):
            ...
            ValueError: Input index 1 is out of bounds: need 0<=i<1



        This module has only one non-zero generator ::

            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.gen(0)
            e


        """
        coords = coords or self._default_coords
        ngens = self.ngens(coords=coords)
        if i < 0 or i >= ngens:
            raise ValueError(f"Input index {i} is out of bounds: need 0<=i<{ngens}")
        x = [0] * ngens
        x[i] = 1
        return self(x, coords=coords)

    def _element_constructor_(self, x, check=True, **kwds):
        r"""
        Construct an element of this finite quadratic module.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([2, 4, 8])
            sage: A([1,1,1])
            e0 + e1 + e2
            sage: A([1])
            Traceback (most recent call last):
            ...
            ValueError: Input vector has wrong length: 1 != 3
            sage: A([1.5,1.5,1.5])
            Traceback (most recent call last):
            ...
            TypeError: unable to convert 1.50000000000000 to an element of Integer Ring
            sage: F=FiniteQuadraticModule('2_1.3')
            sage: F([1,2,3])
            Traceback (most recent call last):
            ...
            ValueError: Input vector has wrong length: 3 != 2

            # Make sure that we can create elements using the correct coordinates
            # when the fundamental coordinates have the same length as the canonical

            sage: A = FiniteQuadraticModule([11,33]);
            sage: B = FiniteQuadraticModule('2_1')
            sage: (A+B).relations()[0,0]
            22
            sage: (A+B).elementary_divisors()
            (2, 22, 66)
            sage: (A+B)([1,0,0]).order()
            22
            sage: (A+B)([1,0,0], coords='fundamental').order()
            2
            sage: A.<a,b,c,d> = FiniteQuadraticModule('23^4.2_2^4.3',default_coords='fundamental')
            sage: A(a.lift())
            a
            sage: A.<a,b,c,d,e,f,g,h,j> = FiniteQuadraticModule('23^4.2_2^4.3')
            sage: A(a.lift())
            a


        """
        if isinstance(x, self.element_class):
            if check and x.parent() != self:
                raise ValueError(f"Can not construct an element of {self} from {x}")
            # Only return x if the parent *is* self to avoid coercion problems.
            if x.parent() is self:
                return x
            x = x.list()
        if check:
            if x != 0 and not isinstance(x, (list, tuple, Vector)):
                raise TypeError(f"Can not construct finite quadratic module element from {x}.")
            if isinstance(x, (list, tuple, Vector)):
                # Check that the length is correct.
                use_fundamental_coords = kwds.get('coords') == 'fundamental' or \
                                         (not kwds.get('coords') and
                                          self._default_coords == 'fundamental')
                if use_fundamental_coords and len(x) != self.__ngens_fundamental:
                    msg = f"Input vector has wrong length: {len(x)} != {self.__ngens_fundamental}"
                    raise ValueError(msg)
                if not use_fundamental_coords and len(x) != self.__ngens_canonical:
                    msg = f"Input vector has wrong length: {len(x)} != {self.__ngens_canonical}"
                    raise ValueError(msg)
        return self.element_class(self, x, check=check, **kwds)

    def coordinate_vector(self, x, reduce=False):
        r"""
        Return the coordinates of x with respect to the fundamental system.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([2, 4, 8])
            sage: A.coordinate_vector(A.0), A.coordinate_vector(A.1), A.coordinate_vector(A.2)
            ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            sage: A.<a,b,c,d> = FiniteQuadraticModule('23^4.2_2^4.3',default_coords='fundamental')
            sage: A.coordinate_vector(A.gens()[0])
            (1, 0, 0, 0)
            sage: A.<a,b,c,d,e,f,g,h,j> = FiniteQuadraticModule('23^4.2_2^4.3')
            sage: A.coordinate_vector(A.gens()[0])
            (0, 0, 0, 114)
        """
        return vector(x.list())

    def gens(self, coords=None):
        r"""
        Return a tuple of generators for self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([2, 4, 8]); A
            Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/8*x0^2 + 1/16*x1^2 + 1/32*x2^2
            sage: A.gens()
            (e0, e1, e2)
            sage: A.1
            e1
            sage: G = matrix([[3/71,12/71],[12/71,48/71]])
            sage: R = matrix([[1,0],[53,71]])
            sage: F = FiniteQuadraticModule(R,G)
            sage: F.gens()
            (18*e1, e1)

        This module has only one non-zero generator::

            sage: F=FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.gens()
            (e,)

        """
        # if self._default_coords == 'canonical' and coords == 'fundamental':
        #     return self.fgens()
        # elif coords is None:
        #     coords = self._default_coords
        coords = coords or self._default_coords
        return tuple([self.gen(i, coords=coords) for i in
                      range(self.ngens(coords=coords))])

    def fgens(self):
        r"""
        Return a fundamental system for the underlying abelian group.

        .. NOTES:
            A fundamental system of a finite abelian group $A$ is a
            set of generators $a_i$ such that $A$ equals the direct sum of
            the cyclic subgroups $\langle a_i \rangle$ generated by the
            $a_i$, and if, for each $i$, the order of $a_i$ equals the
            $i$-th elementary divisor of $A$.

            This method returns a fundamental system (which is, in fact,
            the one which was chosen
            when the quadratic module was initialized, and with respect to
            which all internal computations are actually performed).

            If the finite quadratic module self is initialized with canonical
            coordinates as the default then this method is faster than calling
            .gen(i, coords='fundamental') for each i.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F=FiniteQuadraticModule('2_1.3')
            sage: F.fgens()
            (e0 + e1,)


        """
        return tuple([self(list(x), coords='fundamental') for x in
                      identity_matrix(ZZ, len(self.elementary_divisors()))])

    def relations(self, coords=None):
        r"""
        Return a matrix in Hermite normal form describing the relations
        satisfied by the generators (see class doc string for details).

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^-2'); A
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2
            sage: A.relations()
            [2 0]
            [0 2]

        This module has only one non-zero generator.

            sage: F=FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.relations()
            [5]
        """
        if coords == 'fundamental':
            return self.__E
        return self.__relations

    def gram(self, coords=None):
        r"""
        Return the Gram matrix of the quadratic form with respect to the generators
        (as rational matrix).
        
        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F = FiniteQuadraticModule([2,2])
            sage: F.gram()
            [1/8   0]
            [  0 1/8]
            sage: F=FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.gram()
            [2/5]
            sage: A = FiniteQuadraticModule('11^-7.2^-2',check=True)
            sage: A.gram(coords='canonical')
            [2/11    0    0    0    0    0    0    0    0]
            [   0 2/11    0    0    0    0    0    0    0]
            [   0    0 2/11    0    0    0    0    0    0]
            [   0    0    0 2/11    0    0    0    0    0]
            [   0    0    0    0 2/11    0    0    0    0]
            [   0    0    0    0    0 2/11    0    0    0]
            [   0    0    0    0    0    0 1/11    0    0]
            [   0    0    0    0    0    0    0  1/2  1/4]
            [   0    0    0    0    0    0    0  1/4  1/2]
            sage: A.gram(coords='fundamental')
            [ 2/11     0     0     0     0     0     0]
            [    0  2/11     0     0     0     0     0]
            [    0     0  2/11     0     0     0     0]
            [    0     0     0  2/11     0     0     0]
            [    0     0     0     0  1/11     0     0]
            [    0     0     0     0     0 15/22   1/4]
            [    0     0     0     0     0   1/4 15/22]

        """
        if coords == 'fundamental':
            return self.__J
        return self.__G

    def gram_bilinear(self, coords=None):
        r"""
        Return the Gram matrix of the bilinear form with respect to the generators
        (as rational matrix).

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F = FiniteQuadraticModule([2,2])
            sage: F.gram_bilinear()
            [1/4   0]
            [  0 1/4]
            sage: F=FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.gram_bilinear()
            [4/5]
            sage: A = FiniteQuadraticModule('11^-7.2^-2',check=True)
            sage: A.gram_bilinear(coords='canonical')
            [4/11    0    0    0    0    0    0    0    0]
            [   0 4/11    0    0    0    0    0    0    0]
            [   0    0 4/11    0    0    0    0    0    0]
            [   0    0    0 4/11    0    0    0    0    0]
            [   0    0    0    0 4/11    0    0    0    0]
            [   0    0    0    0    0 4/11    0    0    0]
            [   0    0    0    0    0    0 2/11    0    0]
            [   0    0    0    0    0    0    0    1  1/2]
            [   0    0    0    0    0    0    0  1/2    1]
            sage: A.gram_bilinear(coords='fundamental')
            [ 4/11     0     0     0     0     0     0]
            [    0  4/11     0     0     0     0     0]
            [    0     0  4/11     0     0     0     0]
            [    0     0     0  4/11     0     0     0]
            [    0     0     0     0  2/11     0     0]
            [    0     0     0     0     0 15/11   1/2]
            [    0     0     0     0     0   1/2 15/11]
        """
        return 2*self.gram(coords)

    def elementary_divisors(self):
        r"""
        Return the elementary divisors of this as an abelian group.

        ...NOTE: These correspond to the "elementary_divisors" in the AbelianGroup class
            but are sometimes also called "invariant factors" as we have d_1 | d_2 | ... | d_n
            rather than prime powers.


        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]); A
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2
            sage: A.elementary_divisors ()
            (22, 66)
        """
        return self.__elementary_divisors

    ###################################
    # Coercion
    ###################################

    def __call_test__(self, x, coords=None):
        r"""
        Coerce object into a finite quadratic element belonging to this finite quadratic module.

        We coerce
        - an element of this module,
        - a list of coordinates with respect to the
          fundamental generators,
        - the integer $0$.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F=FiniteQuadraticModule('2_1.3', default_coords='canonical')
            sage: F([1,2])
            e0 + 2*e1
            sage: F(F.gens()[0])
            e0
            sage: F(0)
            0
            sage: F=FiniteQuadraticModule('2_1.3')
            sage: F([1,2,3])
            Traceback (most recent call last):
            ...
            ValueError: Input vector has wrong length: 3 != 2
            sage: F([1,2])
            e0 + 2*e1
            sage: F([5], coords='fundamental')
            e0 + 2*e1
            sage: F(F.gens()[0])
            e0
            sage: F(0)
            0
            sage: F(1)
            Traceback (most recent call last):
            ...
            TypeError: Can not construct finite quadratic module element from 1.
            sage: x = FiniteQuadraticModule('3').gens()[0]
            sage: F(x)
            Traceback (most recent call last):
            ...
            TypeError: unable to convert Finite quadratic module in 2 generators:...
        """
        if isinstance(x, FiniteQuadraticModuleElement):
            if x.parent() is self:
                return x
            raise ValueError(f"Can not coerce {x} with parent {x.parent()} to {self}")
        if isinstance(x, list):
            return FiniteQuadraticModuleElement(self, x, coords=coords)
        if isinstance(x, (Integer, int)) and x == 0:
            return FiniteQuadraticModuleElement(self, 0)
        raise TypeError("cannot coerce {0} to an element of {1}".format(x, self))

    def identity(self):
        """
        Identity in this group, i.e. zero

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F=FiniteQuadraticModule('2_1.3')
            sage: F.identity()
            0
        """
        return self(0)

    ###################################
    # Invariants
    ###################################

    def order(self):
        r"""
        If self is the quadratic module $(M,Q)$, return the order of $M$.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A.order()
            1452
        """
        return prod(e for e in self.elementary_divisors())

    def exponent(self):
        r"""
        If self is the quadratic module $(M,Q)$, then return the exponent of the
        abelian group $M$.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A.exponent()
            66
        """
        return max(self.elementary_divisors())

    def level(self):
        r"""
        If self is the quadratic module $(M,Q)$, then return the smallest positive integer $l$
        such that $l\cdotQ = 0$.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A.level()
            132
        """
        H = copy(self.__J)
        for i in range(H.ncols()):
            for j in range(i+1, H.ncols()):
                H[i, j] = 2*H[i, j]
                H[j, i] = H[i, j]
        return H.denominator()

    def tau_invariant(self, p = None):
        r"""
        Return +1  or -1 accordingly  as the order of the underlying abelian group
        (resp. the largest power of $p$ dividing this order)
        is a perfect square or not.
        
        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F=FiniteQuadraticModule('2_1.3')
            sage: F.tau_invariant()
            -1
            sage: FiniteQuadraticModule('4_1').tau_invariant()
            1

        """
        q = self.order()
        if p is None:
            return +1 if q.is_square() else -1
        return +1 if is_even(q.valuation(p)) else -1

    @cached_method
    def sigma_invariant(self, p=None):
        r"""
        If this quadratic module equals $A=(M,Q)$, return
        $\sigma(A) = \sqrt{|M|}^{-1/2}\sum_{x\in M} \exp(-2\pi i Q(x))$

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: FiniteQuadraticModule('2_1').sigma_invariant()
            -zeta8^3
            sage: FiniteQuadraticModule('3').sigma_invariant()
            zeta8^2
            sage: FiniteQuadraticModule('5').sigma_invariant()
            -1
            sage: FiniteQuadraticModule('4_1').sigma_invariant()
            -zeta8^3
            sage: FiniteQuadraticModule('4^2').sigma_invariant()
            1
            sage: FiniteQuadraticModule('4^-2').sigma_invariant()
            1

        """
        return self.char_invariant(-1, p)[0]

    def witt_invariants(self):
        r"""
        Return the family $\{sigma(A(p)\}_p$ as dictionary,
        where $A$ is this module, $A(p)$ its $p$-part,
        and $p$ is running through the divisors of the exponent of $A$
        (see also A.sigma_invariant()).
        
        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A.witt_invariants()[3]
            (zeta8^2, -1)
            sage: FiniteQuadraticModule('2_1').witt_invariants()
            {2: (-zeta8^3, -1)}
            sage: FiniteQuadraticModule('3').witt_invariants()
            {3: (zeta8^2, -1)}
            sage: FiniteQuadraticModule('5').witt_invariants()
            {5: (-1, -1)}
            sage: FiniteQuadraticModule('4_1').witt_invariants()
            {2: (-zeta8^3, 1)}
            sage: FiniteQuadraticModule('4^2').witt_invariants()
            {2: (1, 1)}
            sage: FiniteQuadraticModule('4^-2').witt_invariants()
            {2: (1, 1)}

        """
        P = prime_divisors(self.exponent())
        d = dict()
        for p in P:
            s = self.sigma_invariant(p)
            t = self.tau_invariant(p)
            d[p] = (s, t)
        return d

    def char_invariant(self, s, p=None, debug=0) -> tuple:
        r"""
        If this quadratic module equals $A = (M,Q)$, return
        the characteristic function of $A$ (or $A(p)$ if $p$ is a prime)
        at $s$, i.e. return
        $$\chi_A (s)= |M|^{-1}\sum_{x\in M} \exp(2\pi i s Q(x))).$$

        .. NOTE::
            We apply the formula in [Sko, Second Proof of Theorem 1.4.1].

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: FiniteQuadraticModule('2_1').char_invariant(1)
            (zeta8, sqrt(1/2))
            sage: FiniteQuadraticModule('3').char_invariant(1)
            (-zeta8^2, sqrt(1/3))
            sage: FiniteQuadraticModule('5').char_invariant(1)
            (-1, sqrt(1/5))
            sage: FiniteQuadraticModule('4_1').char_invariant(1)
            (zeta8, 1/2)
            sage: FiniteQuadraticModule('4^2').char_invariant(1)
            (1, 1/4)
            sage: FiniteQuadraticModule('4^-2').char_invariant(1)
            (1, 1/4)
            sage: FiniteQuadraticModule('4^2.2_1').char_invariant(1)
            (zeta8, 1/4*sqrt(1/2))
            sage: FiniteQuadraticModule('4^-2.2_1').char_invariant(1)
            (zeta8, 1/4*sqrt(1/2))
        """
        s = s % self.level()
        if s == 0:
            return CF8(1), CF8(1)
        if p is not None and not is_prime(p):
            raise TypeError
        if p and 0 != self.order() % p:
            return CF8(1), CF8(1)
        _p = p
        jd = self.jordan_decomposition()
        ci = ci1 = 1
        for c in jd:
            # c[0]: basis, (prime p,  valuation of p-power n, dimension r,
            #                                   determinant d over p [, oddity o])
            p, n, r, d, *o = c.invariants()
            log.debug(f"c={c}")
            log.debug("p={p}, n={n}, r={r}, d={d}")
            if _p and p != _p:
                continue
            o = o[0] if o else None
            log.debug(f"o={o}")
            k = valuation(s, p)
            s1 = Integer(s/p**k)
            h = max(n-k, 0)
            log.debug(f"h={h}")
            q = p**h
            if p != 2:
                lci = z8**((r*(1-q)) % 8) * d**(h % 2) if h > 0 else 1
                lci1 = q**(-r) if h > 0 else 1
            elif k == n and o is not None:
                return 0, 0
            else:
                f = z8**o if o else 1
                lci = f * d**(h % 2) if h > 0 else 1
                lci1 = q**(-r) if h > 0 else 1
                if debug > 0:
                    print(f, d, lci)
            if 2 == p:
                lci = lci**s1
            log.debug(f"lci={lci}")
            log.debug(f"lci1={lci1}")
            ci *= lci * kronecker(s1, lci1**-1)
            ci1 *= lci1
        return ci, QQ(ci1).sqrt()

    def signature(self, p=-1):
        r"""
        Compute the p-signature of self.
        p=-1 is the real signature.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: FiniteQuadraticModule('2_1').signature()
            1
            sage: FiniteQuadraticModule('3').signature()
            6
            sage: FiniteQuadraticModule('5').signature()
            4
            sage: FiniteQuadraticModule('4_1').signature()
            1
            sage: FiniteQuadraticModule('4^2').signature()
            0
            sage: FiniteQuadraticModule('4^-2').signature()
            0
            sage: FiniteQuadraticModule('4^2.2_1').signature()
            1
            sage: FiniteQuadraticModule('11^-1.3.2_3^-1').signature(2)
            7
            sage: FiniteQuadraticModule('11^-1.3^-1.2_3^-1').signature()
            3
            sage: FiniteQuadraticModule('11^-1.3^-1.2_3^-1').signature()
            3
            sage: FiniteQuadraticModule('11^-1.3^-1.2_3^-1').signature(11)
            2
            sage: FiniteQuadraticModule('11^-1.3^-1.2_3^-1').signature(3)
            2
            sage: FiniteQuadraticModule('11^-1.3^-1.2_3^-1').signature(2)
            7
            sage: FiniteQuadraticModule('11^-1.3^-1.2_3^-1').signature(5)
            0


        """
        if p == -1:
            p = None
        inv = self.char_invariant(1, p)
        inv = inv[0].list()
        if inv.count(1) > 0:
            return inv.index(1)
        else:
            return inv.index(-1) + 4

    def Gauss_sum(self, c, x=0, p=None, algorithm='formula', check=False) -> tuple:
        r"""
        If this quadratic module equals $A = (M,Q)$, return
        the Gauss sum of $A$ (or $A(p)$ if $p$ is a prime)
        at $s$, i.e. return the root of unity
        $$\chi_A (c)= 1/\sqrt{|M||M[c]}\sum_{x\in M} \exp(2\pi i c Q(x))).$$

        INPUT:

        - ``c`` -- integer
        - ``x`` -- element of self
        - ``p`` -- prime
        - ``algorithm`` -- string (default: 'formula'; if 'sum' then a "naive" simple sum is used
                                                        only use this for testing)
        - ``check`` -- boolean(default: False) check the value by comparing naive algorithm and formula.
        .. NOTE::
            We apply the formulas in [Str, Sections 3] for each Jordan component.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2_1')
            sage: A.Gauss_sum(1, check=True)
            zeta8
            sage: A.Gauss_sum(1, A.gens()[0], check=True)
            -zeta8^3
            sage: A = FiniteQuadraticModule('3')
            sage: A.Gauss_sum(1, check=True)
            -zeta8^2
            sage: A.Gauss_sum(1, A.gens()[0], check=True)
            zeta24^2
            sage: A = FiniteQuadraticModule('5')
            sage: A.Gauss_sum(1, check=True)
            -1
            sage: FiniteQuadraticModule('4_1').Gauss_sum(1, check=True)
            zeta8
            sage: FiniteQuadraticModule('4^2').Gauss_sum(1, check=True)
            1
            sage: FiniteQuadraticModule('4^-2').Gauss_sum(1, check=True)
            1
            sage: FiniteQuadraticModule('2^-2').Gauss_sum(1, check=True)
            -1
            sage: FiniteQuadraticModule('2^-2').Gauss_sum(2, check=True)
            1
            sage: A = FiniteQuadraticModule('4^2.2_1')
            sage: A.Gauss_sum(1, check=True)
            zeta8
            sage: A.Gauss_sum(1, A.gens()[0], check=True)
            zeta8
            sage: FiniteQuadraticModule('4^-2.2_1').Gauss_sum(1, check=True)
            zeta8
            sage: FiniteQuadraticModule('4^-2.2_1').Gauss_sum(2, check=True)
            0

            sage: FiniteQuadraticModule('4^2').Gauss_sum(1, check=True)
            1
            sage: A = FiniteQuadraticModule('8_1')
            sage: A.Gauss_sum(8, 0, check=True)
            0
            sage: A.Gauss_sum(1, A.gens()[0], check=True)
            zeta16
            sage: A.Gauss_sum(1, 2*A.gens()[0], check=True)
            -zeta8^3
            sage: A.Gauss_sum(4, 2*A.gens()[0], check=True)
            0


        """
        if algorithm == 'naive':
            return self._gauss_sum_as_sum(c, x, p)
        elif algorithm != 'formula':
            raise ValueError("algorithm must be 'formula' or 'naive'")
        if check:
            formula_value = self.Gauss_sum(c, x, p, algorithm='formula', check=False)
            naive_value = self.Gauss_sum(c, x, p, algorithm='naive', check=False)
            if formula_value != naive_value:
                raise ArithmeticError(f"Check failed. {formula_value} != {naive_value}")
            return formula_value
        if c == 0:
            return CF8(1)
        if p is not None and not is_prime(p):
            raise ValueError("p must be a prime")
        if not x:
            return prod(comp.gauss_sum(c) for comp in self.jordan_decomposition() if p is None
                        or comp.p == p)

        xc = self.xc(c)
        if x == xc:
            return self._gauss_sum__xc(c, p)
        y = x - xc
        if y not in self.power_subgroup(c):
            return CF8(0)
        if c != 1:
            y = y / c
        argument = ZZ(self.level()*(-c*self.Q(y) - self.B(xc, y)))
        # Reduce common factors
        d = gcd(argument, self.level())
        zN = CyclotomicField(self.level()//d).gen()
        argument = argument // d
        return self._gauss_sum__xc(c, p)*zN**argument

    def _gauss_sum_as_sum(self, c, x=0, p=None) -> tuple:
        """

        EXAMPLES:

        """
        N = lcm(self.level(), 8)
        CFN = CyclotomicField(N)
        z = CFN.gens()[0]
        gauss_sum = sum(z ** (c * (self.Q(y) * N) + self.B(x, y) * N) for y in self
                        if p is None or y.order() % p == 0)
        M = self.order()
        order_of_kernel_mult_a = prod(gcd(c, o) for o in self.gens_orders())
        result = gauss_sum / CFN(ZZ(M * order_of_kernel_mult_a).sqrt())
        try:
            return CF8(result)
        except (TypeError, ValueError):
            return CFN(result)

    def _gauss_sum__xc(self, c, p0=None) -> tuple:
        r"""
        If this quadratic module equals $A = (M,Q)$, return
        the Gauss sum of $A$ (or $A(p)$ if $p$ is a prime)
        at $s$, i.e. return the root of unity
        $$\chi_A (c)= 1/\sqrt{|M||M[c]}\sum_{x\in M} \exp(2\pi i c Q(x))).$$

        INPUT:
            - `c` -- integer
            - `p0` -- optional prime

        .. NOTE::
            We apply the formulas in [Str, Sections 3] for each Jordan component.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: FiniteQuadraticModule('2_1')._gauss_sum__xc(1)
            zeta8
            sage: FiniteQuadraticModule('3')._gauss_sum__xc(1)
            -zeta8^2
            sage: FiniteQuadraticModule('5')._gauss_sum__xc(1)
            -1
            sage: FiniteQuadraticModule('4_1')._gauss_sum__xc(1)
            zeta8
            sage: FiniteQuadraticModule('4^2')._gauss_sum__xc(1)
            1
            sage: FiniteQuadraticModule('4^-2')._gauss_sum__xc(1)
            1
            sage: FiniteQuadraticModule('4^2.2_1')._gauss_sum__xc(1)
            zeta8
            sage: FiniteQuadraticModule('4^-2.2_1')._gauss_sum__xc(1)
            zeta8
        """
        if c == 0:
            return QQ(1), QQ(1)
        if p0 is not None and not is_prime(p0):
            raise ValueError("p0 must be a prime")
        if p0 and self.order() % p0 != 0:
            return QQ(1), QQ(1)
        q = 2**valuation(c, 2)
        result = QQ(1)
        for comp in self.jordan_decomposition():
            if comp.q == q:
                continue
            result *= comp.gauss_sum(c)
        return result

    ###################################
    # Deriving quadratic modules
    ###################################

    def __add__(self, B):
        r"""
        Return the orthogonal sum of quadratic modules $A + B$,
        where $A$ is this quadratic module.

        INPUT:

            - `B` -- quadratic module

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A + A
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/44*x0^2 + 1/132*x1^2 + 1/44*x2^2 + 1/132*x3^2
            sage: B = FiniteQuadraticModule('2_1')
            sage: B + 0 == B
            True
            sage: B + FiniteQuadraticModule() == B
            True
            sage: B + 1
            Traceback (most recent call last):
            ...
            ValueError: Addition of Finite Quadratic Module and 1 not defined.

        """
        if B == 0:
            return self
        # Type checking does not work properly here.
        # I.e. not isinstance(B, FiniteQuadraticModule_base) does not work.
        if not hasattr(B, 'gram') or not hasattr(B, 'relations'):
            msg = f"Addition of Finite Quadratic Module and {B} not defined."
            raise ValueError(msg)
        if B.is_trivial():
            return self
        return self.__class__(self.relations().block_sum(B.relations()),
                              self.gram().block_sum(B.gram(coords='canonical')),
                              check=False, default_coords=self._default_coords)

    def __radd__(self, left):
        r"""
        Compute left + self

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: B = FiniteQuadraticModule('2_1')
            sage: (B + A).is_isomorphic(A + B)
            True
            sage: 0 + B == B
            True
            sage: 1 + B
            Traceback (most recent call last):
            ...
            ValueError: Addition of Finite Quadratic Module and 1 not defined.

        """
        return self.__add__(left)

    def _mul_(self, n: Integer, switch_sides: bool = False):
        r"""
        Return the $n$ fold orthogonal sum of this quadratic module

        INPUT:

        - ``n`` -- integer

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33], default_coords='canonical');
            sage: A*2
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/44*x0^2 + 1/132*x1^2 + 1/44*x2^2 + 1/132*x3^2
            sage: A = FiniteQuadraticModule([11,33])
            sage: A*2
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/44*x0^2 + 1/132*x1^2 + 1/44*x2^2 + 1/132*x3^2
            sage: A*2 == 2*A
            True
            sage: A*(1/2)
            Traceback (most recent call last):
            ...
            TypeError: Argument n (= 1/2) must be an integer.

        """
        if not isinstance(n, Integer):
            raise TypeError("Argument n (= {0}) must be an integer.".format(n))
        if n < 0:
            raise ValueError("Argument n (= {0}) must be non-negative.".format(n))
        if n == 0:
            M = matrix(ZZ, 0)
            return self.__class__(M, M)
        if n > 0:
            return sum(self for j in range(n))

    def __ne__(self, other):
        r"""
        True if self is not equal to other.

        .. NOTE:: For details about the comparison see __eq__.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A*2 != A
            True
            sage: B = FiniteQuadraticModule('2_4^-2.3.11^2')
            sage: B != A
            True
            sage: C = FiniteQuadraticModule([33,11]);
            sage: C != A
            True
            sage: D = FiniteQuadraticModule(A.relations(),A.gram())
            sage: D != A
            False
            sage: A != 1
            True
            sage: FiniteQuadraticModule(A.relations(),A.gram(), names='f') != A
            True
        """
        return not self.__eq__(other)

    def __eq__(self, other):
        r"""
        Return True if other is a quadratic module having the same generator names,
        satisfying the same relations and having the same Gram matrix as this module.


        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33]);
            sage: A*2 == A
            False
            sage: B = FiniteQuadraticModule('2_4^-2.3.11^2')
            sage: B == A
            False
            sage: C = FiniteQuadraticModule([33,11]);
            sage: C == A
            False
            sage: D = FiniteQuadraticModule(A.relations(),A.gram())
            sage: D == A
            True
            sage: A == 1
            False
            sage: FiniteQuadraticModule(A.relations(),A.gram(), names='f') == A
            False

        """
        if not isinstance(other, self.__class__):
            return False
        return all([
            self._V == other._V,
            self._W == other._W,
            self.gram() == other.gram(),
            self._default_coords == other._default_coords,
            self.variable_names() == other.variable_names()
            ])

    def twist(self, s):
        r"""
        If self is the quadratic module $A = (M,Q)$, return the twisted module $A^s = (M,s*G)$.

        INPUT:

        - `s` -- integer

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('11^-1'); A
            Finite quadratic module in 1 generator:
             gen: e
             form: 1/11*x^2
            sage: B = A.twist(-1); B
            Finite quadratic module in 1 generator:
             gen: e
             form: 10/11*x^2
            sage: C = B.twist(11); C
            Finite quadratic module in 1 generator:
             gen: e
             form: 0
            sage: C.twist(1/2)
            Traceback (most recent call last):
            ...
            TypeError: Argument s (=1/2) must be an integer.

        """
        if not isinstance(s, (Integer, int)):
            raise TypeError(f"Argument s (={s}) must be an integer.")
        return self.__class__(self.relations(), s * self.gram(),
                              default_coords=self._default_coords)

    def __pow__(self, s):
        r"""
        Return the twist $A^s$.

        INPUT:

        - ``s`` -- integer

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('11^-1'); A
            Finite quadratic module in 1 generator:
             gen: e
             form: 1/11*x^2
            sage: B = A^-1; B
            Finite quadratic module in 1 generator:
             gen: e
             form: 10/11*x^2
            sage: C = B^11; C
            Finite quadratic module in 1 generator:
             gen: e
             form: 0
            sage: D = FiniteQuadraticModule('4^-2.5^2'); D
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/4*x0^2 + 1/4*x0*x1 + 1/4*x1^2 + 2/5*x2^2 + 2/5*x3^2
            sage: D = FiniteQuadraticModule('4^-2.5^2', default_coords='canonical'); D
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/4*x0^2 + 1/4*x0*x1 + 1/4*x1^2 + 2/5*x2^2 + 2/5*x3^2
            sage: E = D^2; E
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2 + 4/5*x2^2 + 4/5*x3^2
            sage: E.is_nondegenerate ()
            False

        """
        return self.twist(s)

    ###################################
    # Predicates
    ###################################

    def is_multiplicative(self):
        r"""
        Return False since finite quadratic modules are additive groups.
        
        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: var('x')
            x
            sage: K.<a> = NumberField(x^3-x+1)
            sage: A = FiniteQuadraticModule(1/(a+11))
            sage: A.is_multiplicative()
            False
            sage: A = FiniteQuadraticModule([11])
            sage: A.is_multiplicative()
            False
        """
        return False

    def is_nondegenerate(self):
        r"""
        Return True or False accordingly if this module is nondegenerate or not.

        .. NOTE:: We define a trivial module to be (trivially) nondegenerate.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: N = FiniteQuadraticModule(); N
            Trivial finite quadratic module.
            sage: N.is_nondegenerate()
            True
            sage: D = FiniteQuadraticModule('4^-2.5^2'); D
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/4*x0^2 + 1/4*x0*x1 + 1/4*x1^2 + 2/5*x2^2 + 2/5*x3^2
            sage: E = D^2; E
            Finite quadratic module in 4 generators:
             gens: e0, e1, e2, e3
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2 + 4/5*x2^2 + 4/5*x3^2
            sage: E.is_nondegenerate()
            False
            sage: E.kernel()
            < 2*e0, 2*e1 >

        """
        return self.kernel().order() == 1

    def is_nontrivial(self):
        r"""
        Check if self is nontrivial.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F = FiniteQuadraticModule('3.2_1')
            sage: F.is_nontrivial()
            True
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[1]]),Matrix(ZZ,[[0]]))
            sage: F.is_nontrivial()
            False

        """
        return self.order() > 1

    def is_trivial(self):
        r"""
        Check if self is trivial.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: F = FiniteQuadraticModule('3.2_1')
            sage: F.is_trivial()
            False
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[1]]),Matrix(ZZ,[[0]]))
            sage: F.is_trivial()
            True
        """
        return not self.is_nontrivial()

    def is_isomorphic(self, A):
        r"""
        Return True or False accordingly as this quadratic module
        is isomorphic to $A$ or not.

        INPUT:

        - `A` -- Finite Quadratic Module

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2_2^4')
            sage: B = FiniteQuadraticModule('2_6^-4')
            sage: A.is_isomorphic(B)
            True

        .. TODO:
            Extend this function so to include also degenerate modules.
            Maybe wo do not need to check all divisors?
        """
        if not isinstance(A, type(self)):
            return False
        if not self.is_nondegenerate() or not A.is_nondegenerate():
            raise TypeError('The quadratic modules to compare must be non degenerate')
        if self.elementary_divisors() != A.elementary_divisors():
            return False
        divs = self.level().divisors()
        for t in divs:
            if self.char_invariant(t) != A.char_invariant(t):
                return False
        return True

    def is_witt_equivalent(self, A):
        r"""
        Return True or False accordingly as this quadratic module
        is Witt equivalent to $A$ or not.

        INPUT:

        - ``A`` -- finite quadratic module

        EXAMPLES::
            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2_2^4')
            sage: B = FiniteQuadraticModule('2_6^-4')
            sage: A.is_witt_equivalent(B)
            True

        TODO:
            Extend this function so to also include degenerate modules.
        """
        if not isinstance(A, FiniteQuadraticModule_base):
            return False
        if not self.is_nondegenerate() or not A.is_nondegenerate():
            raise TypeError('The quadratic modules to compare must be non degenerate')
        if self.tau_invariant() == A.tau_invariant():
            a = self.witt_invariants()
            b = A.witt_invariants()
            if a == b:
                return True
        return False

    ###################################
    # Deriving other structures
    ###################################

    def as_torsion_quadratic_module(self, check=True):
        r"""
        Return an instance of TorsionQuadraticModule isomorphic to self if possible.
        Note that

        INPUT:

            - `check` -- check the resulting module.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: FiniteQuadraticModule('2_1').as_torsion_quadratic_module()
             Finite quadratic module over Integer Ring with invariants (2,)
             Gram matrix of the quadratic form with values in Q/4Z:
             [1]
            sage: FiniteQuadraticModule('3^-1').as_torsion_quadratic_module()
             Finite quadratic module over Integer Ring with invariants (3,)
             Gram matrix of the quadratic form with values in Q/3Z:
             [1]
            sage: FiniteQuadraticModule('3^1').as_torsion_quadratic_module()
            Traceback (most recent call last):
            ...
            NotImplementedError: Could not create a TorsionQuadraticModule from this ...
            sage: A = FiniteQuadraticModule('2_2^4')
            sage: A.as_torsion_quadratic_module()
            Finite quadratic module over Integer Ring with invariants (2, 2, 2, 2)
            Gram matrix of the quadratic form with values in Q/4Z:
            [1 0 0 0]
            [0 1 0 0]
            [0 0 0 1]
            [0 0 1 0]

        .. TODO:: See what is the correct way to deal with errors in the value module.
                  For instance '3^1' yields a TQM with values in ZZ/6ZZ instead of ZZ/3ZZ.
                  It might be that the TQMs only work for discriminant forms of ZZ-lattices and
                  not of ZZ_p - lattices.
        """
        G, d = self.gram()._clear_denom()
        Q = FreeQuadraticModule(ZZ, G.nrows(), inner_product_matrix=G)
        try:
            T = TorsionQuadraticModule(Q, Q.span(self.relations()))
            # Some extra checks that we have a correct torsion quadratic module.
            if check and T.value_module_qf().n != d:
                raise ArithmeticError(f"Value module is not ZZ/{d}ZZ")
            if check and T.invariants() != self.elementary_divisors():
                raise ArithmeticError(f"Invs: {T.invariants()} != {self.elementary_divisors()}")
            return T
        except Exception as e:
            raise NotImplementedError(f"Could not create a TorsionQuadraticModule from this "
                                      f"quadratic module. ERROR: {e}")

    def as_discriminant_module(self):
        r"""
        Return a half-integral matrix $F$ such that
        this quadratic module is isomorphic to the
        discriminant module
        $$D_F = (\ZZ^n/2F\ZZ^n, x + 2F\ZZ^n \mapsto \frac 14 F^{-1}[x] + \ZZ).$$

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: FiniteQuadraticModule('2_1').as_discriminant_module()
            Traceback (most recent call last):
            ...
            NotImplementedError


        .. TODO: Look up Wall if his proof is effective ...

        .. NOTE:
            If $D_F$ and $D_G$ are isomorphic then there exist even unimodular matrices $U$ and $V$ such that
            $F+U$ and $G+V$ are equivalent (viewed as quadratic forms) over $\Z$ (see [Sko]). 
        """
        raise NotImplementedError

    def orthogonal_group(self, gens=None, check=False):
        r"""
        Returns the orthogonal group of this quadratic module.

        .. NOTE:: At the moment we use the function from the corresponding 'torsion quadratic form'
            if it can be found.
            The warning and input description is taken from that function.

         .. WARNING::

            This is can be smaller than the orthogonal group of the bilinear form.

        INPUT:

        - ``gens`` --  a list of generators, for instance square matrices,
                       something that acts on ``self``, or an automorphism
                       of the underlying abelian group
        - ``check`` -- perform additional checks on the generators

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: T=FiniteQuadraticModule('3^-1').orthogonal_group(); T
            Group of isometries of
            Finite quadratic module over Integer Ring with invariants (3,)
            Gram matrix of the quadratic form with values in Q/3Z:
            [1]
            generated by 1 elements
            sage: list(T)
            [[1], [2]]
            sage: T=FiniteQuadraticModule('3^-3').orthogonal_group(); T
            Group of isometries of
            Finite quadratic module over Integer Ring with invariants (3, 3, 3)
            Gram matrix of the quadratic form with values in Q/3Z:
            [2 0 0]
            [0 2 0]
            [0 0 1]
            generated by 3 elements
            sage: T.gens()
            ([2 2 2]
             [1 1 2]
             [2 1 0],
             [2 2 2]
             [2 2 1]
             [2 1 0],
             [2 2 2]
             [2 2 1]
             [1 2 0])

        ..TODO: Implement an efficient algorithm for this (Nice topic for a master thesis!)
        """
        return self.as_torsion_quadratic_module().orthogonal_group(gens=gens, check=check)

    ###################################
    # Iterators
    ###################################

    def __iter__(self):
        r"""
        Return a generator over the elements of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^-1')
            sage: list(A)
            [0, e, 2*e]
            sage: [x for x in A]
            [0, e, 2*e]
            sage: list(FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2])))
            [0, e1, 2*e1]
            sage: A.<a,b,c,d,e,f,g,j,k> = FiniteQuadraticModule('23^4.2_2^4.3')
            sage: it = iter(A)
            sage: next(it)
            0
            sage: next(it)
            22*a + e + k

        """
        return(self(x, coords='fundamental') for x in xmrange(self.elementary_divisors()))

    def values(self):
        r"""
        If this is $(M,Q)$, return the values of $Q(x)$ ($x \in M$) as a dictionary d.

        OUTPUT
            dictionary -- the mapping Q(x) --> the number of elements x with the same value Q(x)

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A=FiniteQuadraticModule([1,3])
            sage: A.values()[7/12]
            2

        ..TODO::
            Replace the code by theoretical formulas
            using the Jordan decomposition.
            DONE: See the values function in the class JordanDecomposition  
        """
        valueDict = {}
        
        for x in self:
            v = self.Q(x)
            if v in valueDict:
                valueDict[v] += 1
            else:
                valueDict[v] = 1
                
        return valueDict

    ###################################
    # Auxiliary functions
    ###################################

    @staticmethod
    def _reduce(r):
        r"""
        Return the fractional part of $x$.

        EXAMPLES::

            sage: from fqm_weil.modules.finite_quadratic_module.all import \
            ....: FiniteQuadraticModule_base
            sage: FiniteQuadraticModule_base._reduce(297/100)
            97/100
        """
        return r - r.floor()

    @staticmethod
    def _reduce_mat(A):
        r"""
        Return the fractional part of the symmetric matrix $A$ where the off-diagonal elements
        are reduced modulo 1/2 and the diagonal elements modulo 1.

        EXAMPLES::

            sage: from fqm_weil.modules.finite_quadratic_module.all import \
            ....: FiniteQuadraticModule_base
            sage: F = matrix(3,3,[1,1/2,3/2,1/2,2,1/9,3/2,1/9,1]); F
            [ 1 1/2 3/2]
            [1/2   2 1/9]
            [3/2 1/9   1]
            sage: FiniteQuadraticModule_base._reduce_mat(F)
            [ 0   0   0]
            [ 0   0 1/9]
            [ 0 1/9   0]
        """
        B = matrix(QQ, A.nrows(), A.ncols())
        for i in range(A.nrows()):
            for j in range(A.ncols()):
                if i == j:
                    B[i, j] = FiniteQuadraticModule_base._reduce(A[i, j])
                else:
                    B[i, j] = FiniteQuadraticModule_base._reduce(2 * A[i, j]) / QQ(2)
        return B

    @staticmethod
    def _rdc(R, x):
        r"""
        Returns the $y \equiv x \bmod R\,ZZ^n$ in the fundamental
        mesh $\{R\zeta : 0 \le \zeta_i < 1\}$.
    
        INPUT:
        - ``x`` -- an integral vector of length n
        - ``R`` -- a regular nxn matrix in lower Hermite normal form

        NOTE:

            It is absolutely necessary that $R$ comes in lower Hermite
            normal form.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A._rdc(A.relations(),(1,1))
            [1, 1]
            sage: A._rdc(A.relations(),(1,100))
            [1, 34]
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F._rdc(F.relations(),[1,1])
            [1]
            sage: F._rdc(F.relations(),[10,1])
            [0]
            sage: F._rdc(F.relations(),[])
            []
        """
        if len(x) == 0:
            return []
        y = [0]*R.nrows()
        k = [0]*R.nrows()
        k[0], y[0] = divmod(Integer(x[0]), Integer(R[0, 0]))
        for i in range(1, R.nrows()):
            k[i], y[i] = divmod(Integer(x[i] - sum(R[i, j]*k[j] for j in range(i))), Integer(R[i, i]))
        return y

    def _f2c(self, x):
        r"""
        Transform coordinates w.r.t. to the internal fundamental system to the coordinates w.r.t. the generators.

        INPUT:

        -``x`` -- integer vector

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A._f2c([1,0])
            [1, 0]
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F._f2c([1])
            [1]
            sage: A.<a,b,c,d,e,f,g,h,j> = FiniteQuadraticModule('11^-7.2^-2')
            sage: A._f2c([0,0,0,0,0,0,10])
            [1, 0, 0, 0, 0, 0, 0, 0, 0]
            sage: A._f2c([])
            []
        """
        if not x:
            return []
        v = self.__F2C*vector(ZZ, x)
        return list(self._rdc(self.__R, v))

    def _c2f(self, x):
        r"""
        Transform coordinates w.r.t. the generators to coordinates w.r.t. the internal fundamental system.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A._c2f([1, 0])
            [1, 0]
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F._c2f([1])
            [1]
            sage: F._c2f([0])
            [0]
            sage: A.<a,b,c,d,e,f,g,h,j> = FiniteQuadraticModule('11^-7.2^-2')
            sage: A._c2f([1,0,0,0,0,0,0,0,0])
            [0, 0, 0, 0, 0, 0, 10]

        """
        v = self.__C2F*vector(ZZ, x)
        return list(self._rdc(self.__E, v))

    def Q(self, x):
        r"""
        Return the value $Q(x)$ (as rational reduced mod $\ZZ$).

        INPUT:
        - ``x`` -- a FiniteQuadraticModuleElement of self

        OUTPUT:

        rational number -- the value of the quadratic form on x
            
        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A.Q(A.gens()[0])
            1/44
            sage: A.Q(A.gens()[1])
            1/132
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.Q(F.gens()[0])
            2/5

        """
        c = vector(x.list(coords='fundamental'))
        return self._reduce(c.dot_product(self.__J * c))

    def B(self, x, y):
        r"""
        Return the value $B(x,y) = Q(x+y)-Q(x)-Q(y)$ (as rational reduced mod $\ZZ$).

        INPUT:

            - ``x`` -- FiniteQuadraticModuleElement of self
            - ``y`` -- FiniteQuadraticModuleElement of self

        OUTPUT:

            rational number -- the value of the bilinear form on x and y
        

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A.B(A.gens()[0],A.gens()[0])
            1/22
            sage: A.B(A.gens()[1],A.gens()[0])
            0
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.B(F.0,F.0)
            4/5
            sage: F.B(F.0,0)
            0
            sage: F.B(F.gens()[0],1)
            Traceback (most recent call last):
            ...
            ValueError: Need elements of self not x=e and y=1.
        """
        if x == 0 or y == 0:
            return 0
        if x not in self or y not in self:
            raise ValueError(f"Need elements of self not x={x} and y={y}.")
        c = vector(x.list(coords='fundamental'))
        d = vector(y.list(coords='fundamental'))
        return self._reduce(2 * c.dot_product(self.__J * d))

    #
    @staticmethod
    def _diagonalize(G, n):
        r"""
        Diagonalizes the matrix $G$ over the localisation $Z_(n)$.
        
        INPUT:

        - ``G`` -- a matrix with entries in $Z$
        - ``n`` -- an integer
        
        OUTPUT:

        A matrix $H$ and a matrix $T$ in $GL(r,Z_(n))$
        such that $H = T^tGT$ is in {\em Jordan decomposition form}.

        ..NOTES:

        Here {\em $H$ is in Jordan decomposition form} means the following:

        o $H$ is a block sum of matrices $B_j$ of size 1 or 2
        o For each prime power $q=p^l$ dividing $n$, one has
          - $H \bmod q$ is diagonal,
          - the sequence $gcd(q,H[i,i])$ is increasing,
          - for $i > 1$ one has $H[i,i] \equiv p^k \bmod q$ for some $k$
            unless $gcd(q, H[i-1,i-1]) < gcd(q, H[i,i])$,
        o if $q=2^l$ is the exact 2-power dividing $n$, then
          - $B_j$ is scalar or
            $B_j \equiv p^k [2,1;1,2] \bmod q$
            or $B_j \equiv p^k [0,1;1,0] \bmod q$ for some $k$.
          (- maybe later also a sign walk and oddity fusion normalisation)

        o An implementation could follow:
        [Sko 1] Nils-Peter Skoruppa, Reduction mod $\ell$ of Theta Series of Level $\ell^n$,
                arXiv:0807.4694

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A._diagonalize(A.relations(), 1)
            Traceback (most recent call last):
            ...
            NotImplementedError

        """
        raise NotImplementedError

    ###################################
    # Misc
    ###################################

    def an_element(self):
        r"""
        Returns an element of this finite quadratic module.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A.an_element()
            e0 + e1
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.an_element()
            e

        """
        return self([1]*self.ngens())
    
    def random_element(self, bound=None):
        """
        Return a random element of this group.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A.random_element() in A
            True
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: F.an_element() in F
            True
        """
        coords = [randrange(0, ed) for ed in self.elementary_divisors()]
        return self(coords)

    def cayley_graph(self):
        """
        Returns the cayley graph for this finite group, as a SAGE
        DiGraph object. To plot the graph with different colors

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A.<a,b> = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,6])); A
            Finite quadratic module in 2 generators:
             gens: 5*b, b
             form: 3/11*x0^2 + 10/11*x0*x1 + 1/11*x1^2
            sage: A=FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,6]),default_coords='fundamental')
            sage: A
            Finite quadratic module in 1 generator:
             gen: e
             form: 1/11*x^2
            sage: D = A.cayley_graph (); D
            Digraph on 11 vertices
            sage: g = D.plot(color_by_label=True, edge_labels=True)              

        TODO:
            Make the following work:
            D.show3d(color_by_label=True, edge_size=0.01, edge_size2=0.02, vertex_size=0.03)
            
            Adjust by options so that images is less cluttered
        
        NOTE:
            Copied from sage.groups.group.FiniteGroup.cayley_graph().
        """
        arrows = {}
        for x in self:
            arrows[x] = {}
            for g in self.gens():
                xg = x+g
                if not xg == x:
                    arrows[x][xg] = g
        return DiGraph(arrows)

    def _is_valid_homomorphism_(self, codomain, im_gens, base_map=None):

        r"""
        Return True if \code{im_gens} defines a valid homomorphism
        from self to codomain; otherwise return False.

        INPUT:

        - ``codomain`` -- Finite quadratic module
        - ``im_gens`` -- list of elements of the codomain (of length the number of gens of self)

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A._is_valid_homomorphism_(A,[A.0,A.1])
            True
            sage: A._is_valid_homomorphism_(A,[A.0,1])
            Traceback (most recent call last):
            ...
            TypeError: im_gens (=[e0, 1]) must belong to Finite quadratic module...
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: A._is_valid_homomorphism_(F,[F.0])
            Traceback (most recent call last):
            ...
            TypeError: im_gens (=[e]) must be the same length as the number of gens

        ..NOTE::

            If $A = \ZZ^n/R$ and $B = \ZZ^m/S$, then \code{im_gens}
            defines a valid homomorphism if $S^-1MS$ is an integral
            matrix, where $M$ is the matrix whose columns are the coordinates
            of the elements of \code{im_gens} w.r.t. the can. system.
        """
        if not isinstance(im_gens, (tuple, list)):
            im_gens = [im_gens]
        im_gens = Sequence(im_gens)
        if not im_gens.universe() is codomain:
            raise TypeError(f"im_gens (={im_gens}) must belong to {codomain}")
        if len(im_gens) != self.ngens():
            raise TypeError(f"im_gens (={im_gens}) must be the same length as the number of gens")
        coords = codomain._default_coords
        phi = matrix([x.list(coords=coords) for x in im_gens]).transpose()
        # print("phi=",phi)
        # print("codom relations=",codomain.relations())
        # print("self relations=", self.relations())
        return self._divides(codomain.relations(coords=coords),
                             phi * self.relations(coords=coords))

    @staticmethod
    def _divides(A, B):
        r"""
        Return  True  or  False  accordingly as  the  rational  matrix
        $A^{-1}*B$ is an integer matrix or not.

        EXAMPLES::

            sage: from fqm_weil.modules.finite_quadratic_module.all import \
            ....: FiniteQuadraticModule_base
            sage: M=Matrix(ZZ,[[10,5],[-5,-2]])
            sage: R=Matrix(ZZ,[[10,0],[0,2]])
            sage: FiniteQuadraticModule_base._divides(M,R)
            True
            sage: FiniteQuadraticModule_base._divides(R,M)
            False

        """
        ZZ_N_times_N = MatrixSpace(ZZ, A.nrows(), B.ncols())
        return A ** (-1) * B in ZZ_N_times_N

    def _Hom_(self, B, cat=None):
        r"""
        Return the set of morphism from this quadratic module to $B$.

        INPUT:

        - ``B`` -- Finite quadratic module
        - ``cat`` -- category

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S = A._Hom_(B); S
            Set of Morphisms from Finite quadratic module in 2 generators:
                 gens: e1, e1
                 form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 to Finite quadratic module in 4 generators:
                 gens: e1, e1, e3, e3
                 form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2 in ...
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]), default_coords='canonical')
            sage: B = 2*A
            sage: S = A._Hom_(B); S
            Set of Morphisms from Finite quadratic module in 2 generators:
                 gens: e1, e1
                 form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 to Finite quadratic module in 4 generators:
                 gens: e1, e1, e3, e3
                 form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2 in ...
        """
        if cat and not cat.is_subcategory(self.category()):
            raise TypeError(f"Conversion from category: {cat} is not implemented")
        if not isinstance(B, FiniteQuadraticModule_base):
            raise TypeError("B (={0}) must be finte quadratic module.".format(B))
        return FiniteQuadraticModuleHomset(self, B)

    def _convert_map_from_(self, S):
        if isinstance(S, FiniteQuadraticModule_base):
            return FiniteQuadraticModuleHomset(self, S)
        return None

    def hom(self, im_gens, codomain=None, check=None, base_map=None, category=None, **kwds):
        r"""
        Return the homomorphism from this module to the parent module of the
        elements in the list which maps the $i$-th generator of this module
        to the element \code{x[i]}.

        INPUT:

        - ``im_gens`` -- a list of $n$ elements of a quadratic module, where $n$ is the number
                 of generators of this quadratic module.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule([11,33])
            sage: A.hom([A.0,A.1])
            Homomorphism : Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2 --> Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2
            e0 |--> e0
            e1 |--> e1
            sage: A.hom([A.0,1])
            Traceback (most recent call last):
            ...
            TypeError: B (=Category of objects) must have universe a FiniteQuadraticModule.
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: A.hom([F.0])
            Traceback (most recent call last):
            ...
            TypeError: images (=[e]) do not define a valid homomorphism...
            sage: M = FiniteQuadraticModule('2^-2.2_1^1'); M
            Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2 + 1/4*x2^2
            sage: N = FiniteQuadraticModule('2_1^-3')
            sage: M.hom([N.0,N.1,2*N.2])
            Homomorphism : Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2 + 1/4*x2^2 --> Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/4*x0^2 + 1/2*x1^2 + 1/2*x1*x2 + 1/2*x2^2
            e0 |--> e0
            e1 |--> e1
            e2 |--> 0
        """
        v = Sequence(im_gens)
        B = v.universe()
        if not isinstance(B, FiniteQuadraticModule_base):
            raise TypeError("B (={0}) must have universe a FiniteQuadraticModule.".format(B))
        return self.Hom(B)(im_gens)


###################################
# MORPHISMS
###################################


class FiniteQuadraticModuleHomomorphism_im_gens (Morphism):
    r"""
    Implements elements of the set of morphisms between two quadratic modules.
    
    """
    
    def __init__(self, homset, im_gens, check=True):
        r"""
        INPUT:

            ``homset``  -- a set of modphisms between two quadratic modules
            ``im_gens`` -- a list of elements of the codomain.
            
        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B); S
            Set of Morphisms from Finite quadratic module in 2 generators:
             gens: e1, e1
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 to Finite quadratic module in 4 generators:
             gens: e1, e1, e3, e3
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2 in Category...
            sage: u = A.gens()[0]
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a]); f
             Homomorphism : Finite quadratic module in 2 generators:
             gens: e1, e1
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 --> Finite quadratic module in 4 generators:
             gens: e1, e1, e3, e3
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2
            e1 |--> e1
            e1 |--> e1
            sage: f(u)
            e1
            sage: f(2*u)
            2*e1
            sage: TestSuite(f).run()
        """
        Morphism.__init__(self, homset)
        if not isinstance(im_gens, Sequence_generic):
            if not isinstance(im_gens, (tuple, list)):
                im_gens = [im_gens]
            im_gens = Sequence(im_gens, homset.codomain())
        if check:
            if len(im_gens) != homset.domain().ngens():
                raise ValueError("number of images must equal number of generators")
            t = homset.domain()._is_valid_homomorphism_(homset.codomain(), im_gens)
            if not t:
                msg = "relations do not all (canonically) map to 0 under map determined by images" \
                      " of generators."
                raise ValueError(msg)
        self._im_gens = im_gens
        # We compute the images in the default coordinates of the domain:
        n = len(im_gens)
        self.__c_im_gens = []
        for x in homset.domain().gens():
            cos = x.list(coords=homset.domain()._default_coords)
            y = sum(im_gens[i]*cos[i] for i in range(n))
            self.__c_im_gens.append(y)

    def __ne__(self, other):
        """
        Return True is self is not equal to other, otherwise return False.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: u = A.gens()[0]
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: g = S([b,b])
            sage: f != g
            False
            sage: f != 1
            True
            sage: h = S([c,c])
            sage: f != h
            True

        """
        return not self.__eq__(other)

    def __eq__(self, other):
        """
        Return True is self is equal to other, otherwise return False.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: u = A.gens()[0]
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: g = S([b,b])
            sage: f == g
            True
            sage: f == 1
            False
            sage: h = S([c,c])
            sage: f == h
            False

        """

        if not isinstance(other, FiniteQuadraticModuleHomomorphism_im_gens):
            return False
        return all([self.parent() == other.parent(), self.im_gens()==other.im_gens()])

    def _add_(self, other):
        """
        Add other to this morphism.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: u = A.gens()[0]
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: g = S([b,b])
            sage: f + g
             Homomorphism : Finite quadratic module in 2 generators:
              gens: e1, e1
              form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 --> Finite quadratic module in 4 generators:
              gens: e1, e1, e3, e3
              form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2
             e1 |--> 2*e1
             e1 |--> 2*e1


        """
        im_gens = [self._im_gens[i] + y for (i, y) in enumerate(other.im_gens())]
        return self.parent()(im_gens)

    def im_gens(self):
        """
        Return the image of generators under self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: u = A.gens()[0]
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: g = S([c,c])
            sage: f.im_gens()
            [e1, e1]
            sage: g.im_gens()
            [e3, e3]

        """

        return self._im_gens

    def kernel(self):
        r"""
        Return the kernel of this morphism.

        ..NOTE:: This is currently implemented using brute-force.

        ..TODO:: Implement this in a more efficient way.


        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: f.kernel()
            < 0 >
            sage: A = FiniteQuadraticModule([6])
            sage: B=FiniteQuadraticModule([3])
            sage: S=A._Hom_(B)
            sage: a = A.gen(0)
            sage: b = B.gen(0)
            sage: f = S([b])
            sage: f.kernel()
            < 6*e >
        """
        kernel_gens = []
        for a in self.domain():
            if a == 0:
                continue
            if self(a) == 0:
                kernel_gens.append(a)
        if not kernel_gens:
            kernel_gens = [self.domain()(0)]
        return self.domain().subgroup(tuple(kernel_gens))

    def image(self, J=None):
        r"""
        Return the image of this morphism.

        INPUT:

        - ``J`` -- subset of the domain.

        ..NOTE:: This is currently implemtented using brute-force.

        ..TODO:: Implement this in a more efficient way.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: f.image()
            < e1 >
            sage: f.image([3*A.gen(0)])
            < 0 >
            sage: A = FiniteQuadraticModule([6])
            sage: B = FiniteQuadraticModule([3])
            sage: S = A._Hom_(B)
            sage: a = A.gen(0)
            sage: b = B.gen(0)
            sage: f = S([b])
            sage: f.image()
            < e >
            sage: f.image([2*a])
            < 2*e >
        """
        image_gens = []
        if J is None:
            J = self.domain()
        for a in J:
            image_gens.append(self(a))
        image_gens = tuple(set(image_gens))
        return self.codomain().subgroup(image_gens)

    def is_isomorphism(self):
        r"""
        Return True if this homomorphism is an isomorphism (i.e. both injective and surjective).

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: f.is_isomorphism()
            False

        """
        return self.kernel().is_trivial() and self.image().order() == self.codomain().order()

    def _repr_defn(self):
        """
        Used in constructing string representation of ``self``.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: print(f._repr_defn())
            e1 |--> e1
            e1 |--> e1
        """
        D = self.domain()
        ig = self._im_gens
        s = '\n'.join([f"{D.gen(i)} |--> {ig[i]}" for i in range(D.ngens())])
        return s

    def _repr_(self):
        r"""
        String representation of self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: a,b,c,d = B.gens();
            sage: f = S([a,a]); f
            Homomorphism : Finite quadratic module in 2 generators:
             gens: e1, e1
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 --> Finite quadratic module in 4 generators:
             gens: e1, e1, e3, e3
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2
            e1 |--> e1
            e1 |--> e1
        """
        return f"Homomorphism : {self.domain()} --> {self.codomain()} \n{self._repr_defn()}"

    def _latex_(self):
        r"""
        Return a LaTeX representation of this homomorphism.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: latex(f)
            \left(\left\langle \mathbb{Z}^{2}/\left(\begin{array}{rr}
            1 & 0 \\
            2 & 3
            \end{array}\right)\mathbb{Z}^{2} \right\rangle,\frac{1}{3} x_{0}^{2} + ...
            1 & 0 & 0 & 0 \\
            2 & 3 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 2 & 3
            \end{array}\right)\mathbb{Z}^{4} \right\rangle,\frac{1}{3} x_{0}^{2} + ...

        """
        return "%s \\rightarrow{} %s"%(latex(self.domain()), latex(self.codomain()))

    def __call__(self, x):
        r"""
        Call self.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=A._Hom_(B)
            sage: u = A.gen(0)
            sage: a,b,c,d = B.gens()
            sage: f = S([a,a])
            sage: f(u), f(2*u), f(A(0))
            (e1, 2*e1, 0)
            sage: f(1)
            Traceback (most recent call last):
            ...
            TypeError: x (=1) must be in Finite quadratic module in 2 generators...

        """
        if x not in self.domain():
            raise TypeError('x (={0}) must be in {1}'.format(x, self.domain()))
        n = len(self.__c_im_gens)
        cos = x.list(coords=self.domain()._default_coords)
        return sum(self.__c_im_gens[i] * cos[i] for i in range(n))


class FiniteQuadraticModuleHomset(HomsetWithBase):
    r"""
    Implements the set of morphisms of a quadratic module into another.

    """
    Element = FiniteQuadraticModuleHomomorphism_im_gens

    def __init__(self, A, B):
        r"""
        INPUT:

        - ``A`` -- finite quadratic module
        - ``B`` -- finite quadratic module

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: from fqm_weil.all import FiniteQuadraticModuleHomset
            sage: A = FiniteQuadraticModule([11,33])
            sage: FiniteQuadraticModuleHomset(A,A)
            Set of Morphisms from Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2 to Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2 in Category of modules over Integer Ring
            sage: F = FiniteQuadraticModule(Matrix(ZZ,[[10,5],[-5,-2]]),Matrix(QQ,[[2/5,0],[0,-2]]))
            sage: FiniteQuadraticModuleHomset(A,F)
             Set of Morphisms from Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2 to Finite quadratic module in 1 generator:
             gen: e
             form: 2/5*x^2 in Category of modules over Integer Ring
            sage: M = FiniteQuadraticModule('2^-2.2_1^1')
            sage: N = FiniteQuadraticModule('2_1^-3')
            sage: H = FiniteQuadraticModuleHomset(M,N); H
            Set of Morphisms from Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2 + 1/4*x2^2 to Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/4*x0^2 + 1/2*x1^2 + 1/2*x1*x2 + 1/2*x2^2 in Category of modules over Integer Ring
            sage: TestSuite(H).run()
        """
        if not all([isinstance(A, FiniteQuadraticModule_base),
                   isinstance(B, FiniteQuadraticModule_base)]):
            raise NotImplementedError
        HomsetWithBase.__init__(self, A, B, A.category(), base=A.base_ring())

    def an_element(self):
        r"""
        Return a homomorphism.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: from fqm_weil.all import FiniteQuadraticModuleHomset
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=FiniteQuadraticModuleHomset(A,B)
            sage: S.an_element()
            Homomorphism : Finite quadratic module in 2 generators:
             gens: e1, e1
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 --> Finite quadratic module in 4 generators:
             gens: e1, e1, e3, e3
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2
            e1 |--> 0
            e1 |--> 0
        """
        return self.zero()

    def zero(self):
        r"""
        Return the zero homomorphism.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: from fqm_weil.all import FiniteQuadraticModuleHomset
            sage: A = FiniteQuadraticModule(matrix(QQ, 2, [2,1,1,2]))
            sage: B = 2*A
            sage: S=FiniteQuadraticModuleHomset(A,B)
            sage: S.zero()
            Homomorphism : Finite quadratic module in 2 generators:
             gens: e1, e1
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 --> Finite quadratic module in 4 generators:
             gens: e1, e1, e3, e3
             form: 1/3*x0^2 + 2/3*x0*x1 + 1/3*x1^2 + 1/3*x2^2 + 2/3*x2*x3 + 1/3*x3^2
            e1 |--> 0
            e1 |--> 0
        """
        return self([0] * self.domain().ngens())

    def _element_constructor_(self, im_gens, check=True, **kwds):
        """
        Creat a homomorphism using the images under the generators.

        INPUT:

        - ``im_gens`` -- a list of $n$ elements of a quadratic module, where $n$ is the number
                 of generators of this quadratic module.
        - ``check`` -- boolean (default False). If true perform additional checks.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: from fqm_weil.all import FiniteQuadraticModuleHomset
            sage: A = FiniteQuadraticModule([11,33])
            sage: FiniteQuadraticModuleHomset(A,A)([A.0,A.1])
            Homomorphism : Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2 --> Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/44*x0^2 + 1/132*x1^2
            e0 |--> e0
            e1 |--> e1
        """
        try:
            return self.element_class(self, im_gens, check=check)
        except (NotImplementedError, ValueError) as err:
            raise TypeError(f"images (={im_gens}) do not define a valid homomorphism. ERR: {err}")

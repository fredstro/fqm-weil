r"""
Jordan decompositions of finite quadratic modules.

"""
import logging

from sage.all import ZZ
from sage.arith.functions import lcm
from sage.arith.misc import valuation, kronecker, is_prime, inverse_mod, is_prime_power, gcd
from sage.functions.other import floor, binomial
from sage.matrix.constructor import matrix
from sage.misc.flatten import flatten
from sage.misc.functional import is_odd, is_even
from sage.misc.misc_c import prod
from sage.rings.integer import Integer
from sage.rings.number_field.number_field import CyclotomicField
from sage.structure.sage_object import SageObject

CF8 = CyclotomicField(8)
z8 = CF8.gen()

class JordanComponent(SageObject):
    """
    A Jordan component of a finite quadratic module.

    EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1))
            3^2
            sage: A = FiniteQuadraticModule('3^1')
            sage: JordanComponent(A.gens(), (3, 1, 1, 1))
            3
            sage: A = FiniteQuadraticModule('2_1^1')
            sage: JordanComponent(A.gens(), (2, 1, 1, 1, 1))
            2_1
            sage: B = FiniteQuadraticModule('2^-2')
            sage: JordanComponent(B.gens(), (2, 1, 2, -1))
            2^-2
            sage: C = FiniteQuadraticModule('2^2')
            sage: JordanComponent(C.gens(), (2, 1, 2, 1))
            2^2
            sage: A = FiniteQuadraticModule('2_1^-5.3^-1')
            sage: JordanComponent(A.gens()[0:5], (2, 1, 5, 1, 5))
            2_5^5
            sage: JordanComponent(A.gens()[5:], (3, 1, 1, -1))
            3^-1
            sage: A = FiniteQuadraticModule('2^2.3^2')
            sage: JordanComponent(A.gens()[0:2], (3, 1, 2, 1))
            Traceback (most recent call last):
            ...
            ValueError: Invalid basis: '(e0, e1)' need order 3
            sage: JordanComponent(A.gens()[2:], (2, 1, 2, 1))
            Traceback (most recent call last):
            ...
            ValueError: Invalid basis: '(e2, e3)' need order 2


            TODO: Handle trivial jordan components.
    """

    def __init__(self, basis, invariants, check=True):
        """
        Initialize a Jordan component.

        INPUT:

        - `basis` -- tuple of generators
        - `invariants` -- tuple of invariants (p, k, n, eps, t) (t is optional)
        - `check` -- boolean (default: True)

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1))
            3^2

        """
        super(JordanComponent, self).__init__()
        if not isinstance(invariants, tuple) or not isinstance(basis, tuple):
            raise ValueError("Invariants and basis must be tuples")
        p, k, n, eps, *t = invariants
        logging.debug(f"{invariants}")
        logging.debug(basis)
        if not is_prime(p) or (p > 2 and t != []) or len(t) > 1:
            raise ValueError(f"Invalid invariant data: {invariants}")
        if len(basis) != n:
            raise ValueError(f"Invalid basis: '{basis}' or rank '{n}'")
        if any(b.order() != p**k for b in basis):
            raise ValueError(f"Invalid basis: '{basis}' need order {p**k}")
        self.p = p
        self.q = p**k
        self.k = k
        self.n = n
        self.eps = eps
        self.t = t[0] if t else None
        self._basis = basis
        self._ambient_module = self._basis[0].parent()
        self._invariants = invariants
        self._type_II = False
        self._type_I = False
        if p == 2 and self.t is not None:
            self._type_I = True
            if check and n == 1 and any(b.norm() * 2 ** (k + 1) % 2 ** (k + 1) not in [1, 3] for b in basis):
                raise ValueError(f"Incorrect basis for indecomposable type_I 2-adic component.: "
                                 f"{self.invariants()}, {self.basis()}")
        elif p == 2:
            self._type_II = True
            if check and len(basis) % 2 == 1:
                raise ValueError("Incorrect basis for type_II 2-adic component.")
            if check and n == 2 and self.eps == -1:  # type 'B'
                if any(b.norm() * (2 ** k) != 1 for b in basis):
                    raise ValueError("Incorrect basis for type_II 2-adic component.")
                if any(basis[i].dot(basis[i+1]) * 2 ** k != 1 for i in range(0, len(basis), 2)):
                    raise ValueError("Incorrect basis for type_II 2-adic component.")
            if check and n == 2 and self.eps == 1:
                if any(b.norm() != 0 for b in basis):
                    raise ValueError("Incorrect basis for type_II 2-adic component.")
                if any(basis[i].dot(basis[i+1]) * 2 ** k != 1 for i in range(0, len(basis), 2)):
                    raise ValueError("Incorrect basis for type_II 2-adic component.")
        if n == 1:
            self._indecomposable = True
        elif n == 2 and self._type_II:
            self._indecomposable = True
        else:
            self._indecomposable = False

    def __repr__(self):
        """
        The Jordan symbol of the Jordan component.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1))
            3^2
        """
        return self.genus_symbol()

    def genus_symbol(self):
        """
        The genus symbol of the Jordan component.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: J = JordanComponent(A.gens(), (3, 1, 2, 1))
            sage: J.genus_symbol()
            '3^2'
            sage: A = FiniteQuadraticModule('9^2')
            sage: J = JordanComponent(A.gens(), (3, 2, 2, 1))
            sage: J.genus_symbol()
            '9^2'
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: J = JordanComponent(A.gens()[0:2], (2, 1, 2, 1))
            sage: J.genus_symbol()
            '2^2'
            sage: J = JordanComponent(A.gens()[2:], (2, 2, 1, 1, 1))
            sage: J.genus_symbol()
            '4_1'


        """
        s = str(self.q)
        e = self.n * self.eps
        if self.t:
            s += '_' + str(self.t)
        if e != 1:
            s += '^' + str(e)
        return s

    def invariants(self):
        """
        Invariants of the Jordan component.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1)).invariants()
            (3, 1, 2, 1)
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: J = JordanComponent(A.gens()[0:2], (2, 1, 2, 1))
            sage: J.invariants()
            (2, 1, 2, 1)
            sage: J = JordanComponent(A.gens()[2:], (2, 2, 1, 1, 1))
            sage: J.invariants()
            (2, 2, 1, 1, 1)

        """
        return self._invariants

    def basis(self):
        """
        Basis of the Jordan component.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1)).basis()
            (e0, e1)
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: J = JordanComponent(A.gens()[0:2], (2, 1, 2, 1))
            sage: J.basis()
            (e0, e1)
            sage: J = JordanComponent(A.gens()[0:2], (2, 2, 1, 1, 1))
            Traceback (most recent call last):
            ...
            ValueError: Invalid basis: '(e0, e1)' or rank '1'
            sage: J = JordanComponent(A.gens()[2:], (2, 2, 1, 1, 1))
            sage: J.basis()
            (e2,)
        """
        return self._basis

    def is_type_I(self):
        """
        Return True if self is type I (also called odd 2-adic), else False.


        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1)).is_type_I()
            False
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: JordanComponent(A.gens()[0:2], (2, 1, 2, 1)).is_type_I()
            False
            sage: JordanComponent(A.gens()[2:], (2, 2, 1, 1, 1)).is_type_I()
            True
            sage: A = FiniteQuadraticModule('2^2')
            sage: JordanComponent(A.gens()[0:2], (2, 1, 2, 1)).is_type_I()
            False
            """
        return self._type_I

    def is_type_II(self):
        """
        Return True if self is type II (also called even 2-adic), else False.


        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1)).is_type_II()
            False
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: JordanComponent(A.gens()[0:2], (2, 1, 2, 1)).is_type_II()
            True
            sage: JordanComponent(A.gens()[2:], (2, 2, 1, 1, 1)).is_type_II()
            False
            sage: A = FiniteQuadraticModule('2^2')
            sage: JordanComponent(A.gens()[0:2], (2, 1, 2, 1)).is_type_II()
            True
            """
        return self._type_II

    def is_indecomposable(self):
        """
        Is the Jordan component indecomposable.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1)).is_indecomposable()
            False
            sage: A = FiniteQuadraticModule('3^1')
            sage: JordanComponent(A.gens(), (3, 1, 1, 1)).is_indecomposable()
            True
            sage: A = FiniteQuadraticModule('2_2^2')
            sage: JordanComponent(A.gens(), (2, 1, 2, 1, 2)).is_indecomposable()
            False
            sage: A = FiniteQuadraticModule('2^2')
            sage: JordanComponent(A.gens(), (2, 1, 2, 1)).is_indecomposable()
            True


        """
        return self._indecomposable

    def decompose(self):
        """
        Return a decomposition of self.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: JordanComponent(A.gens(), (3, 1, 2, 1)).decompose()
            [3, 3]
            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2_1^3')
            sage: JordanComponent(A.gens(), (2, 1, 3, -1, 5)).decompose()
            [2_1, 2_1, 2_5^-1]
            sage: A = FiniteQuadraticModule('2_1^-5')
            sage: JordanComponent(A.gens(), (2, 1, 5, 1, 5)).decompose()
            [2_1, 2_1, 2_1, 2_1, 2_1]

        """
        if self.is_indecomposable():
            return [self]
        if self.p > 2:
            b0 = self.basis()[0]
            return [JordanComponent((b0,), (self.p, self.k, 1, 1))] + \
                JordanComponent(self.basis()[1:], (self.p, self.k, self.n-1, self.eps)).decompose()
        if self._type_I:
            b0 = self.basis()[0]
            t_new = self.t - 1
            if is_odd(self.k) and self.eps == -1:
                t_new = (self.t + 4) % 8
            if t_new == 0:
                t_new = None
            return [JordanComponent((b0,), (self.p, self.k, 1, 1, 1))] + \
                JordanComponent(self.basis()[1:], (self.p, self.k, self.n - 1, self.eps,
                                                   t_new)).decompose()
        if self._type_II:
            b0, b1 = self.basis()[0:2]
            # Decide whether to decompose to a 'B' or a 'C':
            if b0.norm() == b1.norm() == 0 and b0.dot(b1) * self.q == 1:
                eps = 1
            elif self.q * b0.norm() == self.q * b1.norm() == 2 and b0.dot(b1) * self.q == 1:
                eps = -1
            else:
                raise ArithmeticError("Component of type II not correct.")
            return [JordanComponent((b0, b1), (self.p, self.k, 2, eps))] + \
                JordanComponent(self.basis()[2:], (self.p, self.k, self.n - 2, self.eps * eps,
                                                   )).decompose()

    def as_finite_quadratic_module(self):
        """
        Return this component as an ambient finite quadratic module.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: A.jordan_decomposition()[0].as_finite_quadratic_module()
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/2*x0*x1
            sage: A.jordan_decomposition()[1].as_finite_quadratic_module()
            Finite quadratic module in 1 generator:
             gen: e
             form: 1/8*x^2
        """
        return self.basis()[0].parent().spawn(self.basis())[0]

    def ambient_finite_quadratic_module(self):
        """
        Return the ambient finite quadratic module.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: A.jordan_decomposition()[0].as_finite_quadratic_module()
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/2*x0*x1
            sage: A.jordan_decomposition()[1].as_finite_quadratic_module()
            Finite quadratic module in 1 generator:
             gen: e
             form: 1/8*x^2
        """
        return self._ambient_module

    def _finite_quadratic_module_hom(self):
        """
        Return this component as an ambient finite quadratic module.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: A.jordan_decomposition()[0].as_finite_quadratic_module()
            Finite quadratic module in 2 generators:
             gens: e0, e1
             form: 1/2*x0*x1
            sage: A.jordan_decomposition()[1].as_finite_quadratic_module()
            Finite quadratic module in 1 generator:
             gen: e
             form: 1/8*x^2
        """
        return self.basis()[0].parent().spawn(self.basis())[1]

    def gauss_sum(self, c):
        """
        Calculate the Gauss sum G(c, x, D) of one Jordan component.
        $$G(c, x, D) = (|D||D^c|^{-1/2} \sum_{x\in D} \exp(2\pi i (c Q(x) + B(x, y))).$$

        INPUT:
            - `c` -- integer
            - `p0` -- optional prime

        .. NOTE::
            We apply the formulas in [Str, Sections 3] for each indecomposable Jordan component.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2_1')
            sage: A.jordan_decomposition()[0].gauss_sum(1)
            zeta8
            sage: A.Gauss_sum(1, A.gens()[0])
            -zeta8^3
            sage: A = FiniteQuadraticModule('3')
            sage: A.jordan_decomposition()[0].gauss_sum(1)
            -zeta8^2
            sage: A.Gauss_sum(1, A.gens()[0])
            zeta24^2
            sage: FiniteQuadraticModule('5').jordan_decomposition()[0].gauss_sum(1)
            -1
            sage: FiniteQuadraticModule('4_1').jordan_decomposition()[0].gauss_sum(1)
            zeta8
            sage: FiniteQuadraticModule('4^2').jordan_decomposition()[0].gauss_sum(1)
            1
            sage: FiniteQuadraticModule('4^-2').jordan_decomposition()[0].gauss_sum(1)
            1
            sage: FiniteQuadraticModule('2^-2').jordan_decomposition()[0].gauss_sum(1)
            -1
            sage: FiniteQuadraticModule('4^2.2_1').jordan_decomposition()[0].gauss_sum(1)
            zeta8
            sage: FiniteQuadraticModule('4^2.2_1').jordan_decomposition()[1].gauss_sum(1)
            1
            sage: FiniteQuadraticModule('4^-2.2_1').jordan_decomposition()[0].gauss_sum(1)
            zeta8
            sage: FiniteQuadraticModule('4^-2.2_1').jordan_decomposition()[1].gauss_sum(1)
            1
            sage: JC = FiniteQuadraticModule('4^-2.2_1').jordan_decomposition()[0]
            sage: JC.as_finite_quadratic_module().Gauss_sum(2, check=True)
            0
            sage: JC = FiniteQuadraticModule('4^-2.2_1').jordan_decomposition()[1]
            sage: JC.as_finite_quadratic_module().Gauss_sum(2, check=True)
            -1

        """
        if not self.is_indecomposable():
            return prod(comp.gauss_sum(c) for comp in self.decompose())
        qc = gcd(c, self.q)
        if self.p > 2:
            arg = self.n - self.n * self.q // qc
            return self.eps ** self.k * kronecker(c // qc, self.q // qc) * z8 ** arg
        if self._type_I:
            vcp = valuation(c, 2)
            if vcp == self.k:
                return 0
            if vcp < self.k:
                arg = self.t * c // 2 ** vcp
            else:
                arg = 0
            return self.eps ** self.k * kronecker(c // qc, self.q // qc) * z8 ** arg
        elif self.eps == -1:
            return kronecker(3, self.q // qc)
        else:
            return 1


class JordanDecomposition(SageObject):
    r"""
    A container class for the Jordan constituents of a
    finite quadratic module.

    EXAMPLES:

        sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
        sage: A = FiniteQuadraticModule('2^2.4_1^1')
        sage: A.jordan_decomposition()
        Jordan decomposition with genus symbol '2^2.4_1'


    """
    def __init__(self, A):
        r"""
        Initialize a jordan decomposition.

        INPUT:

            A -- a nondegenerate finite quadratic module

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: A.jordan_decomposition()
            Jordan decomposition with genus symbol '2^2.4_1'
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition()
            Jordan decomposition with genus symbol '3^2.5^-1.7^4'
            sage: E = FiniteQuadraticModule('4^-2.5^2')^2
            sage: E.jordan_decomposition()
            Traceback (most recent call last):
            ...
            ValueError: Not a nondegenerate module.

        TODO: The case of degenerate modules.

        """
        self.__A = A
        if not A.is_nondegenerate():
            raise ValueError("Not a nondegenerate module.")
        U = A.subgroup(A.gens())
        og_b = U.orthogonal_basis()
        jd = dict()
        ol = []
        primary_comps = sorted(set([x.order() for x in og_b]))
        for q in primary_comps:
            basis = tuple([x for x in og_b if x.order() == q])
            p = q.factor()[0][0]
            n = valuation(q, p)
            r = len(basis)

            def f(x, y):
                return x.norm() * 2 * q if x == y else x.dot(y) * q
            F = matrix(ZZ, r, r, [f(x, y) for x in basis for y in basis])
            # print("Det=",F.det())
            eps = kronecker(F.det(), p)
            genus = [p, n, r, eps]
            if 2 == p and self.is_type_I(F):
                t = sum([x for x in F.diagonal() if is_odd(x)]) % 8
                genus.append(t)
            jd[q] = (basis, tuple(genus))
            ol.append((p, n))
        self.__jd = jd
        self.__ol = ol
        self._jordan_components = {
            q: JordanComponent(x[0], x[1]) for q, x in self.__jd.items()
        }
        super(JordanDecomposition, self).__init__()

    def _repr_(self):
        r"""
        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: A.jordan_decomposition() # indirect doctest
            Jordan decomposition with genus symbol '2^2.4_1'
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition() # indirect doctest
            Jordan decomposition with genus symbol '3^2.5^-1.7^4'

        """
        return f"Jordan decomposition with genus symbol '{self.genus_symbol()}'"

    def _latex_(self):
        r"""
        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: latex(A.jordan_decomposition()) # indirect doctest
            Jordan decomposition with genus symbol '$2^2.4_1$'
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: latex(A.jordan_decomposition()) # indirect doctest
            Jordan decomposition with genus symbol '$3^2.5^-1.7^4$'

        """
        return f"Jordan decomposition with genus symbol '${self.genus_symbol()}$'"

    def __iter__(self):
        r"""
        Return the Jordan decomposition as iterator of JordanComponent objects.

        The returned components in the list are given
        (basis, (prime p,  valuation of p-power n, dimension r, determinant e over p[, oddity o]),
            where $n > 0$, ordered lexicographically by $p$, $n$.



        EXAMPLES:

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: list(A.jordan_decomposition()) # indirect doctest
            [2^2, 4_1]
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: list(A.jordan_decomposition()) # indirect doctest
            [3^2, 5^-1, 7^4]

        """
        return (self._jordan_components[p ** n] for p, n in self.__ol)

    def __getitem__(self, item):
        """
        Get one or more components.

        INPUT:

        - item -- an integer or slice

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: A.jordan_decomposition()[0] # indirect doctest
            2^2
            sage: A.jordan_decomposition()[1] # indirect doctest
            4_1
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition()[0] # indirect doctest
            3^2
            sage: A.jordan_decomposition()[1:3] # indirect doctest
            [5^-1, 7^4]

        """
        p_and_n = self.__ol[item]
        if isinstance(p_and_n, tuple):
            p, n = p_and_n
            return self._jordan_components[p**n]
        return [self._jordan_components[p**n] for (p, n) in p_and_n]

    def __len__(self):
        """
        Length of self.

        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1^1')
            sage: len(A.jordan_decomposition())
            2
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: len(A.jordan_decomposition())
            3

        """
        return len(self.__ol)

    def genus_symbol(self, p=None):
        r"""
        Return the genus symbol of the Jordan constituents
        whose exponent is a power of the prime $p$.
        Return the concatenation of all local genus symbols
        if no argument is given.

        INPUT:

        - ``p`` -- prime (default: ``None``)

        EXAMPLES::


            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: A.jordan_decomposition().genus_symbol()
            '2^2.4_1'
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition().genus_symbol()
            '3^2.5^-1.7^4'
            sage: M = FiniteQuadraticModule('2^-2.2_1^1'); M
            Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2 + 1/4*x2^2
            sage: M.jordan_decomposition().genus_symbol()
            '2_1^-3'
            sage: N = FiniteQuadraticModule('2_1^-3')
            sage: N.is_isomorphic(M)
            True
            sage: N = FiniteQuadraticModule('2_5^-3')
            sage: N.is_isomorphic(M)
            False
        """
        n = self.__A.order()
        if not p:
            _P = n.prime_divisors()
            _P.sort(reverse=True)
        elif is_prime(p):
            _P = [p] if p.divides(n) else []
        else:
            raise TypeError
        s = ''
        while [] != _P:
            p = _P.pop()
            s += self._genus_symbol(p)
            if [] != _P:
                s += '.'
        return s

    def _genus_symbol(self, p):
        r"""
        Return the genus symbol of the Jordan constituent
        whose exponent is a power of the prime $p$.
        Do not use directly, use genus_symbol() instead.

        INPUT:

        - ``p`` -- prime

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: A.jordan_decomposition()._genus_symbol(2)
            '2^2.4_1'
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition()._genus_symbol(3)
            '3^2'
            sage: A.jordan_decomposition()._genus_symbol(5)
            '5^-1'
            sage: A.jordan_decomposition()._genus_symbol(7)
            '7^4'
            sage: M = FiniteQuadraticModule('2^-2.2_1^1'); M
            Finite quadratic module in 3 generators:
             gens: e0, e1, e2
             form: 1/2*x0^2 + 1/2*x0*x1 + 1/2*x1^2 + 1/4*x2^2
            sage: M.jordan_decomposition()._genus_symbol(2)
            '2_1^-3'
            sage: M.jordan_decomposition()._genus_symbol(3)
            ''

        """
        # l = [q for q in self.__jd.keys() if q % p == 0]
        # l.sort(reverse=True)
        return '.'.join([self._jordan_components[p**k].genus_symbol() for (p0, k) in self.__ol if p == p0])

    def orbit_list(self, p=None, short=False):
        r"""
        If this is the Jordan decomposition for $(M,Q)$, return the dictionary of
        dictionaries of orbits corresponding to the p-groups of $M$.
        If a prime p is given only the dictionary of orbits for the p-group is returned.

        INPUT:

        - ``p`` -- prime (default: ``None``)
        - ``short`` -- boolean (default: False)

        OUTPUT:
            dictionary -- the mapping p --> (dictionary -- the mapping orbit --> the number of elements in this orbit)

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, JordanDecomposition
            sage: A = FiniteQuadraticModule('3^-3.25^2')
            sage: J = JordanDecomposition(A)
            sage: J.orbit_list() == \
                  {3: \
                      {(1,): 1, \
                       (3, 1, 0): 8, \
                       (3, 1, 1/3): 6, \
                       (3, 1, 2/3): 12}, \
                   5: {(1,): 1, \
                       (5, 5, 0): 8, \
                       (5, 5, 1/5): 4, \
                       (5, 5, 2/5): 4, \
                       (5, 5, 3/5): 4, \
                       (5, 5, 4/5): 4, \
                       (25, 1, 1, 0, 0): 40, \
                       (25, 1, 1, 1/25, 1/5): 20, \
                       (25, 1, 1, 2/25, 2/5): 20, \
                       (25, 1, 1, 3/25, 3/5): 20, \
                       (25, 1, 1, 4/25, 4/5): 20, \
                       (25, 1, 1, 1/5, 0): 40, \
                       (25, 1, 1, 6/25, 1/5): 20, \
                       (25, 1, 1, 7/25, 2/5): 20, \
                       (25, 1, 1, 8/25, 3/5): 20, \
                       (25, 1, 1, 9/25, 4/5): 20, \
                       (25, 1, 1, 2/5, 0): 40, \
                       (25, 1, 1, 11/25, 1/5): 20, \
                       (25, 1, 1, 12/25, 2/5): 20, \
                       (25, 1, 1, 13/25, 3/5): 20, \
                       (25, 1, 1, 14/25, 4/5): 20, \
                       (25, 1, 1, 3/5, 0): 40, \
                       (25, 1, 1, 16/25, 1/5): 20, \
                       (25, 1, 1, 17/25, 2/5): 20, \
                       (25, 1, 1, 18/25, 3/5): 20, \
                       (25, 1, 1, 19/25, 4/5): 20, \
                       (25, 1, 1, 4/5, 0): 40, \
                       (25, 1, 1, 21/25, 1/5): 20, \
                       (25, 1, 1, 22/25, 2/5): 20, \
                       (25, 1, 1, 23/25, 3/5): 20, \
                       (25, 1, 1, 24/25, 4/5): 20}}
            True
            sage: A = FiniteQuadraticModule('3^-3.27^2')
            sage: J = JordanDecomposition(A)
            sage: J.orbit_list(3) == \
                  {(1,): 1, \
                  (3, 1, 0): 72, \
                  (3, 1, 1/3): 54, \
                  (3, 1, 2/3): 108, \
                  (3, 9, 1/3): 4, \
                  (3, 9, 2/3): 4, \
                  (9, 1, 3, 0, 1/3): 432, \
                  (9, 1, 3, 0, 2/3): 216, \
                  (9, 1, 3, 1/3, 1/3): 288, \
                  (9, 1, 3, 1/3, 2/3): 432, \
                  (9, 1, 3, 2/3, 1/3): 216, \
                  (9, 1, 3, 2/3, 2/3): 288, \
                  (9, 3, 3, 1/9, 1/3): 12, \
                  (9, 3, 3, 2/9, 2/3): 12, \
                  (9, 3, 3, 4/9, 1/3): 12, \
                  (9, 3, 3, 5/9, 2/3): 12, \
                  (9, 3, 3, 7/9, 1/3): 12, \
                  (9, 3, 3, 8/9, 2/3): 12, \
                  (27, 1, 1, 1, 1/27, 1/9, 1/3): 972, \
                  (27, 1, 1, 1, 2/27, 2/9, 2/3): 972, \
                  (27, 1, 1, 1, 4/27, 4/9, 1/3): 972, \
                  (27, 1, 1, 1, 5/27, 5/9, 2/3): 972, \
                  (27, 1, 1, 1, 7/27, 7/9, 1/3): 972, \
                  (27, 1, 1, 1, 8/27, 8/9, 2/3): 972, \
                  (27, 1, 1, 1, 10/27, 1/9, 1/3): 972, \
                  (27, 1, 1, 1, 11/27, 2/9, 2/3): 972, \
                  (27, 1, 1, 1, 13/27, 4/9, 1/3): 972, \
                  (27, 1, 1, 1, 14/27, 5/9, 2/3): 972, \
                  (27, 1, 1, 1, 16/27, 7/9, 1/3): 972, \
                  (27, 1, 1, 1, 17/27, 8/9, 2/3): 972, \
                  (27, 1, 1, 1, 19/27, 1/9, 1/3): 972, \
                  (27, 1, 1, 1, 20/27, 2/9, 2/3): 972, \
                  (27, 1, 1, 1, 22/27, 4/9, 1/3): 972, \
                  (27, 1, 1, 1, 23/27, 5/9, 2/3): 972, \
                  (27, 1, 1, 1, 25/27, 7/9, 1/3): 972, \
                  (27, 1, 1, 1, 26/27, 8/9, 2/3): 972}
            True
        """
        n = self.__A.order()
        if not p:
            _P = n.prime_divisors()
            if 2 in _P:
                _P.remove(2)
            _P.sort(reverse=True)
        elif is_prime(p):
            return self._orbit_list(p, short=short)
        else:
            raise TypeError
        orbit_dict = dict()
        while [] != _P:
            p = _P.pop()
            orbit_dict[p] = self._orbit_list(p, short=short)
        return orbit_dict

    def _orbit_list(self, p, short=False, debug=0):
        r"""
        If this is the Jordan decomposition for $(M,Q)$, return the dictionary of
        orbits corresponding to the p-group of $M$.

        INPUT:

        - ``p`` -- prime
        - ``short`` -- boolean (default: False)

        OUTPUT:
            dictionary -- the mapping orbit --> the number of elements in this orbit

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, JordanDecomposition
            sage: A = FiniteQuadraticModule('3^-3.27^2')
            sage: J = JordanDecomposition(A)
            sage: J._orbit_list(3) == \
                  {(1,): 1, \
                  (3, 1, 0): 72, \
                  (3, 1, 1/3): 54, \
                  (3, 1, 2/3): 108, \
                  (3, 9, 1/3): 4, \
                  (3, 9, 2/3): 4, \
                  (9, 1, 3, 0, 1/3): 432, \
                  (9, 1, 3, 0, 2/3): 216, \
                  (9, 1, 3, 1/3, 1/3): 288, \
                  (9, 1, 3, 1/3, 2/3): 432, \
                  (9, 1, 3, 2/3, 1/3): 216, \
                  (9, 1, 3, 2/3, 2/3): 288, \
                  (9, 3, 3, 1/9, 1/3): 12, \
                  (9, 3, 3, 2/9, 2/3): 12, \
                  (9, 3, 3, 4/9, 1/3): 12, \
                  (9, 3, 3, 5/9, 2/3): 12, \
                  (9, 3, 3, 7/9, 1/3): 12, \
                  (9, 3, 3, 8/9, 2/3): 12, \
                  (27, 1, 1, 1, 1/27, 1/9, 1/3): 972, \
                  (27, 1, 1, 1, 2/27, 2/9, 2/3): 972, \
                  (27, 1, 1, 1, 4/27, 4/9, 1/3): 972, \
                  (27, 1, 1, 1, 5/27, 5/9, 2/3): 972, \
                  (27, 1, 1, 1, 7/27, 7/9, 1/3): 972, \
                  (27, 1, 1, 1, 8/27, 8/9, 2/3): 972, \
                  (27, 1, 1, 1, 10/27, 1/9, 1/3): 972, \
                  (27, 1, 1, 1, 11/27, 2/9, 2/3): 972, \
                  (27, 1, 1, 1, 13/27, 4/9, 1/3): 972, \
                  (27, 1, 1, 1, 14/27, 5/9, 2/3): 972, \
                  (27, 1, 1, 1, 16/27, 7/9, 1/3): 972, \
                  (27, 1, 1, 1, 17/27, 8/9, 2/3): 972, \
                  (27, 1, 1, 1, 19/27, 1/9, 1/3): 972, \
                  (27, 1, 1, 1, 20/27, 2/9, 2/3): 972, \
                  (27, 1, 1, 1, 22/27, 4/9, 1/3): 972, \
                  (27, 1, 1, 1, 23/27, 5/9, 2/3): 972, \
                  (27, 1, 1, 1, 25/27, 7/9, 1/3): 972, \
                  (27, 1, 1, 1, 26/27, 8/9, 2/3): 972}
            True
        """
        if not is_prime(p):
            raise TypeError
        ppowers = [q for q in self.__jd.keys() if 0 == q % p]
        ppowers.sort()
        if debug > 0:
            print("ppowers=", ppowers)
        if not ppowers:
            return dict()
        orbitdict = {(1,): 1}

        levelpower = ppowers[len(ppowers) - 1]

        def _orbit_length(r, eps, t):

            if is_even(r):
                n = Integer(r / 2)
                if t.is_integral():
                    return p ** (r - 1) + eps * kronecker(-1, p) ** n * (p ** n - p ** (n - 1)) - 1
                else:
                    return p ** (r - 1) - eps * kronecker(-1, p) ** n * p ** (n - 1)
            else:
                if t.is_integral():
                    return p ** (r - 1) - 1
                else:
                    n = Integer((r - 1) / 2)
                    return p ** (r - 1) + eps * kronecker(-1, p) ** n * kronecker(2, p) * \
                           kronecker(Integer(p * t), p) * p ** n

        def _multiplicitieslist():
            r"""
            Create a dictionary of possible combinations of
            reduced multiplicities
            """

            completemultiplicitieslist = dict()

            for k in range(0, valuation(levelpower, p)):

                m = p ** (k + 1)
                idash = len([x for x in ppowers if x < m])
                completemultiplicitieslist[k] = []
                usedrankslist = []

                for j in range(idash, len(ppowers)):
                    usedranks = [0 for x in ppowers]
                    usedranks[j] = 1
                    completemultiplicitieslist[k].append([Integer(ppowers[j] / m)])
                    usedrankslist.append(usedranks)

                for j in range(k - 1, -1, -1):

                    for x in range(0, len(completemultiplicitieslist[k])):

                        multiplicities = completemultiplicitieslist[k][x]
                        usedranks = usedrankslist[x]
                        idash = len([xx for xx in ppowers if xx <= p ** j])

                        completemultiplicitieslist[k][x] = [multiplicities[0]] + multiplicities

                        for jj in [xx for xx in range(idash, len(ppowers)) if
                                   usedranks[xx] < self.__jd[ppowers[xx]][1][2] and ppowers[xx]
                                   < p ** (j + 1) * multiplicities[0]]:
                            newusedranks = usedranks[:]
                            newusedranks[jj] += 1
                            newmultiplicities = [Integer(
                                ppowers[jj] / p ** (j + 1))] + multiplicities
                            completemultiplicitieslist[k].append(newmultiplicities)
                            usedrankslist.append(newusedranks)

            multiplicitieslist = []
            for k in completemultiplicitieslist.keys():
                multiplicitieslist += sorted(completemultiplicitieslist[k])

            return multiplicitieslist

        multiplicitieslist = _multiplicitieslist()

        multiplicitieslist.reverse()

        tconstant = Integer(2)
        while kronecker(tconstant, p) != -1 and tconstant < p:
            tconstant += 1

        ranks = [self.__jd[x][1][2] for x in ppowers]
        epsilons = [self.__jd[x][1][3] for x in ppowers]

        while multiplicitieslist != []:

            if debug > 0:
                print("multiplicitieslist=", multiplicitieslist)

            multiplicities = multiplicitieslist.pop()
            k = len(multiplicities) - 1
            pk = p ** k
            m = p * pk

            if debug > 0:
                print("pk={0}, m={1}, k={2}".format(pk, m, k))

            if multiplicities[0] == multiplicities[k]:

                ordersDv1 = [Integer(x / multiplicities[0]) for x in ppowers if
                             x > multiplicities[0]]
                ranksDv1 = ranks[len(ppowers) - len(ordersDv1):]
                ordersDv1pk = [Integer(x / pk) for x in ordersDv1 if x > pk]
                ranksDv1pk = ranksDv1[len(ordersDv1) - len(ordersDv1pk):]
                if debug > 0:
                    print("ordersDv1 = {0}, ranksDv1={1}".format(ordersDv1, ranksDv1))
                if debug > 0:
                    print("ordersDv1pk = {0}, ranksDv1pk={1}".format(ordersDv1pk, ranksDv1pk))

                if len(ordersDv1pk) != 0 and ordersDv1pk[0] == p:

                    constantfactor = Integer(prod([min(pk, ordersDv1[j]) ** ranksDv1[j] for j in
                                                   range(0, len(ordersDv1))]) / pk)
                    if debug > 0:
                        print("constantfactor={0}".format(constantfactor))
                    constantfactor *= prod([p ** x for x in ranksDv1pk[1:]])
                    rank = ranksDv1pk[0]
                    eps = epsilons[len(ranks) - len(ranksDv1pk)]
                    values = [constantfactor * _orbit_length(rank, eps, tconstant / p)]
                    values += [constantfactor * _orbit_length(rank, eps, Integer(0))]
                    values += [constantfactor * _orbit_length(rank, eps, Integer(1) / p)]

                    if short:

                        orbitdict[tuple([m] + multiplicities)] = tuple(values)

                    else:

                        nonzeros = [t for t in range(0, p) if values[kronecker(t, p) + 1] != 0]

                        for tdash in range(0, m, p):

                            for tdashdash in nonzeros:
                                tt = tdash + tdashdash
                                orbitlength = values[kronecker(tdashdash, p) + 1]
                                orbit = tuple([m] + multiplicities + [
                                    x - floor(x) for x in
                                    [tt * p ** j / m for j in range(0, k + 1)]
                                ])
                                orbitdict[orbit] = orbitlength

            else:

                maxdenominators = [p for x in multiplicities]
                for j in range(k - 1, -1, -1):
                    maxdenominators[j] = Integer(
                        max(p * multiplicities[j] / multiplicities[j + 1] *
                            maxdenominators[j + 1], p))

                skips = [0] + [j + 1 for j in range(0, k) if
                               multiplicities[j + 1] > multiplicities[j]]
                noskips = [j for j in range(1, k + 1) if j not in skips]
                ranklist = []
                epslist = []
                constantfactor = p ** (len(skips) - 1 - k)
                ordersD = [Integer(x / multiplicities[0])
                           for x in ppowers if x > multiplicities[0]]
                ranksD = ranks[len(ppowers) - len(ordersD):]

                for skip in skips[1:]:

                    quotient = Integer(multiplicities[skip] / multiplicities[skip - 1])
                    ordersA = [x for x in ordersD if x <= quotient * p ** skip]
                    ranksA = ranksD[:len(ordersA)]
                    ordersB = ordersD[len(ranksA):]
                    ranksB = ranksD[len(ranksA):]
                    pj = p ** (skip - 1)
                    constantfactor *= prod(
                        [min(pj, ordersA[j]) ** ranksA[j] for j in range(0, len(ranksA))])
                    ordersApj = [Integer(x / pj) for x in ordersA if x > pj]
                    ranksApj = ranksA[len(ranksA) - len(ordersApj):]

                    if ordersApj != [] and ordersApj[0] == p:

                        ranklist.append(ranksApj[0])
                        epslist.append(epsilons[len(epsilons) - len(ranksD)])
                        constantfactor *= p ** sum(ranksApj[1:])
                        ordersD = [Integer(x / quotient) for x in ordersB if x > quotient]
                        ranksD = ranksB[len(ranksB) - len(ordersD):]

                    else:

                        constantfactor = 0
                        break

                if constantfactor:

                    constantfactor *= prod(
                        [min(pk, ordersD[j]) ** ranksD[j] for j in range(0, len(ordersD))])
                    ordersDpk = [Integer(x / pk) for x in ordersD if x > pk]
                    ranksDpk = ranksD[len(ranksD) - len(ordersDpk):]

                    if ordersDpk != [] and ordersDpk[0] == p:

                        ranklist.append(ranksDpk[0])
                        epslist.append(epsilons[len(epsilons) - len(ranksDpk)])
                        constantfactor *= p ** sum(ranksDpk[1:])

                    else:

                        constantfactor = 0

                if constantfactor:

                    product = [constantfactor] + [0 for j in skips[2:]]
                    skipindex = 0
                    tpjvalues = [0 for j in skips]
                    tpjs = [[x / maxdenominators[0] for x in range(0, maxdenominators[0])]] + [[]
                                                                                               for
                                                                                               j in
                                                                                               skips[
                                                                                               1:]]

                    # if debug > 0: print "tpjs", tpjs

                    while tpjs[0] != []:

                        if tpjs[skipindex] == []:

                            skipindex -= 1
                            tpjs[skipindex].remove(tpjvalues[skipindex])

                        else:

                            if skipindex == len(skips) - 1:

                                for tpj in tpjs[skipindex]:

                                    tpjvalues[skipindex] = tpj
                                    tpk = p ** (k - skips[skipindex]) * tpj
                                    factor = product[len(product) - 1]
                                    t = p ** (skips[skipindex] - skips[skipindex - 1] - 1) * \
                                        tpjvalues[skipindex - 1]
                                    t -= multiplicities[skips[skipindex]] / \
                                         multiplicities[skips[skipindex] - 1] / p * tpj
                                    orbitlength1 = _orbit_length(ranklist[skipindex - 1],
                                                                 epslist[skipindex - 1], t)
                                    orbitlength2 = _orbit_length(ranklist[skipindex],
                                                                 epslist[skipindex], tpk)
                                    orbitlengths = orbitlength1 * orbitlength2

                                    if orbitlengths != 0:

                                        reducednorms = [0 for j in range(0, k + 1)]
                                        for j in range(0, len(skips)):
                                            reducednorms[skips[j]] = tpjvalues[j]
                                        for j in noskips:
                                            t = p * reducednorms[j - 1]
                                            reducednorms[j] = t - floor(t)

                                        orbitdict[tuple([m] + multiplicities + reducednorms)] = \
                                            factor * orbitlengths

                                skipindex -= 1
                                tpjs[skipindex].remove(tpjvalues[skipindex])

                            else:

                                tpjvalues[skipindex] = tpjs[skipindex][0]

                                if skipindex > 0:

                                    t = p ** (skips[skipindex] - skips[skipindex - 1] - 1) * \
                                        tpjvalues[skipindex - 1]
                                    t -= multiplicities[skips[skipindex]] / \
                                         multiplicities[skips[skipindex] - 1] / \
                                         p * tpjvalues[skipindex]

                                    orbitlength = _orbit_length(ranklist[skipindex - 1],
                                                                epslist[skipindex - 1], t)

                                    if orbitlength == 0:

                                        tpjs[skipindex].remove(tpjvalues[skipindex])

                                    else:
                                        maxdenominator = maxdenominators[skips[skipindex + 1]]
                                        tcandidates = [t / maxdenominator
                                                       for t in range(0, maxdenominator)]
                                        difference = skips[skipindex + 1] - skips[skipindex]
                                        quotient = multiplicities[skips[skipindex + 1]] / \
                                                   multiplicities[skips[skipindex]]
                                        tpjs[skipindex + 1] = [t for t in tcandidates
                                                               if (quotient * t - p ** difference *
                                                                   tpjvalues[skipindex]).
                                                               is_integral()]
                                        product[skipindex] = orbitlength * product[skipindex - 1]
                                        skipindex += 1

                                else:

                                    maxdenominator = maxdenominators[skips[skipindex + 1]]
                                    tcandidates = [t / maxdenominator
                                                   for t in range(0, maxdenominator)]
                                    difference = skips[skipindex + 1] - skips[skipindex]
                                    quotient = multiplicities[skips[skipindex + 1]] / \
                                               multiplicities[skips[skipindex]]
                                    tpjs[skipindex + 1] = [t for t in tcandidates if
                                                           (quotient * t - p ** difference *
                                                            tpjvalues[skipindex]).is_integral()]

                                    skipindex += 1

        return orbitdict

    def values(self, debug=0):
        r"""
        If this is the Jordan decomposition for $(M,Q)$, return the values of $Q(x)$ ($x \in M$)
        as a dictionary d.

        OUTPUT:
            dictionary -- the mapping Q(x) --> the number of elements x with the same value Q(x)

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, JordanDecomposition
            sage: A = FiniteQuadraticModule('3^-3.27^2')
            sage: J = JordanDecomposition(A)
            sage: J.values() == \
                  {0: 729, \
                  1/27: 972, \
                  2/27: 972, \
                  4/27: 972, \
                  5/27: 972, \
                  7/27: 972, \
                  8/27: 972, \
                  1/3: 810, \
                  10/27: 972, \
                  11/27: 972, \
                  13/27: 972, \
                  14/27: 972, \
                  16/27: 972, \
                  17/27: 972, \
                  2/3: 648, \
                  19/27: 972, \
                  20/27: 972, \
                  22/27: 972, \
                  23/27: 972, \
                  25/27: 972, \
                  26/27: 972}
            True
            sage: A = FiniteQuadraticModule('2_3^-3.4^2.8_2^-2')
            sage: J = JordanDecomposition(A)
            sage: J.values() == \
                  {0: 480, \
                  1/8: 512, \
                  3/16: 1024, \
                  1/4: 480, \
                  3/8: 512, \
                  7/16: 1024, \
                  1/2: 544, \
                  5/8: 512, \
                  11/16: 1024, \
                  3/4: 544, \
                  7/8: 512, \
                  15/16: 1024}
            True
        """
        n = self.__A.order()

        values = [1]

        def combine_lists(list1, list2):
            N1 = len(list1)
            N2 = len(list2)
            newlength = lcm(N1, N2)
            newlist = [0 for j1 in range(0, newlength)]
            for j1 in range(0, N1):
                n1 = list1[j1]
                if n1 != 0:
                    for j2 in range(0, N2):
                        n2 = list2[j2]
                        if n2 != 0:
                            newlist[(j1 * Integer(newlength / N1) +
                                     j2 * Integer(newlength / N2)) % newlength] += (n1 * n2)
            return newlist

        def values_even2adic(gs):
            p, l, n, eps = gs
            n /= 2
            factor = 2 ** ((l - 1) * n)
            if n == 1 and eps == 1:
                return [factor * (l + 2)] + [factor * (valuation(j, 2) + 1) for j in
                                             range(1, 2 ** l)]
            else:
                quotient = Integer(2 ** n - eps) / Integer(2 ** (n - 1) - eps)
                return [factor * (quotient * (2 ** ((n - 1) * (l + 1)) - eps ** (l + 1)) + eps ** (
                            l + 1))] + \
                       [factor * quotient * (2 ** ((n - 1) * (l + 1)) -
                                             eps ** (valuation(j, 2) + 1) * 2 ** (
                                                         (n - 1) * (l - valuation(j, 2))))
                        for j in range(1, 2 ** l)]

        def values_odd2adic(gs):
            p, l, n, eps, t = gs
            # t = t%8
            tvalues = None
            if eps == +1:
                if t == 0:
                    tvalues = (1, 7)
                elif t == 1:
                    tvalues = (1,)
                elif t == 2:
                    tvalues = (1, 1)
                elif t == 3:
                    tvalues = (1, 1, 1)
                elif t == 4:
                    tvalues = (1, 1, 1, 1)
                elif t == 5:
                    tvalues = (7, 7, 7)
                elif t == 6:
                    tvalues = (7, 7)
                elif t == 7:
                    tvalues = (7,)
                else:
                    raise TypeError
            elif eps == -1:
                if t == 0:
                    tvalues = (5, 1, 1, 1)
                elif t == 1:
                    tvalues = (3, 7, 7)
                elif t == 2:
                    tvalues = (3, 7)
                elif t == 3:
                    tvalues = (3,)
                elif t == 4:
                    tvalues = (3, 1)
                elif t == 5:
                    tvalues = (5,)
                elif t == 6:
                    tvalues = (5, 1)
                elif t == 7:
                    tvalues = (5, 1, 1)
            else:
                raise TypeError

            level = 2 ** (l + 1)
            squarerepresentationlist = [0 for j in range(0, level)]
            if is_even(l + 1):
                squarerepresentationlist[0] = squarerepresentationlist[2 ** (l - 1)] = 2 ** (
                            (l + 1) / 2)
            else:
                squarerepresentationlist[0] = squarerepresentationlist[2 ** l] = 2 ** (l / 2)
            for k in range(0, l - 1, 2):
                # if debug > 0: print "k:", k
                for a in range(1, 2 ** (l + 1 - k), 8):
                    # if debug > 0: print "a:", a
                    squarerepresentationlist[2 ** k * a] = 2 ** (k / 2 + 2)
            # if debug > 0: print "Test the squarelist:", sum(squarerepresentationlist) == level,
            # squarerepresentationlist, level

            # if debug > 0: print "tvalues", tvalues

            t1inverse = inverse_mod(tvalues[0], level)
            values = [squarerepresentationlist[(j * t1inverse) % level] / 2 for j in
                      range(0, level)]

            if len(tvalues) > 1:
                # The following works only for tvalues where the last elements coincide

                t2inverse = inverse_mod(tvalues[1], level)
                newvalues = [squarerepresentationlist[(j * t2inverse) % level] / 2
                             for j in range(0, level)]

                for j in range(1, len(tvalues)):
                    values = combine_lists(values, newvalues)

            neven = n - len(tvalues)
            if neven > 0:
                values = combine_lists(values, values_even2adic((p, l, neven, +1)))

            return values

        _P = n.prime_divisors()
        if 2 in _P:

            _P.remove(2)

            l = sorted([q for q in self.__jd.keys() if 0 == q % 2])
            if debug > 0:
                print(l)
            while l:
                q = l.pop()
                gs = self.__jd[q][1]
                if len(gs) > 4:
                    values = combine_lists(values, values_odd2adic(gs))
                else:
                    values = combine_lists(values, values_even2adic(gs))
                if debug > 0:
                    print(values)

        _P.sort(reverse=True)

        while _P:
            p = _P.pop()
            if debug > 0:
                print("p = {0}".format(p))
            shortorbitdict = self.orbit_list(p, short=True)
            level = max(q for q in self.__jd.keys() if 0 == q % p)
            newvalues = [0 for j in range(0, level)]
            newvalues[0] = 1

            for orbit in shortorbitdict.keys():

                if orbit != (1,):

                    k = Integer(valuation(orbit[0], p) - 1)
                    # if debug > 0: print orbit
                    v1 = orbit[1]
                    if v1 == orbit[k + 1]:

                        orbitlengthsbykronecker = shortorbitdict[orbit]
                        for t1 in range(0, p ** (k + 1)):
                            newvalues[Integer(v1 * t1 * level / p ** (k + 1)) % level] += \
                                orbitlengthsbykronecker[kronecker(t1, p) + 1]

                    else:
                        index_of_new_value = Integer(v1 * orbit[k + 2] * level) % level
                        newvalues[index_of_new_value] += shortorbitdict[orbit]

                    # if debug > 0: print "Position1:", sum(newvalues)
            # if debug > 0: print "1:", values
            # if debug > 0: print "2:", newvalues
            values = combine_lists(values, newvalues)
            # if debug > 0: print "3:", values
            # if debug > 0: print "Position2:", values, _P, p

        # if debug > 0: print "Position3:", values

        valuesdict = {Integer(j) / len(values): values[j]
                      for j in range(0, len(values)) if values[j] != 0}

        # if debug > 0: print "Position4:", values, valuesdict, "END"

        return valuesdict

    def two_torsion_values(self):
        r"""
        If this is the Jordan decomposition for $(M,Q)$, return the values of $Q(x)$
        for $x \in M_2=\{x \in M | 2*x = 0\}$, the subgroup of two-torsion elements as a dict.

        OUTPUT:

            dict -- the mapping Q(x) --> the number two-torsion elements x with the same value Q(x)

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule, JordanDecomposition
            sage: A = FiniteQuadraticModule('2_3^-3.4^2.8_2^-2')
            sage: J = JordanDecomposition(A)
            sage: J.two_torsion_values() == {0: 48, 1/4: 16, 1/2: 16, 3/4: 48}
            True
        """
        n = self.__A.order()

        values = [1]

        def combine_lists(list1, list2):
            N1 = len(list1)
            N2 = len(list2)
            newlength = lcm(N1, N2)
            newlist = [0] * newlength
            for j1 in range(0, N1):
                n1 = list1[j1]
                if n1 != 0:
                    for j2 in range(0, N2):
                        n2 = list2[j2]
                        if n2 != 0:
                            new_index = (j1 * Integer(newlength / N1) +
                                         j2 * Integer(newlength / N2)) % newlength
                            newlist[new_index] += (n1 * n2)
            return newlist

        def two_torsion_values_even2adic(gs):
            p, k, n, eps = gs
            n /= 2
            fourn = 4 ** n
            if k == 1:
                epstwon = eps * 2 ** n
                return [(fourn + epstwon) / 2, (fourn - epstwon) / 2]
            else:
                return [fourn]

        def two_torsion_values_odd2adic(gs):
            p, k, n, eps, t = gs
            if k == 1:
                # print "n:", n, "eps:", eps, "t:", t, "n-t:", n-t, (n-t)/2
                if eps == -1:
                    t = (t + 4) % 8
                # TODO: Check if this should really use the old integer division
                n2 = ((n - t) // 2) % 4
                n1 = n - n2
                # print "n1:", n1, "n2:", n2, "t:", t
                list1 = [sum([binomial(n1, k) for k in range(j, n1 + 1, 4)]) for j in range(0, 4)]
                # print list1
                if n2 == 0:
                    return list1
                else:
                    list2 = [[1, 0, 0, 1], [1, 0, 1, 2], [1, 1, 3, 3]][n2 - 1]
                    # print list2
                    return combine_lists(list1, list2)
            elif k == 2:
                twonminusone = 2 ** (n - 1)
                return [twonminusone, twonminusone]
            else:
                return [2 ** n]

        even_qs = sorted([q for q in self.__jd.keys() if 0 == q % 2])
        while even_qs:
            q = even_qs.pop()
            gs = self.__jd[q][1]
            if len(gs) > 4:
                values = combine_lists(values, two_torsion_values_odd2adic(gs))
            else:
                values = combine_lists(values, two_torsion_values_even2adic(gs))

        valuesdict = {Integer(j) / len(values): values[j]
                      for j in range(0, len(values)) if values[j] != 0}

        return valuesdict

    def constituent(self, q):
        r"""
        Return the Jordan constituent whose exponent is the
        prime power "q".

        INPUT:

        - ``q`` - a prime power

        EXAMPLES:

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: A.jordan_decomposition().constituent(2)
            2^2
            sage: A.jordan_decomposition().constituent(4)
            4_1
            sage: A.jordan_decomposition().constituent(3) is None
            True
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition().constituent(3)
            3^2
            sage: A.jordan_decomposition().constituent(5)
            5^-1
            sage: A.jordan_decomposition().constituent(7)
            7^4
        """
        if not is_prime_power(q):
            raise TypeError
        return self._jordan_components.get(q)

    def finite_quadratic_module(self):
        r"""
        Return the finite quadratic module who initialized
        this Jordan decomposition.

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: A.jordan_decomposition().finite_quadratic_module() == A
            True
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition().finite_quadratic_module() == A
            True

        """
        return self.__A

    def basis(self, p=None):
        r"""
        Return a basis of this Jordan decomposition given by the basis in each component.

        INPUT:

        - ``p`` -- prime (default: ``None``)

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: A.jordan_decomposition().basis()
            [e0, e1, e2]
            sage: A.jordan_decomposition().basis() == list(A.gens())
            True
            sage: A = FiniteQuadraticModule('3^2.5^-1.7^4')
            sage: A.jordan_decomposition().basis()
            [e0, e1, e2, e3, e4, e5, e6]
            sage: A.jordan_decomposition().basis() == list(A.gens())
            True
        """
        return flatten([x.basis() for x in self if p is None or x.p == p])

    @staticmethod
    def is_type_I(F):
        r"""
        Return True if the matrix F corresponds to a Gram matrix of a type-I component,
        otherwise return False.

        INPUT:

        - 'F' -- an integer matrix

        EXAMPLES::

            sage: from fqm_weil.all import FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('2^2.4_1')
            sage: F = A.gram_bilinear()*4
            sage: A.jordan_decomposition().is_type_I(F)
            True
            sage: A = FiniteQuadraticModule('2^2')
            sage: F = A.gram_bilinear()*2
            sage: A.jordan_decomposition().is_type_I(F)
            False
            sage: A = FiniteQuadraticModule('2_2^2')
            sage: F = A.gram_bilinear() * 2
            sage: A.jordan_decomposition().is_type_I(F)
            True
            sage: F = A.gram_bilinear()
            sage: A.jordan_decomposition().is_type_I(F)
            Traceback (most recent call last):
            ...
            ValueError: F must have integer diagonal


        """
        for i in range(F.nrows()):
            if not F[i, i] in ZZ:
                raise ValueError("F must have integer diagonal")
            if is_odd(F[i, i]):
                return True
        return False

    def decompose(self, p0=None):
        """
        List of all indecomposable Jordan components of self.

        INPUT:

        - ``p0`` -- a prime (default: ``None``)


        EXAMPLES::

            sage: from fqm_weil.all import JordanComponent, FiniteQuadraticModule
            sage: A = FiniteQuadraticModule('3^2')
            sage: A.jordan_decomposition().decompose()
            [3, 3]
            sage: A = FiniteQuadraticModule('2_1^3')
            sage: A.jordan_decomposition().decompose()
            [2_1, 2_1, 2_5^-1]
            sage: A = FiniteQuadraticModule('2_1^-5')
            sage: A.jordan_decomposition().decompose()
            [2_1, 2_1, 2_1, 2_1, 2_1]

        """
        return flatten([jc.decompose() for jc in self if p0 is None or jc.p == p0])

from sage.arith.misc import hilbert_symbol
from sage.matrix.matrix0 import Matrix
from sage.modular.arithgroup.arithgroup_element import ArithmeticSubgroupElement


def _entries(*args):
    r"""
    Returns the entries of A
    where A is one of:
    1) Element of SL2Z
    2) 2x2 integer matrix with determinant 1
    3) list [a,b,c,d] with ad-bc=1

    EXAMPLES::

        sage: from fqm_weil.modules.weil_module.utils import _entries
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

"""
Utility functions for computing with modular forms for Weil representations.
"""
from fqm_weil.modules.finite_quadratic_module.finite_quadratic_module_base import \
    FiniteQuadraticModule_base
from fqm_weil.modules.weil_module.weil_module import WeilModule
import logging
from logging import getLogger

from sage.all import ZZ, QQ
from sage.arith.misc import gcd, xgcd
from sage.matrix.constructor import matrix
from sage.rings.integer import Integer
from sage.rings.number_field.number_field import CyclotomicField
from sage.rings.rational import Rational

log = getLogger(__name__)


def cusp_normalisers_and_stabilisers(group) -> dict:
    cusp_normalisers = {}
    cusp_stabilisers = {}
    cusps = group.cusps()
    cusps.sort(reverse=True)
    for cusp in cusps:
        a = cusp.numerator()
        c = cusp.denominator()
        w = group.level() / gcd(group.level(), c**2)
        Tp = matrix(ZZ, [[1-c*a*w, a**2*w], [-c**2*w, 1+a*c*w]])
        g, s, t = xgcd(a, c)
        Ai = matrix([[a, -t], [c, s]])
        cusp_normalisers[cusp] = Ai
        cusp_stabilisers[cusp] = Tp
    return {
        'cusp_normalisers': cusp_normalisers,
        'cusp_stabilisers': cusp_stabilisers
    }


def exp_as_zN_power(N, arg):
    """
    Return e(arg) as a power of a primitive N-th root of unit z_N if possible.

    INPUT:

    - ``N`` -- positive integer
    - ``arg`` -- rational number
    """
    if not isinstance(N, (int, Integer)):
        raise ValueError("N must be an integer")
    if not isinstance(arg, Rational):
        raise ValueError("arg must be a rational number")

    zN = CyclotomicField(N).gen()
    m = arg.numerator()
    n = arg.denominator()
    # We must have n | N
    if N % n != 0:
        raise ArithmeticError(f"Denominator `{n}` should divide the level `{N}`")
    return zN ** (m * N // n)
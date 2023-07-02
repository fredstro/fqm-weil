"""
Testing routines for finite quadratic modules.

"""
from math import lcm

from sage.all import ZZ, CC
from sage.arith.misc import valuation, kronecker, gcd
from sage.functions.other import floor
from sage.misc.prandom import random
from sage.rings.integer import Integer
from sage.rings.number_field.number_field import CyclotomicField

from fqm_weil.modules.finite_quadratic_module.finite_quadratic_module_ambient import \
    FiniteQuadraticModule, FiniteQuadraticModuleRandom
from fqm_weil.modules.finite_quadratic_module.jordan_decomposition import JordanDecomposition


def test_fqm_random(fqbound=100, nbound=10, cbound=10, size_bd=50, verbose=0):
    r"""
    Check nbound different random finite quadratic modules.

    The tests we do are:
    1) Compute Jordan block
    2) Compute sigma invariant
    """
    # first get a random FQM
    ntest = 0
    for i in range(nbound):
        l = size_bd + 1
        while l > size_bd:
            FQ = FiniteQuadraticModuleRandom(fqbound, nbound, verbose - 1)
            l = len(list(FQ))
            # info=get_factors2(FQ.jordan_decomposition())
        t = test_one_F(FQ)
        if t != True:
            return t
        ntest += 1
        # assert s0 == si[0]/si[1]
    if verbose > 0:
        print("Tested {0} modules!".format(ntest))
    return True


def test_one_F(FQ='4_1', verbose=0):
    if not hasattr(FQ, "Q"):
        FQ = FiniteQuadraticModule(FQ)
    N = FQ.level()
    z = CyclotomicField(N).gens()[0]
    for a in range(1, N):
        s0 = FQ.char_invariant(a)
        s1 = naive_Gauss_sum(FQ, a)
        # print s0,s1
        if abs(CC(s0[0]) * CC(s0[1]) - CC(s1[0]) / CC(s1[1] ** 2)) > 1e-10:
            if verbose > 0:
                print("a={0}".format(a))
                print("s0={0},{1}".format(s0, CC(s0[0] * s0[1])))
                print("s1={0},{1}".format(s1, CC(s1[0] / s1[1] ** 2)))
            return False, a, FQ
    return True


def test_fqm_from_signature(num_tests=10, num_comp=4, prime_bd=5, pow_bd=3, verbose=0):
    r"""
    Check that the genus symbol determines the finite quadratic module.
    """
    from sage.all import prime_range, random, random_prime
    for n in range(num_tests):
        s = ""
        for i in range(num_comp):
            p = random_prime(prime_bd)
            k = ZZ.random_element(1, pow_bd + 1)
            q = p ** k
            if p == 2:
                if random() < 0.5:  ## Make a type 1 factor
                    t = 2 * ZZ.random_element(4) + 1
                    if kronecker(t, 2) == 1:
                        s += "{q}_{t}^1".format(t=t, q=q)
                    else:
                        s += "{q}_{t}^-1".format(t=t, q=q)
                else:
                    if random() < 0.5:
                        s += "{q}^2".format(q=q)  # B
                    else:
                        s += "{q}^-2".format(q=q)  # C
            else:
                if random() < 0.5:
                    s += "{q}^1".format(q=q)
                else:
                    s += "{q}^-1".format(q=q)
            if i < num_comp - 1:
                s += "."
        M = FiniteQuadraticModule(s)
        s0 = M.jordan_decomposition().genus_symbol()
        N = FiniteQuadraticModule(s0)
        if verbose > 0:
            print("s=", s)
            print("s0=", s0)
        if verbose > 1:
            print("M={0}".format(M))
            print("N={0}".format(N))

        if not M.is_isomorphic(N):
            raise ArithmeticError("{0} and {1} with symbol {2} are not isomorphic!".
                                  format(M, N, s))
        # if(FQ.level() % 4 != 0 or is_odd(info['sign'])):
        #     continue
        # if verbose>0:
        #     print "FQ=",FQ," size=",l," level=",FQ.level()," sigma^4=",FQ.sigma_invariant()^4
        #     print "Genus symbol=",FQ.jordan_decomposition().genus_symbol()

        # if test_epd_one_fqm(FQ,cbound):
        #     print "ok for FQ=",i
        # else:
        #     print "*Not* ok for FQ=",i
        #     return False
    # Functions for testing


def naive_Gauss_sum(FQ, a, y=0):
    r"""
    If this quadratic module equals $A = (M,Q)$, return
    the Gauss sum of the of $A$ at $a$, i.e. return
    $$\chi_A (a)= sqrt(|M|/gcd(|M|,a))^{-1}\sum_{x\in M} \exp(2\pi i [a Q(x) + B(x,y)]))$$
    computed naively by summing.
    NOTE: This is slow and should only be used for testing purposes.
    """
    N = lcm(FQ.level(), 8)
    KN = CyclotomicField(N)
    z = KN.gens()[0]
    gauss_sum = sum(z ** (a * (FQ.Q(x) * N) + FQ.B(x, y)) for x in FQ)
    M = FQ.order()
    return CyclotomicField(8)(gauss_sum / KN(ZZ(M * gcd(a, M)).sqrt()))


def orbitlist_test(str=None):
    r"""
    testing if all orbits sum up to the size of the
    finite quadratic module
    """
    if str:
        A = FiniteQuadraticModule(str)
    else:
        A = FiniteQuadraticModuleRandom(discbound=10000, normbound=10000, verbose=0)
    J = JordanDecomposition(A)
    str = J.genus_symbol()
    print(str)
    olist = J.orbit_list()
    testpassed = True
    for p in olist.keys():
        st = J.genus_symbol(p)
        order = p ** valuation(A.order(), p)
        orbitsum = sum(olist[p][key] for key in olist[p].keys())
        # print "   order of A:", order
        # print "sum of orbits:", orbitsum
        testpassed = testpassed and (order == orbitsum)
        s = "{0}: # of elements in the computed orbits sum up to the order of {1} : {2}, {3}, {4}"
        s = s.format(p.str(), st, order == orbitsum, order, orbitsum)
        print(s)
    return testpassed


def values_test(str):
    r"""
    testing if the computed dictionary of values sums up
    to the size of the finite quadratic module
    """
    A = FiniteQuadraticModule(str)
    J = JordanDecomposition(A)

    print(str)

    # valuesdict, values = J.values()
    valuesdict = list(J.values())

    # print "Position1"

    Avalues = list(A.values())
    b1 = valuesdict == Avalues
    print("Test A.values() == J.values():{0}".format(b1))

    # print "Position2"

    b2 = sum([valuesdict[key] for key in valuesdict.keys()]) == A.order()
    print("Test sum(values) == A.order(): {0}".format(b2))

    # print "Position3"

    Atwotorsionvalues = list(A.kernel_subgroup(2).as_ambient()[0].values())
    Jtwotorsionvalues = J.two_torsion_values()

    b3 = Atwotorsionvalues == Jtwotorsionvalues
    print("Test two_torsion_values():    {0}".format(b3))

    # print Avalues
    # print valuesdict

    if not b3:
        print("A:{0}".format(Atwotorsionvalues))
        print("J:{0}".format(Jtwotorsionvalues))

    return b1 and b2 and b3


def testing_routine(p):
    r"""
    testing discriminant only with p components
    """
    k = Integer(3)
    p = Integer(p)
    p1 = p.str()
    p2 = (p ** 2).str()
    p3 = (p ** 3).str()
    p4 = (p ** 4).str()
    p5 = (p ** 5).str()
    str = ''
    for a in range(-k, k + 1):
        if a != 0:
            astr = Integer(a).str()
            str1 = str + p1 + '^' + astr + '.'
        else:
            str1 = str
        print("str1:{0}".format(str1))
        for b in range(-k + 1, k):
            if b != 0:
                bstr = Integer(b).str()
                str2 = str1 + p2 + '^' + bstr + '.'
            else:
                str2 = str1
            print("    str2:{0}".format(str2))
            for c in range(-k + 3, k - 2):
                if c != 0:
                    cstr = Integer(c).str()
                    str3 = str2 + p3 + '^' + cstr + '.'
                else:
                    str3 = str2
                print("        str3:{0}".format(str3))
                for d in range(0, 1):  # range(-k+4,k-3)
                    if d != 0:
                        dstr = Integer(d).str()
                        str4 = str3 + p4 + '^' + dstr + '.'
                    else:
                        str4 = str3
                    print("            str4:{0}".format(str4))
                    for e in range(0, 1):  # range(-k+4,k-3)
                        if e != 0:
                            estr = Integer(e).str()
                            str5 = str4 + p5 + '^' + estr + '.'
                        else:
                            str5 = str4
                        print("                str5: {0}".format(str5))
                        str5 = str5[:-1]
                        if str5 != '':
                            # A = FiniteQuadraticModule(str5)
                            # J = JordanDecomposition(A)
                            if values_test(str5):
                                print(str, True)
                            else:
                                return str, False
    return True, "All tests successful"


def testing_routine_odd2adic():
    for p in [3, 5, 7, 9, 11, 13, 17, 19, 25]:

        q = Integer(2)

        while q < 2 ** 4:

            for oddstr in ['_0^2', '_0^-4', '_1^1', '_1^-3', '_2^2', '_2^-2', '_3^3',
                           '_3^-1', '_4^4', '_4^-2', '_5^3', '_5^-1', '_6^2', '_6^-2',
                           '_7^1', '_7^-3', '_0^4', '_0^-6', '_1^3', '_1^-5', '_2^4',
                           '_2^-4', '_3^5', '_3^-3', '_4^6', '_4^-4', '_5^5', '_5^-3',
                           '_6^4', '_6^-4', '_7^3', '_7^-5']:

                oddprimestr = '.' + Integer(p).str() + '^' + (
                            -1 + 2 * floor(2 * random())).str() + '.27^-1'
                if not values_test(q.str() + oddstr + oddprimestr):
                    return "Test not passed:", q.str() + oddstr + oddprimestr

            q *= 2


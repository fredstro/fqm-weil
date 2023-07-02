"""
Testing WeilModules

"""
from fqm_weil.modules.finite_quadratic_module.finite_quadratic_module_ambient import \
    FiniteQuadraticModuleRandom
from fqm_weil.modules.weil_module.weil_module import WeilModule
from sage.all import QQ, CC
from sage.modular.arithgroup.congroup_sl2z import SL2Z


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

r"""
Jordan decompositions of finite quadratic modules.

"""
from sage.all import ZZ
from sage.arith.functions import lcm
from sage.arith.misc import valuation, kronecker, is_prime, inverse_mod, is_prime_power
from sage.functions.other import floor, binomial
from sage.matrix.constructor import matrix
from sage.misc.functional import is_odd, is_even
from sage.misc.misc_c import prod
from sage.rings.integer import Integer
from sage.structure.sage_object import SageObject


class JordanDecomposition(SageObject):
    r"""
    A container class for the Jordan constituents of a
    finite quadratic module.

    EXAMPLES NONE
    """

    def __init__(self, A):
        r"""
        INPUT:
            A -- a nondegenerate finite quadratic module

        TODO: The case of degenerate modules.

        """
        self.__A = A
        if not A.is_nondegenerate():
            raise TypeError
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

    def _repr_(self):
        r"""
        EXAMPLES NONE
        """
        return 'Jordan decomposition'

    def _latex_(self):
        r"""
        EXAMPLES NONE
        """
        return 'Jordan decomposition'

    def __iter__(self):
        r"""
        Return the Jordan decomposition as iterator.

        NOTE
            The following is guaranteed. Returned is a list of pairs
            basis, (prime p,  valuation of p-power n, dimension r, determinant e over p[, oddity o]),
            where $n > 0$, ordered lexicographically by $p$, $n$.

        EXAMPLES NONE
        """
        return (self.__jd[p ** n] for p, n in self.__ol)

    def genus_symbol(self, p=None):
        r"""
        Return the genus symbol of the Jordan constituents
        whose exponent is a power of the prime $p$.
        Return the concatenation of all local genus symbols
        if no argument is given.

        EXAMPLES::

        We check that the calculation of the genus symbol is correct
        for 2-adic symbols.

            sage: from fqm_weil.all import FiniteQuadraticModule
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

        EXAMPLES NONE
        """
        l = [q for q in self.__jd.keys() if 0 == q % p]
        if [] == l:
            return ''
        l.sort(reverse=True)
        s = ''
        while l != []:
            q = l.pop()
            s += str(q)
            gs = self.__jd[q][1]
            e = gs[2] * gs[3]
            if len(gs) > 4:
                s += '_' + str(gs[4])
            if 1 != e:
                s += '^' + str(e)
            if l != []:
                s += '.'
        return s

    def orbit_list(self, p=None, short=False):
        r"""
        If this is the Jordan decomposition for $(M,Q)$, return the dictionary of
        dictionaries of orbits corresponding to the p-groups of $M$.
        If a prime p is given only the dictionary of orbits for the p-group is returned.
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
            while l != []:
                q = l.pop()
                gs = self.__jd[q][1]
                if len(gs) > 4:
                    values = combine_lists(values, values_odd2adic(gs))
                else:
                    values = combine_lists(values, values_even2adic(gs))
                if debug > 0:
                    print(values)

        _P.sort(reverse=True)

        while [] != _P:
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
        for $x \in M_2=\{x \in M | 2*x = 0\}$, the subgroup of two-torsion elements as a dictionary.

        OUTPUT:
            dictionary -- the mapping Q(x) --> the number two-torsion elements x with the same value Q(x)

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
            p, l, n, eps = gs
            n /= 2
            fourn = 4 ** n
            if l == 1:
                epstwon = eps * 2 ** n
                return [(fourn + epstwon) / 2, (fourn - epstwon) / 2]
            else:
                return [fourn]

        def two_torsion_values_odd2adic(gs):
            p, l, n, eps, t = gs
            if l == 1:
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
            elif l == 2:
                twonminusone = 2 ** (n - 1)
                return [twonminusone, twonminusone]
            else:
                return [2 ** n]

        l = sorted([q for q in self.__jd.keys() if 0 == q % 2])
        while l != []:
            q = l.pop()
            gs = self.__jd[q][1]
            if len(gs) > 4:
                values = combine_lists(values, two_torsion_values_odd2adic(gs))
            else:
                values = combine_lists(values, two_torsion_values_even2adic(gs))

        valuesdict = {Integer(j) / len(values): values[j]
                      for j in range(0, len(values)) if values[j] != 0}

        return valuesdict

    def constituent(self, q, names=None):
        r"""
        Return the Jordan constituent whose exponent is the
        prime power "q".

        EXAMPLES NONE
        """
        if not is_prime_power(q):
            raise TypeError
        gens = self.__jd.get(q, ((), ()))[0]
        # print("gens=",gens)
        return self.__A.spawn(gens, names)

    def finite_quadratic_module(self):
        r"""
        Return the finite quadratic module who initialized
        this Jordan decomposition.

        EXAMPLES NONE
        """
        return self.__A

    def basis(p=None):
        r"""
        TODO
        """
        raise NotImplementedError

    @staticmethod
    def is_type_I(F):
        r"""
        EXAMPLES NONE
        """
        for i in range(F.nrows()):
            if is_odd(F[i, i]):
                return True
        return False

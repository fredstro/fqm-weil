# Utility functions copied over from the main psage repository.
cdef extern from "math.h":
    double fabs(double)
    double fmax(double,double)
    int ceil(double)
    int floor(double)
    double M_LN10
    double log(double)
    float INFINITY

import cython

cpdef factor_matrix_in_sl2z(A, B=None, C=None, D=None, int verbose=0):
    r"""
    Factor a matrix from SL2Z in S and T.
    INPUT:

    - A -- 2x2 integer matrix with determinant 1 (Note: using SL2Z elements are much slower than using integer matrices)

    OUTPUT:
    - l -- list of the form l=[z,n,[a_1,...,a_n]] if A=z T^n ST^{a_1}\cdots ST^{a_n}.

    EXAMPLES::

        sage: from fqm_weil.modules.utils import factor_matrix_in_sl2z
        sage: A=SL2Z([-28,-5,-67,-12])
        sage: factor_matrix_in_sl2z(A)
        (1, 0, [-2, 2, -2, -6, 0])


    """
    cdef int a, b, c, d
    if D is not None:  # If we provide 4 arguments they should be integers. isinstance(A,tuple):
        if A * D - B * C != 1:
            raise ValueError("Matrix does not have determinant 1!")
        return fast_sl2z_factor(A, B, C, D)
    else:
        if isinstance(A, (list, tuple)):
            a, b, c, d = A
        else:
            a = A[0, 0];
            b = A[0, 1];
            c = A[1, 0];
            d = A[1, 1]
        if a * d - b * c != 1:
            raise ValueError("Matrix does not have determinant 1!")
        return fast_sl2z_factor(a, b, c, d)

@cython.cdivision(True)
cdef void _apply_sl2z_map_dp(double *x,double *y,int a, int b, int c,int d):
    """
    Fast double precision application of a map from SL2Z to point in the upper half-plane.
    Note: no checks are done in this function. 
    """
    cdef double ar,br,cr,dr
    cdef double den,tmp1,tmp2
    ar=<double>a
    br=<double>b
    cr=<double>c
    dr=<double>d
    den=(cr*x[0]+dr)**2+(cr*y[0])**2
    tmp1 = ar*cr*(x[0]*x[0]+y[0]*y[0])+(a*d+b*c)*x[0]+br*dr
    if den==0:
        y[0]=INFINITY
        x[0]=0
    else:
        x[0] = tmp1/den
        y[0] = y[0]/den

cdef void _apply_one_pb_map(double *x, double *y, int *mapping, int *n, int * a, int * b, int * c,
                       int * d):
    r"""
    Do one (or rather two) steps in the pullback algorithm
    """

    cdef double absval, minus_one, half
    mapping[0] = 0
    if fabs(x[0]) > 0.5:
        mapping[0] = 1
        half = <double> 0.5
        if x[0] > 0:
            dx = x[0] - half
            n[0] = -ceil(dx)
        else:
            dx = -x[0] - half
            n[0] = ceil(dx)
        x[0] = x[0] + <double> n[0]
        #print "n=",n[0]
        a[0] = a[0] + n[0] * c[0]
        b[0] = b[0] + n[0] * d[0]
    else:
        n[0] = 0
    absval = x[0] * x[0] + y[0] * y[0]
    if absval < 1.0 - 1E-15:
        mapping[0] = 2  # meaning that we applied S*T^n
        minus_one = <double> -1.0
        x[0] = minus_one * x[0] / absval
        y[0] = y[0] / absval
        aa = a[0]
        bb = b[0]
        a[0] = -c[0]
        b[0] = -d[0]
        c[0] = aa
        d[0] = bb

cpdef tuple fast_sl2z_factor(int a, int b, int c, int d, int verbose=0):
    r"""
    Factor a matrix in S and T.
    
    INPUT:
    
    - a,b,c,d -- integers with ad-bc = 1, representing A in SL2Z
    
    OUTPUT:
    
    - pref,ntrans,l -- tuple with
        - pref == integer, 1 or -1
        - ntrans -- integer
        - l -- list of the form l=[ntrans,[a_1,...,a_n]] if A=z pref*T^ntrans ST^{a_1}\cdots ST^{a_n}.

    EXAMPLES:

        sage: from fqm_weil.modules.utils import fast_sl2z_factor
        sage: A=SL2Z([-28,-5,-67,-12])
        sage: a,b,c,d=A
        sage: fast_sl2z_factor(a,b,c,d)
        (1, 0, [-2, 2, -2, -6, 0])
    """
    cdef double x, y
    x = 0.0
    y = 2.0
    _apply_sl2z_map_dp(&x, &y, d, -b, -c, a)
    cdef int aa, bb, cc, dd
    cdef int mapping
    aa = 1
    bb = 0
    cc = 0
    dd = 1
    # Now z = A(2i) and we want to pullback z
    l = list()
    cdef int maxc, n, pref, ntrans
    cdef char ch
    # This number 'maxc' of steps should be sufficient (otherwise an ArithmeticError is raised).
    if d != 0:
        maxc = (abs(c) + abs(d) + 1)
    else:
        maxc = (abs(c) + abs(a) + 1)
    ntrans = 0
    pref = 1
    for i in range(maxc):
        _apply_one_pb_map(&x, &y, &mapping, &n, &aa, &bb, &cc, &dd)
        # Note that we will never use the same mapping twice in a row here
        if mapping == 1:  # then we had a single translation, which might appear at the beginning
            ntrans = n
        elif mapping == 2:
            l.insert(0, n)
        else:
            break
    if i == maxc:
        raise ArithmeticError(
            " Pullback failed! need to increse number of iterations. Used {0}".format(maxc))
    if aa != a or dd != d or cc != c or bb != b:
        # check -A
        if aa == -a and dd == -d and cc == -c and bb == -b:
            pref = -1
        else:
            raise ArithmeticError(
                " Could not pullback! A={0}, AA={1}".format((a, b, c, d), (aa, bb, cc, dd)))
    return pref, ntrans, l

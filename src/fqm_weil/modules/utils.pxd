
cdef void _apply_sl2z_map_dp(double *x, double *y, int a, int b, int c, int d)

cdef void _apply_one_pb_map(double *x, double *y, int *mapping, int *n, int * a, int * b, int * c,
                            int * d)

cdef tuple fast_sl2z_factor(int a, int b, int c, int d, int verbose=?)
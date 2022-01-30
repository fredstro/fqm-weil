"""
Vectors with mpfr entries. 

AUTHOR:
    -- Fredrik Stroemberg (2011)

NOTE: based on vector_complex_dense.pyx which in turn was copied from sage/modules/vector_rational_dense.pyx
and


EXAMPLES:


TESTS:

"""

###############################################################################
#   Sage: System for Algebra and Geometry Experimentation    
#       Copyright (C) 2007 William Stein <wstein@gmail.com>
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
###############################################################################




# set rounding to be nearest integer
# TODO: make t possible to change rounding
from sage.libs.mpfr.types cimport MPFR_RNDN
from sage.libs.mpfr cimport *
from sage.libs.mpfr cimport mpfr_init2, mpfr_set_si, mpfr_sub, mpfr_set_prec
from sage.ext.stdsage cimport PY_NEW
from cysignals.memory cimport sig_free,sig_malloc
from cysignals.signals cimport sig_on,sig_off
from sage.rings.real_mpfr import RealField
from sage.matrix.matrix cimport Matrix

cdef mpfr_rnd_t rnd
rnd = MPFR_RNDN

from sage.structure.element cimport Element, ModuleElement, RingElement, Vector
from sage.all import FreeModule

from sage.rings.integer cimport Integer

from sage.modules.free_module_element cimport FreeModuleElement
from sage.rings.real_mpfr cimport RealNumber,RealField_class
from psage.rings.mpfr_nogil cimport mpfr_t, mpfr_set, mpfr_clear, mpfr_set_ui, mpfr_cmp, mpfr_add, mpfr_mul, mpfr_neg, mpfr_abs, mpfr_sqr, mpfr_sqrt
from psage.matrix.matrix_complex_dense import Matrix_complex_dense

cdef class Vector_real_mpfr_dense(FreeModuleElement):
    cdef bint is_dense_c(self):
        return 1
    cdef bint is_sparse_c(self):
        return 0

    cdef _new_c(self,v=None):
        cdef Vector_real_mpfr_dense y
        y = Vector_real_mpfr_dense.__new__(Vector_real_mpfr_dense,self._parent,v,False,False)
        return y

    def __copy__(self):
        cdef Vector_real_mpfr_dense y
        y = self._new_c()
        cdef Py_ssize_t i
        for i in range(self._degree):
            mpfr_init2(y._entries[i],self._prec)
            mpfr_set(y._entries[i], self._entries[i],rnd)
        return y

    cdef _init(self, Py_ssize_t degree, parent):
        self._degree = degree
        self._parent = parent
        self._base_ring=parent._base
        self._prec = self._base_ring.__prec
        self._entries = <mpfr_t *> sig_malloc(sizeof(mpfr_t) * degree)
        for i in range(self._degree):
            mpfr_init2(self._entries[i],self._prec)
        if self._entries == NULL:
            raise MemoryError

    def __cinit__(self, parent, x, coerce=True,copy=True):
        cdef RealNumber z
        self._entries = NULL
        self._is_mutable = 1
        #print "in __cinit__"
        self._init(parent.degree(), parent)
        # Set entries
        if isinstance(x, (list, tuple)) or (hasattr(x,'parent') and x.parent() == parent) :
            if len(x) != self._degree:
                raise TypeError("entries must be a list of length %s"%self._degree)
            for i in range(self._degree):
                z = RealNumber(self._base_ring,x[i])
                mpfr_set(self._entries[i], z.value,rnd)
        elif x == 0 or x is None:
            for i in range(self._degree):
                mpfr_set_ui(self._entries[i], 0,rnd)
        else:
            raise TypeError("can't initialize vector from nonzero non-list")


        
    def __init__(self, parent, x, coerce=True, copy=True):
        pass



    def __dealloc__(self):
        cdef Py_ssize_t i
        if self._entries!=NULL:
            sig_on()
            for i in range(self._degree):
                mpfr_clear(self._entries[i])
            sig_off()
            #print "clearing python entries"
            sig_free(self._entries)

    cpdef base_ring(self):
        return self._base_ring

    cdef int _cmp_c_impl(left, Element right) except -2:
        """
        
        EXAMPLES::
        
            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,0)
            sage: v == 0
            True
            sage: v == 1
            False
            sage: v == v
            True
            sage: w = Vector_real_mpfr_dense(S, [-1,3/2,0])
            sage: w < v
            True
            sage: w > v
            False
        """
        cdef Py_ssize_t i
        cdef int c
        for i in range(left.degree()):
            c = mpfr_cmp(left._entries[i], (<Vector_real_mpfr_dense>right)._entries[i])
            if c < 0:
                return -1
            elif c > 0:
                return 1
        return 0

    def __len__(self):
        return self._degree

    def __setitem__(self, Py_ssize_t i, x):
        if not self._is_mutable:
            raise ValueError("vector is immutable; please change a copy instead (use copy())")
        cdef RealNumber z
        if i < 0 or i >= self._degree:
            raise IndexError
        else:
            z = RealNumber(self._base_ring,x)
            mpfr_set(self._entries[i], z.value,rnd)
            
    def __getitem__(self, Py_ssize_t i):
        """
        Return the ith entry of self.

        EXAMPLES::

            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,[1/2,2/3,3/4]); v
            (0.500000000000000, 0.666666666666667, 0.750000000000000)
            sage: v[0]
            0.500000000000000
            sage: v[2]
            0.750000000000000
            sage: v[-2]
            0.666666666666667
            sage: v[5]
            Traceback (most recent call last):
            ...
            IndexError: index out of range
            sage: v[-5]
            Traceback (most recent call last):
            ...
            IndexError: index out of range
        """
        cdef RealNumber z
        z = RealNumber(self._base_ring,0)
        if i < 0:
            i += self._degree
        if i < 0 or i >= self._degree:
            raise IndexError('index out of range')
        else:
            mpfr_set(z.value, self._entries[i],rnd)
            return z

    def __reduce__(self):
        return (unpickle_v1, (self._parent, self.list(), self._degree, self._is_mutable))

    cpdef _add_(self, right):
        cdef Vector_real_mpfr_dense z
        cdef Vector_real_mpfr_dense r = right
        #print "in add!"
        z = self._new_c()
        cdef Py_ssize_t i
        #prec=self.parent().base_ring().prec()
        for i in range(self._degree):
            mpfr_init2(z._entries[i],self._prec)
            mpfr_add(z._entries[i], self._entries[i], r._entries[i],rnd)
        return z
        

    cpdef _sub_(self,right):
        cdef Vector_real_mpfr_dense z
        cdef Vector_real_mpfr_dense r = right
        #print "in sub!"
        z = self._new_c()
        cdef Py_ssize_t i
        for i in range(self._degree):
            mpfr_init2(z._entries[i],self._prec)
            mpfr_sub(z._entries[i], self._entries[i], r._entries[i],rnd)
        return z
        
    cpdef Element _dot_product_(self, Vector right):
        """
        Dot product of dense vectors over mpfr.
        
        EXAMPLES::
        
            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,[1,2,-3])
            sage: w = Vector_real_mpfr_dense(S,[4,3,2])
            sage: v*w
            4.00000000000000
            sage: w*v
            4.00000000000000
        """
        cdef Vector_real_mpfr_dense r = right
        cdef RealNumber z
        z = RealNumber(self._base_ring,0)
        cdef mpfr_t t
        mpfr_init2(t,self._prec)
        mpfr_set_si(z.value, 0,rnd)
        cdef Py_ssize_t i
        for i in range(self._degree):
            mpfr_mul(t, self._entries[i], r._entries[i],rnd)
            mpfr_add(z.value, z.value, t,rnd)
        mpfr_clear(t)
        return z
    
    def scalar_product(self,right):
        """

        EXAMPLES::

            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,[1,2,-3])
            sage: w = Vector_real_mpfr_dense(S,[4,3,2])
            sage: v.scalar_product(w)
            4.00000000000000
            sage: w.scalar_product(v)
            4.00000000000000


        """
        return self._scalar_product_(right)

    cdef RealNumber _scalar_product_(self, Vector right):
        """
        Euclidean scalar product of dense vectors over mpfr.

        """
        cdef Vector_real_mpfr_dense r = right
        cdef RealNumber z
        z = RealNumber(self._base_ring,0)
        cdef mpfr_t t
        mpfr_init2(t,self._prec)
        mpfr_set_si(z.value, 0,rnd)
        cdef Py_ssize_t i
        for i in range(self._degree):
            mpfr_mul(t, self._entries[i], r._entries[i],rnd)
            mpfr_add(z.value, z.value, t,rnd)
        mpfr_clear(t)
        return z
        

    cpdef Vector _pairwise_product_(self, Vector right):
        """
        EXAMPLES::
        
            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,[1,2,-3]); v
            (1.00000000000000, 2.00000000000000, -3.00000000000000)            
            sage: w = Vector_real_mpfr_dense(S,[4,3,2]); w
            (4.00000000000000, 3.00000000000000, 2.00000000000000)
            sage: v.pairwise_product(w)
            (4.00000000000000, 6.00000000000000, -6.00000000000000)
        """
        cdef Vector_real_mpfr_dense z, r
        r = right
        z = self._new_c()
        cdef Py_ssize_t i
        for i in range(self._degree):
            mpfr_init2(z._entries[i],self._prec)
            mpfr_mul(z._entries[i], self._entries[i], r._entries[i],rnd)
        return z

        
    def __mul__(self, ModuleElement right):
        """

            EXAMPLES::

            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,[1,2,-3])
            sage: 2*v
            (2.00000000000000, 4.00000000000000, -6.00000000000000)
            sage: v*2
            (2.00000000000000, 4.00000000000000, -6.00000000000000)
            sage: A = matrix(RealField(53),[[1,1,1],[2,2,2],[3,3,3]])
            sage: from psage.matrix.matrix_complex_dense import Matrix_complex_dense
            sage: MS = MatrixSpace(MPComplexField(53),3,3)
            sage: B = Matrix_complex_dense(MS,[1,1,1,2,2,2,3,3,3])

        """
        cdef RealNumber a
        if isinstance(right,Vector_real_mpfr_dense):
            return self._dot_product_(right)
        if isinstance(right, Matrix_complex_dense):
            return Matrix_complex_dense._vector_times_matrix_(right, self)
        if isinstance(right,Matrix):
            return right*self
        return self._lmul_(right)


    cpdef Element _rmul_(self, Element left):
        cdef Vector_real_mpfr_dense z
        cdef Py_ssize_t i
        cdef RealNumber a
        z = self._new_c()
        # we can convert almost anything to MPComplexNumber
        if not isinstance(left,RealNumber):
            a = RealNumber(self._base_ring,left)
            for i in range(self._degree):
                mpfr_init2(z._entries[i],self._prec)
                mpfr_mul(z._entries[i], self._entries[i], a.value,rnd)
        else:
            for i in range(self._degree):
                mpfr_init2(z._entries[i],self._prec)
                mpfr_mul(z._entries[i], self._entries[i], (<RealNumber>(left)).value,rnd)
        return z


    cpdef _lmul_(self, Element right):
        cdef Vector_real_mpfr_dense z
        cdef RealNumber a
        # we can convert almost anything to MPComplexNumber
        if not isinstance(right,RealNumber):
             a = RealNumber(self._base_ring,right)
        else:
            a = right
        z = self._new_c()
        cdef Py_ssize_t i
        for i in range(self._degree):
            mpfr_init2(z._entries[i],self._prec)
            mpfr_mul(z._entries[i], self._entries[i], a.value,rnd)
        return z

    cpdef ModuleElement _neg_(self):
        cdef Vector_real_mpfr_dense z
        z = self._new_c()
        cdef Py_ssize_t i
        for i in range(self._degree):
            mpfr_init2(z._entries[i],self._prec)
            mpfr_neg(z._entries[i], self._entries[i],rnd)
        return z


    cpdef RealNumber norm(self,int ntype=2):
        r"""        
        The Euclidean norm of self.
        
        EXAMPLES:: 
        
            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,[1,2,-3])
            sage: v.norm()
            3.74165738677394
            sage: v.norm()**2
            14.0000000000000
        """
        cdef RealNumber res
        cdef mpfr_t x,s
        mpfr_init2(x,self._prec)
        mpfr_init2(s,self._prec)
        res = RealNumber(self._base_ring._base,0)
        if ntype==2:
            mpfr_set_si(s, 0, rnd)
            for i in range(self._degree):
                mpfr_abs(x,self._entries[i],rnd)
                mpfr_sqr(x,x,rnd)
                mpfr_add(s,s,x,rnd)
            mpfr_sqrt(s,s,rnd)
            mpfr_set(res.value,s,rnd)
            mpfr_clear(x); mpfr_clear(s)
            return res
        else:
            mpfr_clear(x); mpfr_clear(s)
            raise NotImplementedError("Only 2-norm is currently implemented")

    def prec(self):
        r"""

        EXAMPLES::

            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F=RealField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_real_mpfr_dense(S,[1,2,3])
            sage: v.prec()
            53
            sage: F=RealField(103)
            sage: S=FreeModule(F,3)
            sage: v=Vector_real_mpfr_dense(S,[1,2,3])
            sage: v.prec()
            103

        """
        return self._prec

    def set_prec(self,int prec):
        r"""
        Change the precision for self.

        EXAMPLES::

            sage: from psage.modules.vector_real_mpfr_dense import Vector_real_mpfr_dense
            sage: F = RealField(53)
            sage: S = FreeModule(F,3)
            sage: v = Vector_real_mpfr_dense(S,[1,2,-3]);v
            (1.00000000000000, 2.00000000000000, -3.00000000000000)
            sage: v.prec()
            53
            sage: v.set_prec(103)

        """
        cdef Py_ssize_t i,j
        cdef mpfr_t z
        # from sage.rings.complex_mpc import MPComplexField
        from sage.matrix.all import MatrixSpace
        mpfr_init2(z,prec)
        self._prec = prec
        self._base_ring=RealField(prec)
        self._parent = FreeModule(self._base_ring,self._degree)
        for i in range(self._degree):
                mpfr_set(z,self._entries[i],rnd)
                mpfr_set_prec(self._entries[i],prec)
                mpfr_set(self._entries[i],z,rnd)
        mpfr_clear(z)
        


def make_FreeModuleElement_complex_dense_v1(parent, entries, degree):
    """
    If you think you want to change this function, don't.
    Instead make a new version with a name like
       make_FreeModuleElement_generic_dense_v1
    and changed the reduce method below.
    """
    cdef Vector_real_mpfr_dense v
    v = Vector_real_mpfr_dense.__new__(Vector_real_mpfr_dense, parent, entries)
    return v

def unpickle_v1(parent, entries, degree, is_mutable):
    cdef Vector_real_mpfr_dense v
    v = Vector_real_mpfr_dense.__new__(Vector_real_mpfr_dense, parent, entries)
    return v
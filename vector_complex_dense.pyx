# encoding: utf-8
# cython: profile=False
#################################################################################
#
# (c) Copyright 2018 Fredrik Stromberg <fredrik314@gmail.com>
#
#  This file is part of PSAGE
#
#  PSAGE is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  PSAGE is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# 
# Based on the file src/sage/modules/vector_rational_dense.pyx with copyright notice:
###############################################################################
#   Sage: System for Algebra and Geometry Experimentation    
#       Copyright (C) 2007 William Stein <wstein@gmail.com>
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
###############################################################################


"""
Vectors with mpc entries. 

Based on the structure in src/sage/modules/vector_rational_dense.pyx

AUTHOR:
    -- Fredrik Stroemberg (2011)

EXAMPLES:
    sage: v = vector(QQ,[1,2,3,4,5])
    sage: v
    (1, 2, 3, 4, 5)
    sage: 3*v
    (3, 6, 9, 12, 15)
    sage: v/2
    (1/2, 1, 3/2, 2, 5/2)
    sage: -v
    (-1, -2, -3, -4, -5)
    sage: v - v
    (0, 0, 0, 0, 0)
    sage: v + v
    (2, 4, 6, 8, 10)
    sage: v * v
    55

We make a large zero vector:
    sage: k = QQ^100000; k
    Vector space of dimension 100000 over Rational Field
    sage: v = k(0)
    sage: v[:10]
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

TESTS:
    sage: v = vector(QQ, [1,2/5,-3/8,4])
    sage: loads(dumps(v)) == v
    True
"""
cimport cython
from cysignals.memory cimport sig_free,sig_malloc
from cysignals.signals cimport sig_on,sig_off
from sage.rings.complex_mpc cimport MPComplexNumber, MPComplexField_class
from psage.rings.mp_cimports cimport *
from sage.ext.stdsage cimport PY_NEW

from sage.all import FreeModule #,FreeModuleElement,ModuleElement
from sage.structure.element cimport Element,ModuleElement,RingElement
from sage.modules.free_module_element cimport FreeModuleElement

cdef class Vector_complex_dense(FreeModuleElement):

    cdef bint is_dense_c(self):
        return 1
    cdef bint is_sparse_c(self):
        return 0

    cdef _new_c(self,v=None):
        cdef Vector_complex_dense y
        y = Vector_complex_dense.__new__(Vector_complex_dense,self._parent,v,False,False)
        return y

    def __copy__(self):
        cdef Vector_complex_dense y
        y = self._new_c()
        cdef Py_ssize_t i
        for i from 0 <= i < self._degree:
            mpc_init2(y._entries[i],self._prec)
            mpc_set(y._entries[i], self._entries[i],rnd)
        return y

    cdef _init(self, Py_ssize_t degree, parent):
        self._degree = degree
        self._parent = parent
        self._base_ring=parent._base
        self._prec = self._base_ring.__prec
        self._entries = <mpc_t *> sig_malloc(sizeof(mpc_t) * degree)
        for i in range(degree):
            mpc_init2(self._entries[i],self._prec)
        if self._entries == NULL:
            raise MemoryError
        
    def __cinit__(self, parent, x, coerce=True,copy=True):
        cdef MPComplexNumber z
        self._entries = NULL
        self._is_mutable = 1
        # Allocate entries
        self._init(parent.degree(), parent)
        # Set entries
        if isinstance(x, (list, tuple)) or (hasattr(x,'parent') and x.parent() == parent) :
            if len(x) != self._degree:
                raise TypeError("entries must be a list of length %s"%self._degree)
            for i in range(self._degree):
                z = MPComplexNumber(self._base_ring,x[i])
                mpc_set(self._entries[i], z.value,rnd)
        elif x == 0 or x is None:
            for i in range(self._degree):
                mpc_set_ui(self._entries[i], 0, rnd)
        else:
            raise TypeError("can't initialize vector from nonzero non-list")

    def __init__(self, parent, entries, coerce = True, copy = True):
        pass


    def __dealloc__(self):
        cdef Py_ssize_t i
        if self._entries:
            sig_on()
            for i from 0 <= i < self._degree:
                mpc_clear(self._entries[i])
            sig_off()
            sig_free(self._entries)

    cpdef base_ring(self):
        return self._base_ring

    cdef int _cmp_c_impl(left, Element right) except -2:
        """
        EXAMPLES::
        
        sage: from psage.modules.vector_complex_dense import Vector_complex_dense
        sage: F=MPComplexField(53)
        sage: S=FreeModule(F,3)
        sage: v=Vector_complex_dense(S,0)
        sage: v == 0
        True
        sage: v == 1
        False
        sage: v == v
        True
        sage: w = Vector_complex_dense(S,[-1,3/2,0])
        sage: w < v
        True
        sage: w > v
        False
        """
        cdef Py_ssize_t i
        cdef int c
        for i from 0 <= i < left.degree():
            c = mpc_cmp(left._entries[i], (<Vector_complex_dense>right)._entries[i])
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
        cdef MPComplexNumber z
        if i < 0 or i >= self._degree:
            raise IndexError
        else:
            z = MPComplexNumber(self._base_ring,x)
            mpc_set(self._entries[i], z.value,rnd)
            
    def __getitem__(self, Py_ssize_t i):
        """
        Return the ith entry of self.

        EXAMPLES:
            sage: from psage.modules.vector_complex_dense import Vector_complex_dense
            sage: F=MPComplexField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1/2,2/3,3/4]); v
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
            IndexError:  index out of range
            sage: v[-5]
            Traceback (most recent call last):
            ...
            IndexError:  index out of range
        """
        cdef MPComplexNumber z
        z = MPComplexNumber(self._base_ring,0)
        if i < 0:
            i += self._degree
        if i < 0 or i >= self._degree:
            raise IndexError('index out of range')
        else:
            mpc_set(z.value, self._entries[i],rnd)
            return z

    def __reduce__(self):
        return (unpickle_v1, (self._parent, self.list(), self._degree, self._is_mutable))

    cpdef _add_(self, right):
        cdef Vector_complex_dense z
        cdef Vector_complex_dense r
        cdef MPComplexNumber ztmp
        cdef Py_ssize_t i
        if isinstance(right,type(self)):
            r = right
        else:
            r = Vector_complex_dense.__new__(Vector_complex_dense, self._parent, [self._base_ring(x) for x in right])
        z = self._new_c()
        for i from 0 <= i < self._degree:
            mpc_init2(z._entries[i],self._prec)
            mpc_add(z._entries[i], self._entries[i], r._entries[i],rnd)
        return z

    cpdef _sub_(self, right):
        cdef Vector_complex_dense z
        cdef Vector_complex_dense r 
        cdef MPComplexNumber ztmp
        cdef Py_ssize_t i
        if isinstance(right,type(self)):
            r = right
        else:
            r = Vector_complex_dense.__new__(Vector_complex_dense, self._parent, [self._base_ring(x) for x in right])
        z = self._new_c()
        for i from 0 <= i < self._degree:
            mpc_init2(z._entries[i],self._prec)
            mpc_sub(z._entries[i], self._entries[i], r._entries[i],rnd)
        return z
        
    cpdef Element _dot_product_(self, Vector right):
        """
        Dot (i.e. entry-wise) product of dense vectors over mpc.
        Note: this is different from the scalar product since it does not involve conjugation the second component.
        
        EXAMPLES::        
        
            sage: from psage.modules.vector_complex_dense import Vector_complex_dense
            sage: F=MPComplexField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1,2,3])
            sage: w=Vector_complex_dense(S,[4,5,6])
            sage: v.dot_product(w)
            32.0000000000000
            sage: w=Vector_complex_dense(S,[4,5,F(1+I)])
            sage: v.dot_product(w)
            17.0000000000000 + 3.00000000000000*I
        """
        cdef Vector_complex_dense r
        cdef MPComplexNumber z
        z = MPComplexNumber(self._base_ring,0)
        cdef mpc_t t
        if isinstance(right,type(self)):
            r = right
        else:
            r = Vector_complex_dense.__new__(Vector_complex_dense, self._parent, [self._base_ring(x) for x in right])

        mpc_init2(t,self._prec)
        mpc_set_si(z.value, 0,rnd)
        cdef Py_ssize_t i
        for i from 0 <= i < self._degree:
            mpc_mul(t, self._entries[i], r._entries[i],rnd)
            mpc_add(z.value, z.value, t,rnd)
        mpc_clear(t)
        return z
    
    def scalar_product(self,right):
        r"""

        EXAMPLES::

        sage: from psage.modules.vector_complex_dense import Vector_complex_dense
        sage: F=MPComplexField(53)
        sage: S=FreeModule(F,3)
        sage: v=Vector_complex_dense(S,[1,2,3])
        sage: w=Vector_complex_dense(S,[4,5,6])
        sage: v.scalar_product(w)
        32.0000000000000
        sage: w=Vector_complex_dense(S,[4,5,F(1+I)])
        sage: v.scalar_product(w)
        17.0000000000000 - 3.00000000000000*I
        """
        return self._scalar_product_(right)

    cdef MPComplexNumber _scalar_product_(self, Vector right):
        """
        Hermitian scalar product of dense vectors over mpc.

        Note: conjugation on the second component.
        

        """
        cdef Vector_complex_dense r = right
        cdef MPComplexNumber z
        z = MPComplexNumber(self._base_ring,0)
        cdef mpc_t t,x
        mpc_init2(t,self._prec)
        mpc_init2(x,self._prec)
        mpc_set_si(z.value, 0,rnd)
        cdef Py_ssize_t i
        for i from 0 <= i < self._degree:
            mpc_conj(x,r._entries[i],rnd)
            mpc_mul(t, self._entries[i], x,rnd)
            mpc_add(z.value, z.value, t,rnd)
        mpc_clear(t)
        mpc_clear(x)
        return z
        

    cpdef Vector _pairwise_product_(self, Vector right):
        """
        
        EXAMPLES::
        
        sage: from psage.modules.vector_complex_dense import Vector_complex_dense
        sage: F=MPComplexField(53)
        sage: S=FreeModule(F,3)
        sage: v=Vector_complex_dense(S,[1,2,3])
        sage: w=Vector_complex_dense(S,[4,5,6])
        sage: v.pairwise_product(w)
        (4.00000000000000, 10.0000000000000, 18.0000000000000)
        """
        cdef Vector_complex_dense z, r
        r = right
        z = self._new_c()
        cdef Py_ssize_t i
        for i from 0 <= i < self._degree:
            mpc_init2(z._entries[i],self._prec)
            mpc_mul(z._entries[i], self._entries[i], r._entries[i],rnd)
        return z

        
    def __mul__(self, ModuleElement right):
        """

        EXAMPLES::

            sage: from psage.modules.vector_complex_dense import Vector_complex_dense
            sage: F=MPComplexField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1,2,3])
            sage: 2*v
            (2.00000000000000, 4.00000000000000, 6.00000000000000)
            sage: v*2
            (2.00000000000000, 4.00000000000000, 6.00000000000000)
            sage: v*F(1+I)
            (1.00000000000000 + 1.00000000000000*I, 2.00000000000000 + 2.00000000000000*I, 3.00000000000000 + 3.00000000000000*I)

        """
        cdef MPComplexNumber a
        if isinstance(right,Vector_complex_dense):
            return self._dot_product_(right)
        if isinstance(right,Matrix_complex_dense):
            return Matrix_complex_dense._vector_times_matrix_(right,self)
        return self._lmul_(right)
        
    cpdef _rmul_(self, Element left):
        cdef Vector_complex_dense z
        cdef Py_ssize_t i
        cdef MPComplexNumber a
        z = self._new_c()
        # we can convert almost anything to MPComplexNumber
        if not isinstance(left,MPComplexNumber):
            a = MPComplexNumber(self._base_ring,left)
            for i from 0 <= i < self._degree:
                mpc_init2(z._entries[i],self._prec)
                mpc_mul(z._entries[i], self._entries[i], a.value,rnd)
        else:
            for i from 0 <= i < self._degree:
                mpc_init2(z._entries[i],self._prec)
                mpc_mul(z._entries[i], self._entries[i], (<MPComplexNumber>(left)).value,rnd)
        return z


    cpdef _lmul_(self, Element right):
        cdef Vector_complex_dense z
        cdef MPComplexNumber a
        # we can convert almost anything to MPComplexNumber
        if not isinstance(right,MPComplexNumber):
             a = MPComplexNumber(self._base_ring,right)
        else:
            a = right
        z = self._new_c()
        cdef Py_ssize_t i
        for i from 0 <= i < self._degree:
            mpc_init2(z._entries[i],self._prec)
            mpc_mul(z._entries[i], self._entries[i], a.value,rnd)
        return z

    cpdef ModuleElement _neg_(self):
        r"""
        
        EXAMPLES::
        
            sage: from psage.modules.vector_complex_dense import Vector_complex_dense
            sage: F=MPComplexField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1,2,3])
            sage: -v
            (-1.00000000000000, -2.00000000000000, -3.00000000000000)           
        
        """
        cdef Vector_complex_dense z
        z = self._new_c()
        cdef Py_ssize_t i
        for i from 0 <= i < self._degree:
            mpc_init2(z._entries[i],self._prec)
            mpc_neg(z._entries[i], self._entries[i],rnd)
        return z


    cpdef RealNumber norm(self,int ntype=2):
        r"""
        The norm of self.
        
        EXAMPLES::
        
            sage: from psage.modules.vector_complex_dense import Vector_complex_dense
            sage: F=MPComplexField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1,2,3])
            sage: v.norm()
            3.74165738677394
            sage: v=Vector_complex_dense(S,[1+I,2+I,3+I])
            sage: v.norm()
            4.12310562561766
            
        """
        cdef RealNumber res
        cdef mpfr_t x,s
        mpfr_init2(x,self._prec)
        mpfr_init2(s,self._prec)
        res = RealNumber(self._base_ring._base,0)
        if ntype==2:
            mpfr_set_si(s, 0, MPFR_RNDU)
            for i from 0 <= i < self._degree:
                mpc_abs(x,self._entries[i],MPFR_RNDU)
                mpfr_sqr(x,x,MPFR_RNDU)
                mpfr_add(s,s,x,MPFR_RNDU)
            mpfr_sqrt(s,s,MPFR_RNDU)
            mpfr_set(res.value,s,MPFR_RNDU)
            mpfr_clear(x); mpfr_clear(s)
            return res
        else:
            mpfr_clear(x); mpfr_clear(s)
            raise NotImplementedError,"Only 2-norm is currently implemented"

    def prec(self):
        r"""

        EXAMPLES::

            sage: from psage.modules.vector_complex_dense import Vector_complex_dense
            sage: F=MPComplexField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1+I,2+I,3+I]); v
            (1.00000000000000 + 1.00000000000000*I, 2.00000000000000 + 1.00000000000000*I, 3.00000000000000 + 1.00000000000000*I)
            sage: v.prec()
            53
            sage: F=MPComplexField(103)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1+I,2+I,3+I]); v
            (1.00000000000000000000000000000 + 1.00000000000000000000000000000*I, 2.00000000000000000000000000000 + 1.00000000000000000000000000000*I, 3.00000000000000000000000000000 + 1.00000000000000000000000000000*I)
            sage: v.prec()
            103

        """
        return self._prec

    def set_prec(self,int prec):
        r"""
        Change the default precision for self.

        EXAMPLES::

            sage: from psage.modules.vector_complex_dense import Vector_complex_dense
            sage: F=MPComplexField(53)
            sage: S=FreeModule(F,3)
            sage: v=Vector_complex_dense(S,[1+I,2+I,3+I]); v
            (1.00000000000000 + 1.00000000000000*I, 2.00000000000000 + 1.00000000000000*I, 3.00000000000000 + 1.00000000000000*I)
            sage: v.prec()
            53
            sage: v.set_prec(103); v
            (1.00000000000000000000000000000 + 1.00000000000000000000000000000*I, 2.00000000000000000000000000000 + 1.00000000000000000000000000000*I, 3.00000000000000000000000000000 + 1.00000000000000000000000000000*I)
            sage: v.prec()
            103
        """        
        cdef Py_ssize_t i,j
        cdef mpc_t z
        from sage.rings.complex_mpc import MPComplexField
        from sage.matrix.all import MatrixSpace
        mpc_init2(z,prec)
        self._prec = prec
        self._base_ring=MPComplexField(prec)
        self._parent = FreeModule(self._base_ring,self._degree)
        #self._parent = FreeModule(self._base_ring,self._nrows,self._ncols)
        for i from 0 <= i < self._degree:
                mpc_set(z,self._entries[i],rnd)
                mpc_set_prec(self._entries[i],prec)
                mpc_set(self._entries[i],z,rnd)
        mpc_clear(z)


@cython.binding(True)
def make_FreeModuleElement_complex_dense_v1(parent, entries, degree):
    r"""
    If you think you want to change this function, don't.
    Instead make a new version with a name like
       make_FreeModuleElement_generic_dense_v1
    and changed the reduce method below.

    EXAMPLES::

    sage: F = MPComplexField(53)
    sage: S = FreeModule(F,3)
    sage: from psage.modules.vector_complex_dense import make_FreeModuleElement_complex_dense_v1
    sage: entries = [F(1),F(2),F(3)]
    sage: v = make_FreeModuleElement_complex_dense_v1(S,entries,3); v
    (1.00000000000000, 2.00000000000000, 3.00000000000000)
    """
    cdef Vector_complex_dense v
    v = Vector_complex_dense.__new__(Vector_complex_dense,parent,entries)
    return v

def unpickle_v1(parent, entries, degree, is_mutable):
    cdef Vector_complex_dense v
    v = Vector_complex_dense.__new__(Vector_complex_dense,parent,entries)
    return v
    # v._init(degree, parent)
    # prec=parent.prec()
    # cdef MPComplexNumber z
    # base_ring=parent.base_ring()
    # for i from 0 <= i < degree:
    #     z = MPComplexNumber(base_ring,entries[i])
    #     mpc_init2(v._entries[i],prec)
    #     mpc_set(v._entries[i], z.value,rnd)
    # v._is_mutable = is_mutable
    # return v

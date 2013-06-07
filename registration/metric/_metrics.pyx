# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import cython
from cython.parallel import parallel, prange

from libc.math cimport exp, sqrt, M_PI
from libc.stdlib cimport malloc, free

import numpy
cimport numpy
cimport openmp

numpy.import_array()

ctypedef numpy.float64_t DTYPE_t

cdef extern DTYPE_t dnrm2_(int* N, DTYPE_t* X, int* incX) nogil
cdef extern DTYPE_t ddot_(int* N, DTYPE_t* DX, int* incX, DTYPE_t* DY, int* incY) nogil
cdef extern void dposv_(char* uplo, int* N, int* NRHS, DTYPE_t* A, int* LDA, DTYPE_t* B, int* LDB, int* INFO) nogil
cdef extern void dpotrf_(char* uplo, int* N, DTYPE_t* A, int* LDA, int* INFO ) nogil
cdef extern void dsysv_(char* uplo, int* N, int* NRHS, DTYPE_t* A, int* LDA, int* IPIV, DTYPE_t* B, int* LDB, double* work, int* lwork, int* INFO) nogil

cdef extern from "omp.h":
    ctypedef struct omp_lock_t:
        pass



@cython.profile(False)
cdef DTYPE_t norm(numpy.ndarray x) nogil:
    cdef int n    = x.shape[0]
    cdef int incX = x.strides[0] / sizeof(DTYPE_t)

    return dnrm2_(&n, <DTYPE_t*> x.data ,&incX)

@cython.profile(False)
cdef DTYPE_t dot(numpy.ndarray x, numpy.ndarray y) nogil:
    cdef int n    = x.shape[0]
    cdef int incX = x.strides[0] / sizeof(DTYPE_t)
    cdef int incY = y.strides[0] / sizeof(DTYPE_t)

    return ddot_(&n, <DTYPE_t *>x.data, &incX, <DTYPE_t *>y.data, &incY)

@cython.profile(False)
cdef DTYPE_t dot_plain(DTYPE_t* x, DTYPE_t* y, int n) nogil:
    cdef int incX = 1
    cdef int incY = 1

    return ddot_(&n, x, &incX, y, &incY)

@cython.profile(False)
cdef int posv(numpy.ndarray a, numpy.ndarray b, char uplo='U') nogil:
    cdef int n = a.shape[0]
    cdef int nrhs = 1
    cdef int info=0
    cdef int lda=n
    cdef int ldb=n
    dposv_(&uplo, &n, &nrhs, <DTYPE_t*> a.data, &lda, <DTYPE_t*> b.data, &ldb, &info)

    return info

@cython.profile(False)
cdef int posv_plain(DTYPE_t* a, DTYPE_t* b, int n, char uplo='U') nogil:
    cdef int nrhs = 1
    cdef int info=0
    cdef int lda=n
    cdef int ldb=n
    dposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info)

    return info

@cython.profile(False)
cdef int sysv(numpy.ndarray a, numpy.ndarray b, char uplo='U') nogil:
    cdef int n = a.shape[0]
    cdef int* ipiv = <int*> malloc(sizeof(int) * n)
    cdef int lwork = n * n
    cdef double* work = <double*> malloc(sizeof(double) * n)
    cdef int nrhs = 1
    cdef int info=0
    cdef int lda=n
    cdef int ldb=n
    dsysv_(&uplo, &n, &nrhs, <DTYPE_t*> a.data, &lda, ipiv, <DTYPE_t*> b.data, &ldb, work, &lwork, &info)

    free(ipiv)
    free(work)

    return info

def gauss_transform(
    numpy.ndarray[DTYPE_t, ndim=2, mode='c'] f not None, numpy.ndarray[DTYPE_t, ndim=2, mode='c'] g not None,
    float scale
):
    """Compute the inner product and evaluation between two gaussian mixtures at a certain scale.
    The inner product is defined as:
    :math: `<f, g> = \sum_i \sum_j \int exp( -((f_i - x)^2 + (g_i -x)^2)/scale^2) dx`

    Parameters
    ----------
        f: points of the gaussian mixture
        g: points of the gaussian mixture
        scale: scale of the inner product

    Returns
    -------
        inner_product: The L2 inner product between the two Gaussian mixtures
        value_ff: The value of the first gaussian mixture at every point of f
        value_fg: The value of the first gaussian mixture at every point of g
        grad_f: The gradient of the inner product at every point of f
    """
    cdef numpy.npy_intp n = f.shape[0]
    cdef numpy.npy_intp m = g.shape[0]
    cdef numpy.npy_intp dim = f.shape[1]

    cdef omp_lock_t lock

    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] grad = numpy.zeros_like(f)
    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] valueFF = numpy.zeros((n, 1))
    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] valueFG = numpy.zeros((m, 1))

    cdef numpy.npy_intp i, j, d
    cdef DTYPE_t inner_product = 0
    cdef DTYPE_t scale_sqr = scale ** 2

    cdef DTYPE_t* dist_ij = NULL
    cdef DTYPE_t cost_ij

    openmp.omp_init_lock(&lock)

    for i in prange(n, nogil=True):
        for j in range(m):
            dist_ij = <DTYPE_t*> malloc(sizeof(DTYPE_t))
            dist_ij[0] = 0
            for d in range(dim):
                dist_ij[0] += (f[i, d] - g[j, d]) ** 2
            cost_ij = exp(-dist_ij[0] / (2 * scale_sqr))
            valueFF[i, 0] += cost_ij

            openmp.omp_set_lock(&lock)
            inner_product += cost_ij
            valueFG[j, 0] += cost_ij
            openmp.omp_unset_lock(&lock)

            for d in range(dim):
                grad[i, d] += -cost_ij * (f[i, d] - g[j, d]) / scale_sqr
        free(dist_ij)

    openmp.omp_destroy_lock(&lock)

    #for i in range(n):
    #    valueFF[i, 0] /= n
    #    for d in range(dim):
    #        grad[i, d] *= scale_sqr # n * m * scale_sqr * 2

    #for j in range(m):
    #    valueFG[j, 0] /= m

    #inner_product /= n * m
    return inner_product, valueFF, valueFG, grad


def gauss_process_transform(
    numpy.ndarray[DTYPE_t, ndim=2, mode='c'] f not None, numpy.ndarray[DTYPE_t, ndim=2, mode='c'] g not None,
    numpy.ndarray[DTYPE_t, ndim=1, mode='c'] f_weights not None, numpy.ndarray[DTYPE_t, ndim=1, mode='c'] g_weights not None,
    float scale
):
    """Compute the inner product and evaluation between two gaussian mixtures at a certain scale.
    The inner product is defined as:
    :math: `<f, g> = \sum_i \sum_j \int f\_weights_i exp(-(f_i - x)^2 / scale^2) g\_weights_i exp(-(g_i - x)^2 / scale^2)dx`

    Parameters
    ----------
        f: points of the gaussian mixture
        g: points of the gaussian mixture
        scale: scale of the inner product

    Returns
    -------
        inner_product: The L2 inner product between the two Gaussian mixtures
        value_ff: The value of the first gaussian mixture at every point of f
        value_fg: The value of the first gaussian mixture at every point of g
        grad_f: The gradient of the inner product at every point of f
    """
    cdef numpy.npy_intp n = f.shape[0]
    cdef numpy.npy_intp m = g.shape[0]
    cdef numpy.npy_intp dim = f.shape[1]

    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] grad = numpy.zeros_like(f)
    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] valueFF = numpy.zeros((n, 1))
    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] valueFG = numpy.zeros((m, 1))

    cdef numpy.npy_intp i, j, d
    cdef DTYPE_t inner_product = 0
    cdef DTYPE_t scale_sqr = scale ** 2

    cdef DTYPE_t dist_ij
    cdef DTYPE_t cost_ij

    for i in range(n):
        for j in range(m):
            dist_ij = 0
            for d in range(dim):
                dist_ij += (f[i, d] - g[j, d]) ** 2
            cost_ij = exp(-dist_ij / scale_sqr) * f_weights[i] * g_weights[j]
            inner_product += cost_ij
            valueFF[i, 0] += cost_ij
            valueFG[j, 0] += cost_ij

            for d in range(dim):
                grad[i, d] += -cost_ij * 2 * (f[i, d] - g[j, d])

    for i in range(n):
        valueFF[i, 0] /= n
        for d in range(dim):
            grad[i, d] /= n * m * scale_sqr

    for j in range(m):
        valueFG[j, 0] /= m

    inner_product /= n * m

    return inner_product, valueFF, valueFG, grad


def gauss_process_w_covar_transform(
    numpy.ndarray[DTYPE_t, ndim=2, mode='c'] f not None, numpy.ndarray[DTYPE_t, ndim=2, mode='c'] g not None,
    numpy.ndarray[DTYPE_t, ndim=3, mode='c'] f_covar not None, numpy.ndarray[DTYPE_t, ndim=3, mode='c'] g_covar not None,
    numpy.ndarray[DTYPE_t, ndim=1, mode='c'] f_weights=None, numpy.ndarray[DTYPE_t, ndim=1, mode='c'] g_weights=None,
    float scale=0
):
    """Compute the inner product and evaluation between two gaussian mixtures at a certain scale.
    The inner product is defined as:
    :math: `<f, g> = \sum_i \sum_j \int f\_weights_i exp(-(f_i - x) C_f^{-1} (f_i - x)) g\_weights_i exp(-(g_i - x) C_g^{-1} (g_i - x))dx`
    where
    :math: `\int  exp(-(f_i - x) C_f^{-1} (f_i - x)) g\_weights_i exp(-(g_i - x) C_g^{-1} (g_i -x)) = exp(-(f_i - g_i)(C_{f_i}+C_{g_i})^{-1}(f_i - g_i))

    Parameters
    ----------
        f: points of the gaussian mixture
        g: points of the gaussian mixture
        f_covar: covariances of the f points
        g_covar: covariances of the g points
        scale: scale of the inner product

    Returns
    -------
        inner_product: The L2 inner product between the two Gaussian mixtures
        value_ff: The value of the first gaussian mixture at every point of f
        value_fg: The value of the first gaussian mixture at every point of g
        grad_f: The gradient of the inner product at every point of f
    """
    cdef numpy.npy_intp n = f.shape[0]
    cdef numpy.npy_intp m = g.shape[0]
    cdef numpy.npy_intp dim = f.shape[1]

    cdef omp_lock_t lock

    cdef DTYPE_t* aux_covar = NULL
    cdef DTYPE_t* aux_diff_vector = NULL
    cdef DTYPE_t* aux_vector = NULL

    if f_weights is None:
        f_weights = numpy.ones(n)
    if g_weights is None:
        g_weights = numpy.ones(m)

    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] grad = numpy.zeros_like(f)
    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] valueFF = numpy.zeros((n, 1))
    cdef numpy.ndarray[DTYPE_t, ndim=2, mode='c'] valueFG = numpy.zeros((m, 1))

    cdef DTYPE_t inner_product = 0
    cdef DTYPE_t scale_sqr = 2 * scale ** 2

    cdef DTYPE_t dist_ij
    cdef DTYPE_t cost_ij

    cdef int info = 0
    cdef int nrhs = 1

    cdef int i = 0
    cdef int j = 0
    cdef int d = 0
    cdef int e = 0

    openmp.omp_init_lock(&lock)

    for i in prange(n, nogil=True):
        aux_covar = <DTYPE_t*> malloc(sizeof(DTYPE_t) * dim * dim)
        aux_diff_vector = <DTYPE_t*> malloc(sizeof(DTYPE_t) * dim)
        aux_vector =  <DTYPE_t*> malloc(sizeof(DTYPE_t) * dim)

        for j in range(m):

            for d in range(dim):
                aux_diff_vector[d] = f[i, d] - g[j, d]
                aux_vector[d] = aux_diff_vector[d]
                for e in range(d, dim):
                    aux_covar[d + e * dim] = f_covar[i, d, e] + g_covar[j, d, e]
                aux_covar[d + d * dim] += scale_sqr

            info = posv_plain(aux_covar, aux_diff_vector, dim)

            if info != 0:
                break

            dist_ij = dot_plain(aux_diff_vector, aux_vector, dim)
            for d in range(dim):
                aux_vector[d] *= aux_diff_vector[d]

            cost_ij = exp(-dist_ij) * f_weights[i] * g_weights[j]
            valueFF[i, 0] += cost_ij

            openmp.omp_set_lock(&lock)
            inner_product += cost_ij
            valueFG[j, 0] += cost_ij
            openmp.omp_unset_lock(&lock)

            for d in range(dim):
                grad[i, d] += -cost_ij * 2 * aux_vector[d]

        free(aux_covar)
        free(aux_diff_vector)
        free(aux_vector)

        if info != 0:
            break

    openmp.omp_destroy_lock(&lock)
    if info != 0:
        #with gil:
        raise ValueError("Error in inverting covariances f: %d g: %d. Code: %d" % (i, j, info))


    for i in range(n):
        valueFF[i, 0] /= n
        for d in range(dim):
            grad[i, d] /= n * m

    for j in range(m):
        valueFG[j, 0] /= m

    inner_product /= n * m

    return inner_product, valueFF, valueFG, grad

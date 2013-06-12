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

cdef extern from "omp.h":
    ctypedef struct omp_lock_t:
        pass


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

        dist_ij = <DTYPE_t*> malloc(sizeof(DTYPE_t))
        for j in range(m):
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


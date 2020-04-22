# Copyright (c) 2020 Dimitrios-Georgios Akestoridis
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


def dcor(z_matrix):
    """Python implementation of the dCor correlation method.

    For more information, see the following publications:
      * G. J. Székely, M. L. Rizzo, and N. K. Bakirov, "Measuring and testing
        dependence by correlation of distances," The Annals of Statistics,
        vol. 35, no. 6, pp. 2769--2794, 2007. DOI: 10.1214/009053607000000505.
      * G. J. Székely and M. L. Rizzo, "Brownian distance covariance,"
        The Annals of Applied Statistics, vol. 3, no. 4, pp. 1236--1265, 2009.
        DOI: 10.1214/09-AOAS312.
    """
    # Make sure that the matrix is a float64 NumPy array
    if type(z_matrix) is not np.ndarray:
        z_matrix = np.array(z_matrix, dtype=np.float64)
    elif z_matrix.dtype is not np.float64:
        z_matrix = np.array(z_matrix, dtype=np.float64)

    # Initialize the matrix for the distance correlation coefficients
    dcor_matrix = np.ones((z_matrix.shape[1], z_matrix.shape[1]),
                          dtype=np.float64)

    # Compute the matrix of squared distance covariances
    dcov2_matrix = squared_dcov_matrix(z_matrix)

    # Compute the distance correlation coefficients
    for j in range(z_matrix.shape[1]):
        # Get the squared distance variance of the j-th criterion
        j_dvar2 = dcov2_matrix[j, j]

        for l in range(j + 1, z_matrix.shape[1]):
            # Get the squared distance variance of the l-th criterion
            l_dvar2 = dcov2_matrix[l, l]

            # Compare the product of their squared distance variances
            if j_dvar2 * l_dvar2 == 0.0:
                # The two criteria are independent
                dcor_matrix[j, l] = 0.0
                dcor_matrix[l, j] = 0.0
            else:
                # Get the squared distance covariance of the two criteria
                jl_dcov2 = dcov2_matrix[j, l]

                # Compute the squared distance correlation of the two criteria
                jl_dcor2 = squared_dcor(jl_dcov2, j_dvar2, l_dvar2)

                # Compute the distance correlation of the two criteria
                dcor_matrix[j, l] = np.sqrt(jl_dcor2)
                dcor_matrix[l, j] = dcor_matrix[j, l]

    return dcor_matrix


def squared_dcov_matrix(z_matrix):
    # Initialize the distance covariance matrix
    dcov2_matrix = np.zeros((z_matrix.shape[1], z_matrix.shape[1]),
                            dtype=np.float64)

    for j in range(z_matrix.shape[1]):
        # Compute the Euclidean distance matrix of the j-th criterion
        j_dmatrix = dist_matrix(z_matrix[:, j])

        # Compute the linear function of its Euclidean distance matrix
        j_func = lin_func(j_dmatrix)

        for l in range(j, z_matrix.shape[1]):
            if j == l:
                # Compute the distance variance of the j-th criterion
                dcov2_matrix[j, j] = squared_dcov(j_func, j_func)
            else:
                # Compute the Euclidean distance matrix of the l-th criterion
                l_dmatrix = dist_matrix(z_matrix[:, l])

                # Compute the linear function of its Euclidean distance matrix
                l_func = lin_func(l_dmatrix)

                # Compute the squared distance covariance of the two criteria
                dcov2_matrix[j, l] = squared_dcov(j_func, l_func)
                dcov2_matrix[l, j] = dcov2_matrix[j, l]

    return dcov2_matrix


def dist_matrix(z_vector):
    # Initialize the Euclidean distance matrix
    dmatrix = np.zeros((z_vector.shape[0], z_vector.shape[0]),
                       dtype=np.float64)

    for i in range(z_vector.shape[0]):
        for k in range(i + 1, z_vector.shape[0]):
            # The Euclidean distance of two real-valued scalars corresponds
            # to the absolute value of their difference
            dmatrix[i, k] = np.fabs(z_vector[i] - z_vector[k])
            dmatrix[k, i] = dmatrix[i, k]

    return dmatrix


def lin_func(dmatrix):
    return (dmatrix
            - np.mean(dmatrix, axis=0)
            - np.reshape(np.mean(dmatrix, axis=1), (dmatrix.shape[0], 1))
            + np.mean(dmatrix))


def squared_dcov(j_func, l_func):
    return np.sum(np.multiply(j_func, l_func)) / (j_func.shape[0] ** 2)


def squared_dcor(jl_dcov2, j_dvar2, l_dvar2):
    return jl_dcov2 / np.sqrt(j_dvar2 * l_dvar2)

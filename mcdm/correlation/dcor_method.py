# Copyright (c) 2020-2021 Dimitrios-Georgios Akestoridis
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Python implementation of the dCor correlation method. For more information,
see the following publications:

* G. J. Székely, M. L. Rizzo, and N. K. Bakirov, “Measuring and testing
  dependence by correlation of distances,” *Ann. Statist.*, vol. 35, no. 6,
  pp. 2769–2794, 2007, doi: `10.1214/009053607000000505
  <https://doi.org/10.1214/009053607000000505>`_.

* G. J. Székely and M. L. Rizzo, “Brownian distance covariance,” *Ann. Appl.
  Statist.*, vol. 3, no. 4, pp. 1236–1265, 2009, doi: `10.1214/09-AOAS312
  <https://doi.org/10.1214/09-AOAS312>`_.

"""

import numpy as np


def dcor(z_matrix):
    """
    Return the distance correlation coefficients of the provided matrix.
    """
    # Make sure that the provided matrix is a float64 NumPy array
    z_matrix = np.array(z_matrix, dtype=np.float64)

    # Initialize the matrix for the distance correlation coefficients
    dcor_matrix = np.ones(
        (z_matrix.shape[1], z_matrix.shape[1]),
        dtype=np.float64,
    )

    # Compute the matrix of squared distance covariances
    dcov2_matrix = squared_dcov_matrix(z_matrix)

    # Compute the distance correlation coefficients
    for j_col in range(z_matrix.shape[1]):
        # Get the squared distance variance of the j-th criterion
        j_dvar2 = dcov2_matrix[j_col, j_col]

        for l_col in range(j_col + 1, z_matrix.shape[1]):
            # Get the squared distance variance of the l-th criterion
            l_dvar2 = dcov2_matrix[l_col, l_col]

            # Compare the product of their squared distance variances
            if j_dvar2 * l_dvar2 == 0.0:
                # The two criteria are independent
                dcor_matrix[j_col, l_col] = 0.0
                dcor_matrix[l_col, j_col] = 0.0
            else:
                # Get the squared distance covariance of the two criteria
                jl_dcov2 = dcov2_matrix[j_col, l_col]

                # Compute the squared distance correlation of the two criteria
                jl_dcor2 = squared_dcor(jl_dcov2, j_dvar2, l_dvar2)

                # Compute the distance correlation of the two criteria
                dcor_matrix[j_col, l_col] = np.sqrt(jl_dcor2)
                dcor_matrix[l_col, j_col] = dcor_matrix[j_col, l_col]

    return dcor_matrix


def squared_dcov_matrix(z_matrix):
    """
    Return the matrix of squared distance covariance between the columns of
    the provided matrix.
    """
    # Initialize the distance covariance matrix
    dcov2_matrix = np.zeros(
        (z_matrix.shape[1], z_matrix.shape[1]),
        dtype=np.float64,
    )

    for j_col in range(z_matrix.shape[1]):
        # Compute the Euclidean distance matrix of the j-th criterion
        j_dmatrix = dist_matrix(z_matrix[:, j_col])

        # Compute the linear function of its Euclidean distance matrix
        j_func = lin_func(j_dmatrix)

        for l_col in range(j_col, z_matrix.shape[1]):
            if j_col == l_col:
                # Compute the distance variance of the j-th criterion
                dcov2_matrix[j_col, j_col] = squared_dcov(j_func, j_func)
            else:
                # Compute the Euclidean distance matrix of the l-th criterion
                l_dmatrix = dist_matrix(z_matrix[:, l_col])

                # Compute the linear function of its Euclidean distance matrix
                l_func = lin_func(l_dmatrix)

                # Compute the squared distance covariance of the two criteria
                dcov2_matrix[j_col, l_col] = squared_dcov(j_func, l_func)
                dcov2_matrix[l_col, j_col] = dcov2_matrix[j_col, l_col]

    return dcov2_matrix


def dist_matrix(z_vector):
    """
    Return the Euclidean distance matrix of the provided vector.
    """
    # Initialize the Euclidean distance matrix
    dmatrix = np.zeros(
        (z_vector.shape[0], z_vector.shape[0]),
        dtype=np.float64,
    )

    for i_row in range(z_vector.shape[0]):
        for k_row in range(i_row + 1, z_vector.shape[0]):
            # The Euclidean distance of two real-valued scalars corresponds
            # to the absolute value of their difference
            dmatrix[i_row, k_row] = np.fabs(z_vector[i_row] - z_vector[k_row])
            dmatrix[k_row, i_row] = dmatrix[i_row, k_row]

    return dmatrix


def lin_func(dmatrix):
    """
    Return the result of the linear function for the provided distance matrix.
    """
    return (
        dmatrix
        - np.mean(dmatrix, axis=0)
        - np.reshape(np.mean(dmatrix, axis=1), (dmatrix.shape[0], 1))
        + np.mean(dmatrix)
    )


def squared_dcov(j_func, l_func):
    """
    Return the squared distance covariance between the corresponding columns.
    """
    return np.sum(np.multiply(j_func, l_func)) / (j_func.shape[0] ** 2)


def squared_dcor(jl_dcov2, j_dvar2, l_dvar2):
    """
    Return the squared distance correlation between the corresponding columns.
    """
    return jl_dcov2 / np.sqrt(j_dvar2 * l_dvar2)

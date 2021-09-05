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
Python implementation of the Linear1 normalization method. For more
information, see the following publications:

* C.-L. Hwang and K. Yoon, *Multiple Attribute Decision Making*, ser. Lecture
  Notes in Economics and Mathematical Systems. Springer-Verlag Berlin
  Heidelberg, 1981, vol. 186, isbn: 9783540105589.

* H.-S. Shih, H.-J. Shyur, and E. S. Lee, “An extension of TOPSIS for group
  decision making,” *Math. Comput. Model.*, vol. 45, no. 7–8, pp. 801–813,
  2007, doi: `10.1016/j.mcm.2006.03.023
  <https://doi.org/10.1016/j.mcm.2006.03.023>`_.

"""

import numpy as np


def linear1(x_matrix, is_benefit_x):
    """
    Return the normalized version of the provided matrix using the Linear
    Normalization (1) method.
    """
    # Make sure that the provided matrix is a float64 NumPy array
    x_matrix = np.array(x_matrix, dtype=np.float64)

    # Sanity check
    if len(is_benefit_x) != x_matrix.shape[1]:
        raise ValueError(
            "The number of variables in the list that determines whether "
            + "each criterion is a benefit or a cost criterion does not "
            + "match the number of columns in the provided matrix",
        )

    # Make sure that it does not contain any negative numbers
    if np.sum(np.less(x_matrix, 0.0)) > 0:
        raise ValueError(
            "The provided matrix must not contain any negative numbers in "
            + "order to apply the Linear1 normalization method",
        )

    # Construct the normalized matrix
    z_matrix = np.zeros(x_matrix.shape, dtype=np.float64)
    for j in range(x_matrix.shape[1]):
        if is_benefit_x[j]:
            max_value = np.amax(x_matrix[:, j])
            if max_value == 0.0:
                raise ValueError(
                    "The maximum value of a benefit criterion must not be "
                    + "zero in order to apply the Linear1 normalization "
                    + "method",
                )
            z_matrix[:, j] = x_matrix[:, j] / max_value
        else:
            min_value = np.amin(x_matrix[:, j])
            if min_value == 0.0:
                raise ValueError(
                    "The minimum value of a cost criterion must not be zero "
                    + "in order to apply the Linear1 normalization method",
                )
            z_matrix[:, j] = min_value / x_matrix[:, j]

    # All criteria have been transformed into benefit criteria
    is_benefit_z = [True for _ in range(x_matrix.shape[1])]

    return z_matrix, is_benefit_z

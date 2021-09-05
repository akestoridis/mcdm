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
Python implementation of the Vector normalization method. For more
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

from ..helper_validation import check_normalization_input


def vector(x_matrix, is_benefit_x):
    """
    Return the normalized version of the provided matrix using the Vector
    Normalization method.
    """
    # Perform sanity checks
    x_matrix = np.array(x_matrix, dtype=np.float64)
    check_normalization_input(x_matrix, is_benefit_x, "Vector")

    # Construct the normalized matrix
    z_matrix = np.zeros(x_matrix.shape, dtype=np.float64)
    for j in range(x_matrix.shape[1]):
        denominator = np.sqrt(np.sum(x_matrix[:, j] ** 2))
        if denominator == 0.0:
            raise ValueError(
                "The square root of a criterion's sum of squared values must "
                + "not be equal to zero in order to apply the Vector "
                + "normalization method",
            )
        z_matrix[:, j] = x_matrix[:, j] / denominator

    # The criteria have not been transformed into benefit or cost criteria
    is_benefit_z = is_benefit_x.copy()

    return z_matrix, is_benefit_z

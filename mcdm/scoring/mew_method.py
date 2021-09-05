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
Python implementation of the MEW scoring method. For more information, see the
following publication:

* S. H. Zanakis, A. Solomon, N. Wishart, and S. Dublish, “Multi-attribute
  decision making: A simulation comparison of select methods,” *Eur. J. Oper.
  Res.*, vol. 107, no. 3, pp. 507–529, 1998, doi:
  `10.1016/S0377-2217(97)00147-1
  <https://doi.org/10.1016/S0377-2217(97)00147-1>`_.

"""

import numpy as np

from ..helper_validation import check_scoring_input


def mew(z_matrix, w_vector, is_benefit_z):
    """
    Return the Multiplicative Exponential Weighting scores of the provided
    decision matrix with the provided weight vector.
    """
    # Perform sanity checks
    z_matrix = np.array(z_matrix, dtype=np.float64)
    w_vector = np.array(w_vector, dtype=np.float64)
    check_scoring_input(z_matrix, w_vector, is_benefit_z, "MEW")

    # Determine whether the scores should be sorted in descending order
    if sum(is_benefit_z) == len(is_benefit_z):
        desc_order = True
    elif sum(is_benefit_z) == 0:
        desc_order = False
    else:
        raise ValueError(
            "All criteria must be either benefit or cost criteria in order "
            + "to use the MEW method",
        )

    # Compute the score of each alternative
    s_vector = np.ones(z_matrix.shape[0], dtype=np.float64)
    for i in range(z_matrix.shape[0]):
        for j in range(z_matrix.shape[1]):
            s_vector[i] *= z_matrix[i, j] ** w_vector[j]

    return s_vector, desc_order

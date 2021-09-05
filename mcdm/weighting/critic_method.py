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
Python implementation of the CRITIC weighting method. For more information,
see the following publication:

* D. Diakoulaki, G. Mavrotas, and L. Papayannakis, “Determining objective
  weights in multiple criteria problems: The CRITIC method,” *Comput. Oper.
  Res.*, vol. 22, no. 7, pp. 763–770, 1995, doi: `10.1016/0305-0548(94)00059-H
  <https://doi.org/10.1016/0305-0548(94)00059-H>`_.

"""

import numpy as np

from ..helper_correlation import correlate
from ..helper_validation import check_weighting_input


def critic(z_matrix, c_method="Pearson"):
    """
    Return the weight vector of the provided decision matrix using the
    Criteria Importance Through Intercriteria Correlation method.
    """
    # Perform sanity checks
    z_matrix = np.array(z_matrix, dtype=np.float64)
    if c_method is None:
        c_method = "Pearson"
    check_weighting_input(z_matrix, c_method, "CRITIC")

    # Compute the standard deviation of each criterion
    sd_vector = np.std(z_matrix, axis=0, dtype=np.float64)

    # Compute the correlation coefficients between pairs of criteria
    corr_matrix = correlate(z_matrix, c_method)

    # Compute the importance of each criterion
    imp_vector = np.zeros(z_matrix.shape[1], dtype=np.float64)
    for j_col in range(z_matrix.shape[1]):
        tmp_sum = 0.0
        for l_col in range(z_matrix.shape[1]):
            tmp_sum += 1.0 - corr_matrix[j_col, l_col]
        imp_vector[j_col] = sd_vector[j_col] * tmp_sum

    # Normalize the importance of each criterion
    return imp_vector / np.sum(imp_vector)

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
Python implementation of the mTOPSIS scoring method. For more information, see
the following publication:

* H. Deng, C.-H. Yeh, and R. J. Willis, “Inter-company comparison using
  modified TOPSIS with objective weights,” *Comput. Oper. Res.*, vol. 27, no.
  10, pp. 963–973, 2000, doi: `10.1016/S0305-0548(99)00069-6
  <https://doi.org/10.1016/S0305-0548(99)00069-6>`_.

"""

import numpy as np

from ..helper_validation import check_scoring_input


def mtopsis(z_matrix, w_vector, is_benefit_z):
    """
    Return the Modified Technique for Order Preference by Similarity to Ideal
    Solution scores of the provided decision matrix with the provided weight
    vector.
    """
    # Perform sanity checks
    z_matrix = np.array(z_matrix, dtype=np.float64)
    w_vector = np.array(w_vector, dtype=np.float64)
    check_scoring_input(z_matrix, w_vector, is_benefit_z, "mTOPSIS")

    # mTOPSIS scores should always be sorted in descending order
    desc_order = True

    # Derive the positive and negative ideal solutions
    pos_ideal_sol = np.zeros(z_matrix.shape[1], dtype=np.float64)
    neg_ideal_sol = np.zeros(z_matrix.shape[1], dtype=np.float64)
    for j in range(z_matrix.shape[1]):
        if is_benefit_z[j]:
            pos_ideal_sol[j] = np.amax(z_matrix[:, j])
            neg_ideal_sol[j] = np.amin(z_matrix[:, j])
        else:
            pos_ideal_sol[j] = np.amin(z_matrix[:, j])
            neg_ideal_sol[j] = np.amax(z_matrix[:, j])

    # Compute the score of each alternative
    s_vector = np.zeros(z_matrix.shape[0], dtype=np.float64)
    for i in range(z_matrix.shape[0]):
        pos_ideal_dist = 0.0
        neg_ideal_dist = 0.0
        for j in range(z_matrix.shape[1]):
            pos_ideal_dist += (
                w_vector[j] * (pos_ideal_sol[j] - z_matrix[i, j])**2
            )
            neg_ideal_dist += (
                w_vector[j] * (z_matrix[i, j] - neg_ideal_sol[j])**2
            )
        pos_ideal_dist = np.sqrt(pos_ideal_dist)
        neg_ideal_dist = np.sqrt(neg_ideal_dist)
        s_vector[i] = neg_ideal_dist / (neg_ideal_dist + pos_ideal_dist)

    return s_vector, desc_order

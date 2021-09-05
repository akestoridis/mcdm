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
Python implementation of the TOPSIS scoring method. For more information, see
the following publication:

* C.-L. Hwang and K. Yoon, *Multiple Attribute Decision Making*, ser. Lecture
  Notes in Economics and Mathematical Systems. Springer-Verlag Berlin
  Heidelberg, 1981, vol. 186, isbn: 9783540105589.

"""

import numpy as np

from ..helper_validation import check_scoring_input


def topsis(z_matrix, w_vector, is_benefit_z):
    """
    Return the Technique for Order Preference by Similarity to Ideal Solution
    scores of the provided decision matrix with the provided weight vector.
    """
    # Perform sanity checks
    z_matrix = np.array(z_matrix, dtype=np.float64)
    w_vector = np.array(w_vector, dtype=np.float64)
    check_scoring_input(z_matrix, w_vector, is_benefit_z, "TOPSIS")

    # TOPSIS scores should always be sorted in descending order
    desc_order = True

    # Construct the weighted normalized decision matrix
    t_matrix = np.multiply(z_matrix, w_vector)

    # Derive the positive and negative ideal solutions
    pos_ideal_sol = np.zeros(t_matrix.shape[1], dtype=np.float64)
    neg_ideal_sol = np.zeros(t_matrix.shape[1], dtype=np.float64)
    for j in range(t_matrix.shape[1]):
        if is_benefit_z[j]:
            pos_ideal_sol[j] = np.amax(t_matrix[:, j])
            neg_ideal_sol[j] = np.amin(t_matrix[:, j])
        else:
            pos_ideal_sol[j] = np.amin(t_matrix[:, j])
            neg_ideal_sol[j] = np.amax(t_matrix[:, j])

    # Compute the score of each alternative
    s_vector = np.zeros(t_matrix.shape[0], dtype=np.float64)
    for i in range(t_matrix.shape[0]):
        pos_ideal_dist = np.linalg.norm(pos_ideal_sol - t_matrix[i, :])
        neg_ideal_dist = np.linalg.norm(t_matrix[i, :] - neg_ideal_sol)
        s_vector[i] = neg_ideal_dist / (neg_ideal_dist + pos_ideal_dist)

    return s_vector, desc_order

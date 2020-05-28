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

from .normalize import normalize
from .score import score
from .weigh import weigh


def rank(x_matrix, alt_names=None, is_benefit_x=None, n_method=None,
         w_vector=None, c_method=None, w_method="MW", s_method="SAW"):
    """Rank a decision matrix using the selected methods."""
    # Make sure that the decision matrix is a float64 NumPy array
    if type(x_matrix) is not np.ndarray:
        x_matrix = np.array(x_matrix, dtype=np.float64)
    elif x_matrix.dtype is not np.float64:
        x_matrix = np.array(x_matrix, dtype=np.float64)

    # Create a list of names for the alternatives, if none were given
    if alt_names is None:
        alt_names = ["a" + str(i + 1) for i in range(x_matrix.shape[0])]

    # Sanity check
    if len(alt_names) != x_matrix.shape[0]:
        raise ValueError("The number of names for the alternatives does not "
                         "match the number of rows in the decision matrix")

    # If not specified, consider all criteria as benefit criteria
    if is_benefit_x is None:
        is_benefit_x = [True for _ in range(x_matrix.shape[1])]

    # Sanity check
    if len(is_benefit_x) != x_matrix.shape[1]:
        raise ValueError("The number of variables in the list that "
                         "determines whether each criterion is a benefit "
                         "or a cost criterion does not match the number "
                         "of columns in the decision matrix")

    # Normalize the decision matrix using the selected method
    z_matrix, is_benefit_z = normalize(x_matrix, is_benefit_x, n_method)

    # Determine the weight of each criterion
    if w_vector is None:
        # Weigh each criterion using the selected methods
        w_vector = weigh(z_matrix, w_method, c_method)
    else:
        # Make sure that the weight vector is a float64 NumPy array
        if type(w_vector) is not np.ndarray:
            w_vector = np.array(w_vector, dtype=np.float64)
        elif w_vector.dtype is not np.float64:
            w_vector = np.array(w_vector, dtype=np.float64)

        # Sanity checks
        if w_vector.shape != (x_matrix.shape[1],):
            raise ValueError("The shape of the weight vector is not "
                             "appropriate for the number of columns in the "
                             "decision matrix")
        elif not np.isclose(np.sum(w_vector), 1.0):
            raise ValueError("The weight vector's elements must sum to 1")

    # Score each alternative using the selected method
    s_vector, desc_order = score(z_matrix, is_benefit_z, w_vector, s_method)

    # Get the indices of the sorted scores
    if desc_order:
        r_indices = np.argsort(-s_vector)
    else:
        r_indices = np.argsort(s_vector)

    # Create a list of tuples that includes the names of the alternatives and
    # their corresponding scores in descending order
    ranking = []
    for i in range(len(alt_names)):
        ranking.append((alt_names[r_indices[i]], s_vector[r_indices[i]]))

    return ranking

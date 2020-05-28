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

from . import normalization


def normalize(x_matrix, is_benefit_x, n_method):
    """Normalize a decision matrix using the selected normalization method."""
    # Use the selected normalization method
    if n_method is None:
        # Make sure that the decision matrix is a float64 NumPy array
        if type(x_matrix) is not np.ndarray:
            x_matrix = np.array(x_matrix, dtype=np.float64)
        elif x_matrix.dtype is not np.float64:
            x_matrix = np.array(x_matrix, dtype=np.float64)

        # Sanity check
        if len(is_benefit_x) != x_matrix.shape[1]:
            raise ValueError("The number of variables in the list that "
                             "determines whether each criterion is a benefit "
                             "or a cost criterion does not match the number "
                             "of columns in the decision matrix")

        # Make sure that the decision matrix is already normalized
        if (np.sum(np.less(x_matrix, 0.0)) > 0
                or np.sum(np.greater(x_matrix, 1.0)) > 0):
            raise ValueError("The decision matrix is not normalized such "
                             "that each element is between 0 and 1")
        else:
            return np.copy(x_matrix), is_benefit_x.copy()
    elif n_method.upper() == "LINEAR1":
        return normalization.linear1(x_matrix, is_benefit_x)
    elif n_method.upper() == "LINEAR2":
        return normalization.linear2(x_matrix, is_benefit_x)
    elif n_method.upper() == "LINEAR3":
        return normalization.linear3(x_matrix, is_benefit_x)
    elif n_method.upper() == "VECTOR":
        return normalization.vector(x_matrix, is_benefit_x)
    else:
        raise ValueError("Unknown normalization method ({})".format(n_method))

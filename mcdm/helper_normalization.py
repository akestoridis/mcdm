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
Helper module for the normalization methods of the ``mcdm`` package.
"""

import numpy as np

from . import normalization
from .helper_validation import check_normalization_input


def normalize(x_matrix, is_benefit_x, n_method):
    """
    Return the normalized version of the provided matrix using the selected
    normalization method.
    """
    # Use the selected normalization method
    if n_method is None:
        # Perform sanity checks
        x_matrix = np.array(x_matrix, dtype=np.float64)
        check_normalization_input(x_matrix, is_benefit_x, None)

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

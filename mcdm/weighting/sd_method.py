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
Python implementation of the SD weighting method. For more information, see
the following publication:

* D. Diakoulaki, G. Mavrotas, and L. Papayannakis, “Determining objective
  weights in multiple criteria problems: The CRITIC method,” *Comput. Oper.
  Res.*, vol. 22, no. 7, pp. 763–770, 1995, doi: `10.1016/0305-0548(94)00059-H
  <https://doi.org/10.1016/0305-0548(94)00059-H>`_.

"""

import numpy as np

from ..helper_validation import check_weighting_input


def sd(z_matrix):
    """
    Return the weight vector of the provided decision matrix using the
    Standard Deviation method.
    """
    # Perform sanity checks
    z_matrix = np.array(z_matrix, dtype=np.float64)
    check_weighting_input(z_matrix, "", "SD")

    # Compute the standard deviation of each criterion
    sd_vector = np.std(z_matrix, axis=0, dtype=np.float64)

    # The importance of each criterion corresponds to
    # its normalized standard deviation
    return sd_vector / np.sum(sd_vector)

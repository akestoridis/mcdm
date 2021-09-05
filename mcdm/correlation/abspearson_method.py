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
Python implementation of the AbsPearson correlation method. For more
information, see the following publication:

* J. L. Rodgers and W. A. Nicewander, “Thirteen ways to look at the
  correlation coefficient,” *Amer. Statist.*, vol. 42, no. 1, pp. 59–66, 1988,
  doi: `10.2307/2685263
  <https://doi.org/10.2307/2685263>`_.

"""

import numpy as np


def abspearson(z_matrix):
    """
    Return the absolute value of the Pearson correlation coefficients of the
    provided matrix.
    """
    # Make sure that the provided matrix is a float64 NumPy array
    z_matrix = np.array(z_matrix, dtype=np.float64)

    return np.absolute(np.corrcoef(z_matrix, rowvar=False))

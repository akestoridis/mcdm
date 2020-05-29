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

from mcdm.correlate import correlate


def vic(z_matrix, c_method="dCor"):
    """Python implementation of the VIC weighting method.

    For more information, see the following publication:
      * D.-G. Akestoridis and E. Papapetrou, "A framework for the evaluation
        of routing protocols in opportunistic networks," Computer
        Communications, vol. 145, pp. 14--28, 2019.
        DOI: 10.1016/j.comcom.2019.06.003.
    """
    # Make sure that the decision matrix is a float64 NumPy array
    if type(z_matrix) is not np.ndarray:
        z_matrix = np.array(z_matrix, dtype=np.float64)
    elif z_matrix.dtype is not np.float64:
        z_matrix = np.array(z_matrix, dtype=np.float64)

    # Make sure that the decision matrix is normalized
    if (np.sum(np.less(z_matrix, 0.0)) > 0
            or np.sum(np.greater(z_matrix, 1.0)) > 0):
        raise ValueError("The decision matrix must be normalized "
                         "in order to apply the VIC weighting method")

    # Compute the standard deviation of each criterion
    sd_vector = np.std(z_matrix, axis=0, dtype=np.float64)

    # By default, VIC is using distance correlation (dCor) coefficients
    if c_method is None:
        c_method = "dCor"

    # Make sure that VIC is compatible with the selected correlation method
    if c_method.upper() in {"PEARSON"}:
        raise ValueError("The VIC weighting method is not compatible "
                         "with the {} correlation method"
                         "".format(c_method))
    elif c_method.upper() not in {"ABSPEARSON", "DCOR"}:
        raise ValueError("Unknown compatibility of the VIC weighting "
                         "method with the {} correlation method"
                         "".format(c_method))

    # Compute the correlation coefficients between pairs of criteria
    corr_matrix = correlate(z_matrix, c_method)

    # Compute the importance of each criterion
    imp_vector = np.zeros(z_matrix.shape[1], dtype=np.float64)
    for j in range(z_matrix.shape[1]):
        tmp_sum = 0.0
        for l in range(z_matrix.shape[1]):
            tmp_sum += corr_matrix[j, l]
        imp_vector[j] = sd_vector[j] / tmp_sum

    # Normalize the importance of each criterion
    return imp_vector / np.sum(imp_vector)

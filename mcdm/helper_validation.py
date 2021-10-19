# Copyright (c) 2021 Dimitrios-Georgios Akestoridis
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
Helper module for the validation functions of the ``mcdm`` package.
"""

import numpy as np


def is_normalized_matrix(z_matrix):
    """
    Return a Boolean value to indicate whether the matrix is normalized or not
    """
    return (
        np.sum(np.less(z_matrix, 0.0)) == 0
        and np.sum(np.greater(z_matrix, 1.0)) == 0
    )


def is_normalized_vector(w_vector):
    """
    Return a Boolean value to indicate whether the vector is normalized or not
    """
    return (
        np.sum(np.less(w_vector, 0.0)) == 0
        and np.isclose(np.sum(w_vector), 1.0)
    )


def check_scoring_input(z_matrix, w_vector, is_benefit_z, s_method):
    """
    Raise an exception if any argument is inappropriate for the corresponding
    scoring method
    """
    if s_method.upper() in {"SAW", "MEW", "TOPSIS", "MTOPSIS"}:
        if not is_normalized_matrix(z_matrix):
            raise ValueError(
                "The decision matrix must be normalized in order to apply "
                + "the {} scoring method".format(s_method),
            )
        if not is_normalized_vector(w_vector):
            raise ValueError(
                "The weight vector must be normalized in order to apply "
                + "the {} scoring method".format(s_method),
            )
        if w_vector.shape != (z_matrix.shape[1],):
            raise ValueError(
                "The shape of the weight vector is not appropriate for the "
                + "number of columns in the decision matrix",
            )
        if len(is_benefit_z) != z_matrix.shape[1]:
            raise ValueError(
                "The number of variables in the list that determines whether "
                + "each criterion is a benefit or a cost criterion does not "
                + "match the number of columns in the decision matrix",
            )
    else:
        raise ValueError("Unknown scoring method ({})".format(s_method))


def check_weighting_input(z_matrix, c_method, w_method):
    """
    Raise an exception if any argument is inappropriate for the corresponding
    weighting method
    """
    if w_method.upper() in {"MW", "EM", "SD", "CRITIC", "VIC"}:
        if not is_normalized_matrix(z_matrix):
            raise ValueError(
                "The decision matrix must be normalized in order to apply "
                + "the {} weighting method".format(w_method),
            )
        if w_method.upper() == "EM":
            if (
                not np.all(
                    np.isclose(
                        np.sum(z_matrix, axis=0),
                        np.ones(z_matrix.shape[1]),
                    )
                )
            ):
                raise ValueError(
                    "The columns of the decision matrix must sum to 1 in "
                    + "order to apply the EM weighting method",
                )
        elif w_method.upper() == "CRITIC":
            if c_method.upper() not in {"PEARSON", "ABSPEARSON", "DCOR"}:
                raise ValueError(
                    "Unknown compatibility of the CRITIC weighting method "
                    + "with the {} correlation method".format(c_method)
                )
        elif w_method.upper() == "VIC":
            if c_method.upper() in {"PEARSON"}:
                raise ValueError(
                    "The VIC weighting method is not compatible with the "
                    + "{} correlation method".format(c_method),
                )
            if c_method.upper() not in {"ABSPEARSON", "DCOR"}:
                raise ValueError(
                    "Unknown compatibility of the VIC weighting method with "
                    + "the {} correlation method".format(c_method),
                )
    else:
        raise ValueError("Unknown weighting method ({})".format(w_method))


def check_normalization_input(x_matrix, is_benefit_x, n_method):
    """
    Raise an exception if any argument is inappropriate for the corresponding
    normalization method
    """
    if (
        n_method is None
        or n_method.upper() in {"LINEAR1", "LINEAR2", "LINEAR3", "VECTOR"}
    ):
        if len(is_benefit_x) != x_matrix.shape[1]:
            raise ValueError(
                "The number of variables in the list that determines whether "
                + "each criterion is a benefit or a cost criterion does not "
                + "match the number of columns in the matrix",
            )
        if n_method is None:
            if not is_normalized_matrix(x_matrix):
                raise ValueError(
                    "The matrix is not normalized such that each element is "
                    + "between 0 and 1",
                )
        elif n_method.upper() in {"LINEAR1", "LINEAR3", "VECTOR"}:
            if np.sum(np.less(x_matrix, 0.0)) > 0:
                raise ValueError(
                    "The matrix must not contain any "
                    + "negative numbers in order to apply the "
                    + "{} normalization method".format(n_method),
                )
    else:
        raise ValueError("Unknown normalization method ({})".format(n_method))

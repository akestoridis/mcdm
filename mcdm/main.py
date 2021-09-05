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
Main module for the ``mcdm`` package.
"""

import csv
import numpy as np

from .helper_normalization import normalize
from .helper_scoring import score
from .helper_weighting import weigh


def rank(
    x_matrix,
    alt_names=None,
    is_benefit_x=None,
    n_method=None,
    w_vector=None,
    c_method=None,
    w_method="MW",
    s_method="SAW",
):
    """
    Return the ranking of the alternatives, in descending order, using the
    selected methods.
    """
    # Perform sanity checks
    x_matrix = np.array(x_matrix, dtype=np.float64)
    if alt_names is None:
        alt_names = ["a" + str(i + 1) for i in range(x_matrix.shape[0])]
    if len(alt_names) != x_matrix.shape[0]:
        raise ValueError(
            "The number of names for the alternatives does not match the "
            + "number of rows in the decision matrix",
        )

    # If not specified, consider all criteria as benefit criteria
    if is_benefit_x is None:
        is_benefit_x = [True for _ in range(x_matrix.shape[1])]

    # Normalize the decision matrix using the selected method
    z_matrix, is_benefit_z = normalize(x_matrix, is_benefit_x, n_method)

    # Determine the weight of each criterion
    if w_vector is None:
        # Weigh each criterion using the selected methods
        w_vector = weigh(z_matrix, w_method, c_method)

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


def load(filepath, delimiter=",", skiprows=0, labeled_rows=False):
    """
    Return a matrix, and potentially row labels, from a text file.
    """
    matrix = None
    row_labels = None
    if labeled_rows:
        # Separate the row labels from the matrix data
        row_labels = []
        matrix_data = []
        num_columns = None
        with open(filepath, mode="r", encoding="utf-8") as fp:
            rows = csv.reader(fp, delimiter=delimiter)
            for i, row in enumerate(rows, start=1):
                # Skip the selected number of rows
                if i <= skiprows:
                    continue

                # Determine the expected number of columns
                if num_columns is None:
                    num_columns = len(row) - 1

                # Perform sanity checks
                if len(row) <= 1:
                    raise ValueError(
                        "The matrix should have at least 1 column with data",
                    )
                if len(row) - 1 != num_columns:
                    raise ValueError(
                        "Wrong number of columns at line {}".format(i),
                    )

                # The row labels are expected to be
                # in the first column of the text file
                row_labels.append(row[0])
                matrix_data.append(row[1:])
        # Convert the matrix data into a float64 NumPy array
        matrix = np.array(matrix_data, dtype=np.float64)
    else:
        # Load the matrix from the text file as a float64 NumPy array
        matrix = np.loadtxt(
            filepath,
            dtype=np.float64,
            delimiter=delimiter,
            skiprows=skiprows,
        )

    return matrix, row_labels

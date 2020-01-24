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

import csv
import numpy as np


def load(filepath, delimiter=",", skiprows=0, labeled_rows=False):
    """Load a matrix, and potentially row labels, from a text file."""
    matrix = None
    row_labels = None
    if labeled_rows:
        # Separate the row labels from the matrix data
        row_labels = []
        matrix_data = []
        num_columns = None
        with open(filepath, "r") as fp:
            rows = csv.reader(fp, delimiter=delimiter)
            for i, row in enumerate(rows, start=1):
                # Skip the selected number of rows
                if i <= skiprows:
                    continue

                # Determine the expected number of columns
                if num_columns is None:
                    num_columns = len(row) - 1

                # Sanity checks
                if len(row) <= 1:
                    raise ValueError("The matrix should have at "
                                     "least 1 column with data")
                elif len(row) - 1 != num_columns:
                    raise ValueError("Wrong number of columns at "
                                     "line {}".format(i))

                # The row labels are expected to be
                # in the first column of the text file
                row_labels.append(row[0])
                matrix_data.append(row[1:])
        # Convert the matrix data into a float64 NumPy array
        matrix = np.array(matrix_data, dtype=np.float64)
    else:
        # Load the matrix from the text file as a float64 NumPy array
        matrix = np.loadtxt(filepath, dtype=np.float64, delimiter=delimiter,
                            skiprows=skiprows)

    return matrix, row_labels

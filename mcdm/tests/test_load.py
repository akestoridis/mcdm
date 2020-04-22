#!/usr/bin/env python3

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

import mcdm
import numpy as np
import os
import unittest


DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TestLoad(unittest.TestCase):
    def test_default_loading(self):
        """Test loading a matrix with the default parameter values."""
        filepath = os.path.join(DIR_PATH, "data", "example1.csv")
        obtained_matrix, obtained_row_labels = mcdm.load(filepath)
        expected_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_labeled_loading(self):
        """Test loading a matrix with column and row labels."""
        filepath = os.path.join(DIR_PATH, "data", "example2.csv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, skiprows=1, labeled_rows=True)
        expected_matrix = np.array(
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.2, 0.5, 0.0],
             [0.2, 0.5, 1.0],
             [0.4, 1.0, 0.0],
             [0.4, 1.0, 1.0],
             [0.6, 1.0, 0.0],
             [0.6, 1.0, 1.0],
             [0.8, 0.5, 0.0],
             [0.8, 0.5, 1.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        expected_row_labels = [
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "a8",
            "a9",
            "a10",
            "a11",
            "a12"
        ]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_unlabeled_tsv_loading(self):
        """Test loading a matrix without any labels from a TSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example3.tsv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, delimiter="\t")
        expected_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_labeled_tsv_loading(self):
        """Test loading a matrix with row labels from a TSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example4.tsv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, delimiter="\t", labeled_rows=True)
        expected_matrix = np.array(
            [[ 2.0,  0.0, 5.0,  -2.0, 26.0, 7.0,  6.0],
             [ 4.0, -1.0, 4.0,   0.0,  5.0, 7.0,  3.0],
             [12.0, -2.0, 3.0,   5.0, 10.0, 7.0,  1.0],
             [ 3.0, -3.0, 2.0,   2.0, 50.0, 7.0, -4.0],
             [ 6.0, -4.0, 1.0,  -5.0,  2.0, 7.0, -1.0],
             [ 1.0, -5.0, 0.0, -10.0, 18.0, 7.0, -5.0]],
            dtype=np.float64)
        expected_row_labels = ["A", "B", "C", "D", "E", "F"]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_num_columns_exception(self):
        """Test loading a matrix with a wrong number of columns."""
        filepath = os.path.join(DIR_PATH, "data", "failure1.tsv")
        self.assertRaises(ValueError, mcdm.load, filepath,
                          delimiter="\t", labeled_rows=True)

    def test_no_columns_exception(self):
        """Test loading a matrix without any columns."""
        filepath = os.path.join(DIR_PATH, "data", "failure2.tsv")
        self.assertRaises(ValueError, mcdm.load, filepath,
                          delimiter="\t", labeled_rows=True)

    def test_wrong_type_exception(self):
        """Test loading a matrix with a wrong value type."""
        filepath = os.path.join(DIR_PATH, "data", "failure3.csv")
        self.assertRaises(ValueError, mcdm.load, filepath,
                          skiprows=1, labeled_rows=True)


if __name__ == "__main__":
    unittest.main()

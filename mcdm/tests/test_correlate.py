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
import unittest


class TestCorrelate(unittest.TestCase):
    def test_pearson_linear(self):
        """Test the Pearson method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_corr_matrix = mcdm.correlate(z_matrix, "Pearson")
        expected_corr_matrix = np.array(
            [[ 1.0000000,  0.9314381, -0.9314381],
             [ 0.9314381,  1.0000000, -1.0000000],
             [-0.9314381, -1.0000000,  1.0000000]],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_corr_matrix,
                                   expected_corr_matrix)
        self.assertEqual(obtained_corr_matrix.dtype,
                         expected_corr_matrix.dtype)

    def test_pearson_nonlinear(self):
        """Test the Pearson method with a non-linear association."""
        z_matrix = np.array(
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
        obtained_corr_matrix = mcdm.correlate(z_matrix, "Pearson")
        expected_corr_matrix = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_corr_matrix,
                                   expected_corr_matrix)
        self.assertEqual(obtained_corr_matrix.dtype,
                         expected_corr_matrix.dtype)

    def test_abspearson_linear(self):
        """Test the AbsPearson method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_corr_matrix = mcdm.correlate(z_matrix, "AbsPearson")
        expected_corr_matrix = np.array(
            [[1.0000000, 0.9314381, 0.9314381],
             [0.9314381, 1.0000000, 1.0000000],
             [0.9314381, 1.0000000, 1.0000000]],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_corr_matrix,
                                   expected_corr_matrix)
        self.assertEqual(obtained_corr_matrix.dtype,
                         expected_corr_matrix.dtype)

    def test_abspearson_nonlinear(self):
        """Test the AbsPearson method with a non-linear association."""
        z_matrix = np.array(
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
        obtained_corr_matrix = mcdm.correlate(z_matrix, "AbsPearson")
        expected_corr_matrix = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_corr_matrix,
                                   expected_corr_matrix)
        self.assertEqual(obtained_corr_matrix.dtype,
                         expected_corr_matrix.dtype)

    def test_dcor_linear(self):
        """Test the dCor method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_corr_matrix = mcdm.correlate(z_matrix, "dCor")
        expected_corr_matrix = np.array(
            [[1.0000000, 0.9369189, 0.9369189],
             [0.9369189, 1.0000000, 1.0000000],
             [0.9369189, 1.0000000, 1.0000000]],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_corr_matrix,
                                   expected_corr_matrix)
        self.assertEqual(obtained_corr_matrix.dtype,
                         expected_corr_matrix.dtype)

    def test_dcor_nonlinear(self):
        """Test the dCor method with a non-linear association."""
        z_matrix = np.array(
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
        obtained_corr_matrix = mcdm.correlate(z_matrix, "dCor")
        expected_corr_matrix = np.array(
            [[1.0000000, 0.5186014, 0.0000000],
             [0.5186014, 1.0000000, 0.0000000],
             [0.0000000, 0.0000000, 1.0000000]],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_corr_matrix,
                                   expected_corr_matrix)
        self.assertEqual(obtained_corr_matrix.dtype,
                         expected_corr_matrix.dtype)

    def test_unknown_correlate_exception(self):
        """Test the selection of an unknown correlation method."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.correlate, z_matrix, "Unknown")


if __name__ == "__main__":
    unittest.main()

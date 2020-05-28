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


class TestNormalize(unittest.TestCase):
    def test_none_calculations(self):
        """Test a decision matrix that is already normalized."""
        x_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, True, True]
        obtained_z_matrix, obtained_is_benefit_z = mcdm.normalize(
            x_matrix, is_benefit_x, None)
        expected_z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        expected_is_benefit_z = [True, True, True]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_none_over_exception(self):
        """Test a decision matrix with a value greater than 1."""
        x_matrix = np.array(
            [[0.0, 0.0, 1.1],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, True, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, None)

    def test_none_under_exception(self):
        """Test a decision matrix with a value less than 0."""
        x_matrix = np.array(
            [[ 0.0, 0.0, 1.0],
             [-0.1, 0.2, 0.8],
             [ 0.2, 0.4, 0.6],
             [ 0.3, 0.7, 0.3],
             [ 0.6, 0.8, 0.2],
             [ 0.8, 0.9, 0.1],
             [ 1.0, 1.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, True, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, None)

    def test_none_is_benefit_x_exception(self):
        """Test a decision matrix with an invalid Boolean list."""
        x_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, True, True, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, None)

    def test_linear1_calculations(self):
        """Test the calculations of the Linear1 method."""
        x_matrix = np.array(
            [[ 2.0,  12.0, 7.0, 7.0],
             [ 4.0, 100.0, 7.0, 7.0],
             [10.0, 200.0, 7.0, 7.0],
             [ 0.0, 300.0, 7.0, 7.0],
             [ 6.0, 400.0, 7.0, 7.0],
             [ 1.0, 600.0, 7.0, 7.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        obtained_z_matrix, obtained_is_benefit_z = mcdm.normalize(
            x_matrix, is_benefit_x, "Linear1")
        expected_z_matrix = np.array(
            [[0.2, 1.00, 1.0, 1.0],
             [0.4, 0.12, 1.0, 1.0],
             [1.0, 0.06, 1.0, 1.0],
             [0.0, 0.04, 1.0, 1.0],
             [0.6, 0.03, 1.0, 1.0],
             [0.1, 0.02, 1.0, 1.0]],
            dtype=np.float64)
        expected_is_benefit_z = [True, True, True, True]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_linear1_negative_exception(self):
        """Test the Linear1 method with a negative value."""
        x_matrix = np.array(
            [[ 2.0,  12.0, 7.0, 7.0],
             [-4.0, 100.0, 7.0, 7.0],
             [10.0, 200.0, 7.0, 7.0],
             [ 0.0, 300.0, 7.0, 7.0],
             [ 6.0, 400.0, 7.0, 7.0],
             [ 1.0, 600.0, 7.0, 7.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear1")

    def test_linear1_zero_benefit_exception(self):
        """Test the Linear1 method with a zero benefit vector."""
        x_matrix = np.array(
            [[ 2.0,  12.0, 0.0, 7.0],
             [ 4.0, 100.0, 0.0, 7.0],
             [10.0, 200.0, 0.0, 7.0],
             [ 0.0, 300.0, 0.0, 7.0],
             [ 6.0, 400.0, 0.0, 7.0],
             [ 1.0, 600.0, 0.0, 7.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear1")

    def test_linear1_zero_cost_exception(self):
        """Test the Linear1 method with a zero cost vector."""
        x_matrix = np.array(
            [[ 2.0,  12.0, 7.0, 0.0],
             [ 4.0, 100.0, 7.0, 0.0],
             [10.0, 200.0, 7.0, 0.0],
             [ 0.0, 300.0, 7.0, 0.0],
             [ 6.0, 400.0, 7.0, 0.0],
             [ 1.0, 600.0, 7.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear1")

    def test_linear1_is_benefit_x_exception(self):
        """Test the Linear1 method with an invalid Boolean list."""
        x_matrix = np.array(
            [[ 2.0,  12.0, 7.0, 7.0],
             [ 4.0, 100.0, 7.0, 7.0],
             [10.0, 200.0, 7.0, 7.0],
             [ 0.0, 300.0, 7.0, 7.0],
             [ 6.0, 400.0, 7.0, 7.0],
             [ 1.0, 600.0, 7.0, 7.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear1")

    def test_linear2_calculations(self):
        """Test the calculations of the Linear2 method."""
        x_matrix = np.array(
            [[ 8.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
             [24.0, 24.0, -11.0, -11.0,   0.0,   0.0],
             [ 4.0,  4.0, -10.0, -10.0,  40.0,  40.0],
             [14.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
             [ 6.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
             [18.0, 18.0,  -5.0,  -5.0, -10.0, -10.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True, False]
        obtained_z_matrix, obtained_is_benefit_z = mcdm.normalize(
            x_matrix, is_benefit_x, "Linear2")
        expected_z_matrix = np.array(
            [[0.2, 0.8, 1.0, 0.0, 0.3, 0.7],
             [1.0, 0.0, 0.0, 1.0, 0.2, 0.8],
             [0.0, 1.0, 0.1, 0.9, 1.0, 0.0],
             [0.5, 0.5, 0.2, 0.8, 0.5, 0.5],
             [0.1, 0.9, 0.4, 0.6, 0.1, 0.9],
             [0.7, 0.3, 0.6, 0.4, 0.0, 1.0]],
            dtype=np.float64)
        expected_is_benefit_z = [True, True, True, True, True, True]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_linear2_positive_constant_exception(self):
        """Test the Linear2 method with a positive constant vector."""
        x_matrix = np.array(
            [[7.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
             [7.0, 24.0, -11.0, -11.0,   0.0,   0.0],
             [7.0,  4.0, -10.0, -10.0,  40.0,  40.0],
             [7.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
             [7.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
             [7.0, 18.0,  -5.0,  -5.0, -10.0, -10.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear2")

    def test_linear2_negative_constant_exception(self):
        """Test the Linear2 method with a negative constant vector."""
        x_matrix = np.array(
            [[-7.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
             [-7.0, 24.0, -11.0, -11.0,   0.0,   0.0],
             [-7.0,  4.0, -10.0, -10.0,  40.0,  40.0],
             [-7.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
             [-7.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
             [-7.0, 18.0,  -5.0,  -5.0, -10.0, -10.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear2")

    def test_linear2_zero_constant_exception(self):
        """Test the Linear2 method with a zero constant vector."""
        x_matrix = np.array(
            [[0.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
             [0.0, 24.0, -11.0, -11.0,   0.0,   0.0],
             [0.0,  4.0, -10.0, -10.0,  40.0,  40.0],
             [0.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
             [0.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
             [0.0, 18.0,  -5.0,  -5.0, -10.0, -10.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear2")

    def test_linear2_is_benefit_x_exception(self):
        """Test the Linear2 method with an invalid Boolean list."""
        x_matrix = np.array(
            [[ 8.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
             [24.0, 24.0, -11.0, -11.0,   0.0,   0.0],
             [ 4.0,  4.0, -10.0, -10.0,  40.0,  40.0],
             [14.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
             [ 6.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
             [18.0, 18.0,  -5.0,  -5.0, -10.0, -10.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True, False, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear2")

    def test_linear3_calculations(self):
        """Test the calculations of the Linear3 method."""
        x_matrix = np.array(
            [[4.0, 4.0, 7.0, 7.0],
             [3.0, 3.0, 7.0, 7.0],
             [2.0, 2.0, 7.0, 7.0],
             [1.0, 1.0, 7.0, 7.0],
             [0.0, 0.0, 7.0, 7.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        obtained_z_matrix, obtained_is_benefit_z = mcdm.normalize(
            x_matrix, is_benefit_x, "Linear3")
        expected_z_matrix = np.array(
            [[0.4, 0.4, 0.2, 0.2],
             [0.3, 0.3, 0.2, 0.2],
             [0.2, 0.2, 0.2, 0.2],
             [0.1, 0.1, 0.2, 0.2],
             [0.0, 0.0, 0.2, 0.2]],
            dtype=np.float64)
        expected_is_benefit_z = [True, False, True, False]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_linear3_negative_exception(self):
        """Test the Linear3 method with a negative value."""
        x_matrix = np.array(
            [[ 4.0, 4.0, 7.0, 7.0],
             [ 3.0, 3.0, 7.0, 7.0],
             [-2.0, 2.0, 7.0, 7.0],
             [ 1.0, 1.0, 7.0, 7.0],
             [ 0.0, 0.0, 7.0, 7.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear3")

    def test_linear3_zero_constant_exception(self):
        """Test the Linear3 method with a zero constant vector."""
        x_matrix = np.array(
            [[4.0, 4.0, 7.0, 0.0],
             [3.0, 3.0, 7.0, 0.0],
             [2.0, 2.0, 7.0, 0.0],
             [1.0, 1.0, 7.0, 0.0],
             [0.0, 0.0, 7.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear3")

    def test_linear3_is_benefit_x_exception(self):
        """Test the Linear3 method with an invalid Boolean list."""
        x_matrix = np.array(
            [[4.0, 4.0, 7.0, 7.0],
             [3.0, 3.0, 7.0, 7.0],
             [2.0, 2.0, 7.0, 7.0],
             [1.0, 1.0, 7.0, 7.0],
             [0.0, 0.0, 7.0, 7.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Linear3")

    def test_vector_calculations(self):
        """Test the calculations of the Vector method."""
        x_matrix = np.array(
            [[0.0, 0.0, 5.0, 5.0],
             [6.0, 6.0, 5.0, 5.0],
             [0.0, 0.0, 5.0, 5.0],
             [8.0, 8.0, 5.0, 5.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        obtained_z_matrix, obtained_is_benefit_z = mcdm.normalize(
            x_matrix, is_benefit_x, "Vector")
        expected_z_matrix = np.array(
            [[0.0, 0.0, 0.5, 0.5],
             [0.6, 0.6, 0.5, 0.5],
             [0.0, 0.0, 0.5, 0.5],
             [0.8, 0.8, 0.5, 0.5]],
            dtype=np.float64)
        expected_is_benefit_z = [True, False, True, False]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_vector_negative_exception(self):
        """Test the Vector method with a negative value."""
        x_matrix = np.array(
            [[0.0,  0.0, 5.0, 5.0],
             [6.0, -6.0, 5.0, 5.0],
             [0.0,  0.0, 5.0, 5.0],
             [8.0,  8.0, 5.0, 5.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Vector")

    def test_vector_zero_constant_exception(self):
        """Test the Vector method with a zero constant vector."""
        x_matrix = np.array(
            [[0.0, 0.0, 5.0, 0.0],
             [6.0, 6.0, 5.0, 0.0],
             [0.0, 0.0, 5.0, 0.0],
             [8.0, 8.0, 5.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Vector")

    def test_vector_is_benefit_x_exception(self):
        """Test the Vector method with an invalid Boolean list."""
        x_matrix = np.array(
            [[0.0, 0.0, 5.0, 5.0],
             [6.0, 6.0, 5.0, 5.0],
             [0.0, 0.0, 5.0, 5.0],
             [8.0, 8.0, 5.0, 5.0]],
            dtype=np.float64)
        is_benefit_x = [True, False, True, False, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Vector")

    def test_unknown_normalize_exception(self):
        """Test the selection of an unknown normalization method."""
        x_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        is_benefit_x = [True, True, True]
        self.assertRaises(ValueError, mcdm.normalize,
                          x_matrix, is_benefit_x, "Unknown")


if __name__ == "__main__":
    unittest.main()

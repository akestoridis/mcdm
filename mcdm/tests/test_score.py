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


class TestScore(unittest.TestCase):
    def test_saw_balanced(self):
        """Test the SAW method with a balanced decision matrix."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "SAW")
        expected_s_vector = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_saw_simple_benefit(self):
        """Test the SAW method with simple benefit criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, True, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "SAW")
        expected_s_vector = np.array(
            [0.54, 0.5, 0.46],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_saw_simple_cost(self):
        """Test the SAW method with simple cost criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [False, False, False, False, False]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "SAW")
        expected_s_vector = np.array(
            [0.54, 0.5, 0.46],
            dtype=np.float64)
        expected_desc_order = False
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_saw_over_exception(self):
        """Test the SAW method with a value greater than 1."""
        z_matrix = np.array(
            [[0.00, 1.01],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "SAW")

    def test_saw_under_exception(self):
        """Test the SAW method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.00, 1.00],
             [-0.25, 0.75],
             [ 0.50, 0.50],
             [ 0.75, 0.25],
             [ 1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "SAW")

    def test_saw_w_vector_length_exception(self):
        """Test the SAW method with an invalid weight vector length."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "SAW")

    def test_saw_w_vector_sum_exception(self):
        """Test the SAW method with an invalid weight vector sum."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.4], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "SAW")

    def test_saw_is_benefit_z_exception(self):
        """Test the SAW method with an invalid Boolean list."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "SAW")

    def test_saw_mixture_exception(self):
        """Test the SAW method with a mixture of criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, False, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "SAW")

    def test_mew_balanced(self):
        """Test the MEW method with a balanced decision matrix."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "MEW")
        expected_s_vector = np.array(
            [0.0000000, 0.4330127, 0.5000000, 0.4330127, 0.0000000],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_mew_simple_benefit(self):
        """Test the MEW method with simple benefit criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, True, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "MEW")
        expected_s_vector = np.array(
            [0.4418200, 0.5000000, 0.3163389],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_mew_simple_cost(self):
        """Test the MEW method with simple cost criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [False, False, False, False, False]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "MEW")
        expected_s_vector = np.array(
            [0.4418200, 0.5000000, 0.3163389],
            dtype=np.float64)
        expected_desc_order = False
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_mew_over_exception(self):
        """Test the MEW method with a value greater than 1."""
        z_matrix = np.array(
            [[0.00, 1.01],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "MEW")

    def test_mew_under_exception(self):
        """Test the MEW method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.00, 1.00],
             [-0.25, 0.75],
             [ 0.50, 0.50],
             [ 0.75, 0.25],
             [ 1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "MEW")

    def test_mew_w_vector_length_exception(self):
        """Test the MEW method with an invalid weight vector length."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "MEW")

    def test_mew_w_vector_sum_exception(self):
        """Test the MEW method with an invalid weight vector sum."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.4], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "MEW")

    def test_mew_is_benefit_z_exception(self):
        """Test the MEW method with an invalid Boolean list."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "MEW")

    def test_mew_mixture_exception(self):
        """Test the MEW method with a mixture of criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, False, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "MEW")

    def test_topsis_balanced(self):
        """Test the TOPSIS method with a balanced decision matrix."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "TOPSIS")
        expected_s_vector = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_topsis_simple_benefit(self):
        """Test the TOPSIS method with simple benefit criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, True, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "TOPSIS")
        expected_s_vector = np.array(
            [0.6194425, 0.5000000, 0.3805575],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_topsis_simple_cost(self):
        """Test the TOPSIS method with simple cost criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [False, False, False, False, False]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "TOPSIS")
        expected_s_vector = np.array(
            [0.3805575, 0.5000000, 0.6194425],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_topsis_simple_mixture(self):
        """Test the TOPSIS method with a mixture of criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, False, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "TOPSIS")
        expected_s_vector = np.array(
            [0.6177727, 0.5000000, 0.3822273],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_topsis_over_exception(self):
        """Test the TOPSIS method with a value greater than 1."""
        z_matrix = np.array(
            [[0.00, 1.01],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "TOPSIS")

    def test_topsis_under_exception(self):
        """Test the TOPSIS method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.00, 1.00],
             [-0.25, 0.75],
             [ 0.50, 0.50],
             [ 0.75, 0.25],
             [ 1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "TOPSIS")

    def test_topsis_w_vector_length_exception(self):
        """Test the TOPSIS method with an invalid weight vector length."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "TOPSIS")

    def test_topsis_w_vector_sum_exception(self):
        """Test the TOPSIS method with an invalid weight vector sum."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.4], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "TOPSIS")

    def test_topsis_is_benefit_z_exception(self):
        """Test the TOPSIS method with an invalid Boolean list."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "TOPSIS")

    def test_mtopsis_balanced(self):
        """Test the mTOPSIS method with a balanced decision matrix."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "mTOPSIS")
        expected_s_vector = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_mtopsis_simple_benefit(self):
        """Test the mTOPSIS method with simple benefit criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, True, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "mTOPSIS")
        expected_s_vector = np.array(
            [0.5767680, 0.5000000, 0.4232320],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_mtopsis_simple_cost(self):
        """Test the mTOPSIS method with simple cost criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [False, False, False, False, False]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "mTOPSIS")
        expected_s_vector = np.array(
            [0.4232320, 0.5000000, 0.5767680],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_mtopsis_simple_mixture(self):
        """Test the mTOPSIS method with a mixture of criteria."""
        z_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        is_benefit_z = [True, False, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = mcdm.score(
            z_matrix, is_benefit_z, w_vector, "mTOPSIS")
        expected_s_vector = np.array(
            [0.5714286, 0.5000000, 0.4285714],
            dtype=np.float64)
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_mtopsis_over_exception(self):
        """Test the mTOPSIS method with a value greater than 1."""
        z_matrix = np.array(
            [[0.00, 1.01],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "mTOPSIS")

    def test_mtopsis_under_exception(self):
        """Test the mTOPSIS method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.00, 1.00],
             [-0.25, 0.75],
             [ 0.50, 0.50],
             [ 0.75, 0.25],
             [ 1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "mTOPSIS")

    def test_mtopsis_w_vector_length_exception(self):
        """Test the mTOPSIS method with an invalid weight vector length."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "mTOPSIS")

    def test_mtopsis_w_vector_sum_exception(self):
        """Test the mTOPSIS method with an invalid weight vector sum."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.4], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "mTOPSIS")

    def test_mtopsis_is_benefit_z_exception(self):
        """Test the mTOPSIS method with an invalid Boolean list."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "mTOPSIS")

    def test_unknown_score_exception(self):
        """Test the selection of an unknown scoring method."""
        z_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(ValueError, mcdm.score,
                          z_matrix, is_benefit_z, w_vector, "Unknown")


if __name__ == "__main__":
    unittest.main()

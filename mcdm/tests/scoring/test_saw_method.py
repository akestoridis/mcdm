#!/usr/bin/env python3

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
Test module for the ``scoring/saw_method.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np

from mcdm.scoring import saw


class TestSaw(unittest.TestCase):
    """
    Test class for the ``saw`` function of the ``mcdm.scoring`` package.
    """
    def test_balanced(self):
        """
        Test the SAW scoring method with a balanced decision matrix.
        """
        z_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = saw(
            z_matrix,
            w_vector,
            is_benefit_z,
        )
        expected_s_vector = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float64,
        )
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_simple_benefit(self):
        """
        Test the SAW scoring method with simple benefit criteria.
        """
        z_matrix = np.array(
            [
                [0.5, 0.6, 0.3, 0.2, 0.9],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.4, 0.7, 0.8, 0.1],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, True, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = saw(
            z_matrix,
            w_vector,
            is_benefit_z,
        )
        expected_s_vector = np.array(
            [0.54, 0.5, 0.46],
            dtype=np.float64,
        )
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_simple_cost(self):
        """
        Test the SAW scoring method with simple cost criteria.
        """
        z_matrix = np.array(
            [
                [0.5, 0.6, 0.3, 0.2, 0.9],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.4, 0.7, 0.8, 0.1],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [False, False, False, False, False]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        obtained_s_vector, obtained_desc_order = saw(
            z_matrix,
            w_vector,
            is_benefit_z,
        )
        expected_s_vector = np.array(
            [0.54, 0.5, 0.46],
            dtype=np.float64,
        )
        expected_desc_order = False
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_float32(self):
        """
        Test the SAW scoring method with float32 NumPy arrays.
        """
        z_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float32,
        )
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float32)
        obtained_s_vector, obtained_desc_order = saw(
            z_matrix,
            w_vector,
            is_benefit_z,
        )
        expected_s_vector = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float64,
        )
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_nested_list(self):
        """
        Test the SAW scoring method with nested lists.
        """
        z_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        is_benefit_z = [True, True]
        w_vector = [0.5, 0.5]
        obtained_s_vector, obtained_desc_order = saw(
            z_matrix,
            w_vector,
            is_benefit_z,
        )
        expected_s_vector = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5],
            dtype=np.float64,
        )
        expected_desc_order = True
        np.testing.assert_allclose(obtained_s_vector, expected_s_vector)
        self.assertEqual(obtained_s_vector.dtype, expected_s_vector.dtype)
        self.assertEqual(obtained_desc_order, expected_desc_order)

    def test_missing_element_exception(self):
        """
        Test the SAW scoring method with a missing element.
        """
        z_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75],
            [1.00, 0.00],
        ]
        is_benefit_z = [True, True]
        w_vector = [0.5, 0.5]
        self.assertRaises(
            ValueError,
            saw,
            z_matrix,
            w_vector,
            is_benefit_z,
        )

    def test_over_exception(self):
        """
        Test the SAW scoring method with a value greater than one.
        """
        z_matrix = np.array(
            [
                [0.00, 1.01],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(
            ValueError,
            saw,
            z_matrix,
            w_vector,
            is_benefit_z,
        )

    def test_under_exception(self):
        """
        Test the SAW scoring method with a value less than zero.
        """
        z_matrix = np.array(
            [
                [ 0.00, 1.00],  # noqa: E201
                [-0.25, 0.75],  # noqa: E201
                [ 0.50, 0.50],  # noqa: E201
                [ 0.75, 0.25],  # noqa: E201
                [ 1.00, 0.00],  # noqa: E201
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(
            ValueError,
            saw,
            z_matrix,
            w_vector,
            is_benefit_z,
        )

    def test_w_vector_length_exception(self):
        """
        Test the SAW scoring method with an invalid weight vector length.
        """
        z_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, True]
        w_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self.assertRaises(
            ValueError,
            saw,
            z_matrix,
            w_vector,
            is_benefit_z,
        )

    def test_w_vector_sum_exception(self):
        """
        Test the SAW scoring method with an invalid weight vector sum.
        """
        z_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, True]
        w_vector = np.array([0.5, 0.4], dtype=np.float64)
        self.assertRaises(
            ValueError,
            saw,
            z_matrix,
            w_vector,
            is_benefit_z,
        )

    def test_is_benefit_z_exception(self):
        """
        Test the SAW scoring method with an invalid Boolean list.
        """
        z_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, True, True]
        w_vector = np.array([0.5, 0.5], dtype=np.float64)
        self.assertRaises(
            ValueError,
            saw,
            z_matrix,
            w_vector,
            is_benefit_z,
        )

    def test_mixture_exception(self):
        """
        Test the SAW scoring method with a mixture of benefit and cost
        criteria.
        """
        z_matrix = np.array(
            [
                [0.5, 0.6, 0.3, 0.2, 0.9],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.4, 0.7, 0.8, 0.1],
            ],
            dtype=np.float64,
        )
        is_benefit_z = [True, False, True, True, True]
        w_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        self.assertRaises(
            ValueError,
            saw,
            z_matrix,
            w_vector,
            is_benefit_z,
        )


if __name__ == "__main__":
    unittest.main()

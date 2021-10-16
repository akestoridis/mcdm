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
Test module for the ``weighting/em_method.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np

from mcdm.weighting import em


class TestEm(unittest.TestCase):
    """
    Test class for the ``em`` function of the ``mcdm.weighting`` package.
    """
    def test_linear(self):
        """
        Test the EM weighting method with a linear association.
        """
        z_matrix = np.array(
            [
                [0.000, 0.000, 0.333],
                [0.033, 0.050, 0.267],
                [0.067, 0.100, 0.200],
                [0.100, 0.175, 0.100],
                [0.200, 0.200, 0.067],
                [0.267, 0.225, 0.033],
                [0.333, 0.250, 0.000],
            ],
            dtype=np.float64,
        )
        obtained_w_vector = em(z_matrix)
        expected_w_vector = np.array(
            [0.37406776, 0.25186448, 0.37406776],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_nonlinear(self):
        """
        Test the EM weighting method with a non-linear association.
        """
        z_matrix = np.array(
            [
                [0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.16666667],
                [0.03333333, 0.08333333, 0.00000000],
                [0.03333333, 0.08333333, 0.16666667],
                [0.06666667, 0.16666667, 0.00000000],
                [0.06666667, 0.16666667, 0.16666667],
                [0.10000000, 0.16666667, 0.00000000],
                [0.10000000, 0.16666667, 0.16666667],
                [0.13333333, 0.08333333, 0.00000000],
                [0.13333333, 0.08333333, 0.16666667],
                [0.16666667, 0.00000000, 0.00000000],
                [0.16666667, 0.00000000, 0.16666667],
            ],
            dtype=np.float64,
        )
        obtained_w_vector = em(z_matrix)
        expected_w_vector = np.array(
            [0.20724531, 0.31710188, 0.47565280],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_float32(self):
        """
        Test the EM weighting method with a float32 NumPy array.
        """
        z_matrix = np.array(
            [
                [0.000, 0.000, 0.333],
                [0.033, 0.050, 0.267],
                [0.067, 0.100, 0.200],
                [0.100, 0.175, 0.100],
                [0.200, 0.200, 0.067],
                [0.267, 0.225, 0.033],
                [0.333, 0.250, 0.000],
            ],
            dtype=np.float32,
        )
        obtained_w_vector = em(z_matrix)
        expected_w_vector = np.array(
            [0.37406776, 0.25186448, 0.37406776],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_nested_list(self):
        """
        Test the EM weighting method with a nested list.
        """
        z_matrix = [
            [0.000, 0.000, 0.333],
            [0.033, 0.050, 0.267],
            [0.067, 0.100, 0.200],
            [0.100, 0.175, 0.100],
            [0.200, 0.200, 0.067],
            [0.267, 0.225, 0.033],
            [0.333, 0.250, 0.000],
        ]
        obtained_w_vector = em(z_matrix)
        expected_w_vector = np.array(
            [0.37406776, 0.25186448, 0.37406776],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_missing_element_exception(self):
        """
        Test the EM weighting method with a missing element.
        """
        z_matrix = [
            [0.000, 0.000, 0.333],
            [0.033, 0.050, 0.267],
            [0.067, 0.100, 0.200],
            [0.100, 0.175, 0.100],
            [0.200, 0.200, 0.067],
            [0.267, 0.225],
            [0.333, 0.250, 0.000],
        ]
        self.assertRaises(ValueError, em, z_matrix)

    def test_over_exception(self):
        """
        Test the EM weighting method with a value greater than one.
        """
        z_matrix = np.array(
            [
                [0.000, 0.000, 1.333],
                [0.033, 0.050, 0.267],
                [0.067, 0.100, 0.200],
                [0.100, 0.175, 0.100],
                [0.200, 0.200, 0.067],
                [0.267, 0.225, 0.033],
                [0.333, 0.250, 0.000],
            ],
            dtype=np.float64,
        )
        self.assertRaises(ValueError, em, z_matrix)

    def test_under_exception(self):
        """
        Test the EM weighting method with a value less than zero.
        """
        z_matrix = np.array(
            [
                [ 0.000, 0.000, 0.333],  # noqa: E201
                [-0.033, 0.050, 0.267],  # noqa: E201
                [ 0.067, 0.100, 0.200],  # noqa: E201
                [ 0.100, 0.175, 0.100],  # noqa: E201
                [ 0.200, 0.200, 0.067],  # noqa: E201
                [ 0.267, 0.225, 0.033],  # noqa: E201
                [ 0.333, 0.250, 0.000],  # noqa: E201
            ],
            dtype=np.float64,
        )
        self.assertRaises(ValueError, em, z_matrix)

    def test_sum_exception(self):
        """
        Test the EM weighting method with a column that does not sum to one.
        """
        z_matrix = np.array(
            [
                [0.000, 0.0, 0.333],
                [0.033, 0.2, 0.267],
                [0.067, 0.4, 0.200],
                [0.100, 0.7, 0.100],
                [0.200, 0.8, 0.067],
                [0.267, 0.9, 0.033],
                [0.333, 1.0, 0.000],
            ],
            dtype=np.float64,
        )
        self.assertRaises(ValueError, em, z_matrix)


if __name__ == "__main__":
    unittest.main()

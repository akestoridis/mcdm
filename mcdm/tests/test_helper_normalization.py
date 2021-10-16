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
Test module for the ``helper_normalization.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np

from mcdm import normalize


class TestNormalize(unittest.TestCase):
    """
    Test class for the ``normalize`` function of the ``mcdm`` package.
    """
    def test_none_calculations(self):
        """
        Test the processing of a decision matrix that is already normalized.
        """
        x_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, True, True]
        obtained_z_matrix, obtained_is_benefit_z = normalize(
            x_matrix,
            is_benefit_x,
            None,
        )
        expected_z_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected_is_benefit_z = [True, True, True]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_none_float32(self):
        """
        Test the processing of a float32 NumPy array that is already
        normalized.
        """
        x_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        is_benefit_x = [True, True, True]
        obtained_z_matrix, obtained_is_benefit_z = normalize(
            x_matrix,
            is_benefit_x,
            None,
        )
        expected_z_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected_is_benefit_z = [True, True, True]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_none_nested_list(self):
        """
        Test the processing of a nested list that is already normalized.
        """
        x_matrix = [
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 0.8],
            [0.2, 0.4, 0.6],
            [0.3, 0.7, 0.3],
            [0.6, 0.8, 0.2],
            [0.8, 0.9, 0.1],
            [1.0, 1.0, 0.0],
        ]
        is_benefit_x = [True, True, True]
        obtained_z_matrix, obtained_is_benefit_z = normalize(
            x_matrix,
            is_benefit_x,
            None,
        )
        expected_z_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected_is_benefit_z = [True, True, True]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_none_missing_element_exception(self):
        """
        Test the processing of a nested list with a missing element.
        """
        x_matrix = [
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 0.8],
            [0.2, 0.4, 0.6],
            [0.3, 0.7, 0.3],
            [0.6, 0.8, 0.2],
            [0.8, 0.9],
            [1.0, 1.0, 0.0],
        ]
        is_benefit_x = [True, True, True]
        self.assertRaises(
            ValueError,
            normalize,
            x_matrix,
            is_benefit_x,
            None,
        )

    def test_none_over_exception(self):
        """
        Test the processing of a decision matrix with a value greater than
        one.
        """
        x_matrix = np.array(
            [
                [0.0, 0.0, 1.1],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, True, True]
        self.assertRaises(
            ValueError,
            normalize,
            x_matrix,
            is_benefit_x,
            None,
        )

    def test_none_under_exception(self):
        """
        Test the processing of a decision matrix with a value less than zero.
        """
        x_matrix = np.array(
            [
                [ 0.0, 0.0, 1.0],  # noqa: E201
                [-0.1, 0.2, 0.8],  # noqa: E201
                [ 0.2, 0.4, 0.6],  # noqa: E201
                [ 0.3, 0.7, 0.3],  # noqa: E201
                [ 0.6, 0.8, 0.2],  # noqa: E201
                [ 0.8, 0.9, 0.1],  # noqa: E201
                [ 1.0, 1.0, 0.0],  # noqa: E201
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, True, True]
        self.assertRaises(
            ValueError,
            normalize,
            x_matrix,
            is_benefit_x,
            None,
        )

    def test_none_is_benefit_x_exception(self):
        """
        Test the processing of a decision matrix with an invalid Boolean list.
        """
        x_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, True, True, True]
        self.assertRaises(
            ValueError,
            normalize,
            x_matrix,
            is_benefit_x,
            None,
        )

    def test_unknown_selection_exception(self):
        """
        Test the selection of an unknown normalization method.
        """
        x_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.8],
                [0.2, 0.4, 0.6],
                [0.3, 0.7, 0.3],
                [0.6, 0.8, 0.2],
                [0.8, 0.9, 0.1],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, True, True]
        self.assertRaises(
            ValueError,
            normalize,
            x_matrix,
            is_benefit_x,
            "Unknown",
        )


if __name__ == "__main__":
    unittest.main()

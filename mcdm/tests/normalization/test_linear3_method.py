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
Test module for the ``normalization/linear3_method.py`` file of the ``mcdm``
package.
"""

import unittest

import numpy as np

from mcdm.normalization import linear3


class TestLinear3(unittest.TestCase):
    """
    Test class for the ``linear3`` function of the ``mcdm.normalization``
    package.
    """
    def test_calculations(self):
        """
        Test the calculations of the Linear3 normalization method.
        """
        x_matrix = np.array(
            [
                [4.0, 4.0, 7.0, 7.0],
                [3.0, 3.0, 7.0, 7.0],
                [2.0, 2.0, 7.0, 7.0],
                [1.0, 1.0, 7.0, 7.0],
                [0.0, 0.0, 7.0, 7.0],
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, False, True, False]
        obtained_z_matrix, obtained_is_benefit_z = linear3(
            x_matrix,
            is_benefit_x,
        )
        expected_z_matrix = np.array(
            [
                [0.4, 0.4, 0.2, 0.2],
                [0.3, 0.3, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2],
                [0.1, 0.1, 0.2, 0.2],
                [0.0, 0.0, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
        expected_is_benefit_z = [True, False, True, False]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_float32(self):
        """
        Test the Linear3 normalization method with a float32 NumPy array.
        """
        x_matrix = np.array(
            [
                [4.0, 4.0, 7.0, 7.0],
                [3.0, 3.0, 7.0, 7.0],
                [2.0, 2.0, 7.0, 7.0],
                [1.0, 1.0, 7.0, 7.0],
                [0.0, 0.0, 7.0, 7.0],
            ],
            dtype=np.float32,
        )
        is_benefit_x = [True, False, True, False]
        obtained_z_matrix, obtained_is_benefit_z = linear3(
            x_matrix,
            is_benefit_x,
        )
        expected_z_matrix = np.array(
            [
                [0.4, 0.4, 0.2, 0.2],
                [0.3, 0.3, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2],
                [0.1, 0.1, 0.2, 0.2],
                [0.0, 0.0, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
        expected_is_benefit_z = [True, False, True, False]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_nested_list(self):
        """
        Test the Linear3 normalization method with a nested list.
        """
        x_matrix = [
            [4.0, 4.0, 7.0, 7.0],
            [3.0, 3.0, 7.0, 7.0],
            [2.0, 2.0, 7.0, 7.0],
            [1.0, 1.0, 7.0, 7.0],
            [0.0, 0.0, 7.0, 7.0],
        ]
        is_benefit_x = [True, False, True, False]
        obtained_z_matrix, obtained_is_benefit_z = linear3(
            x_matrix,
            is_benefit_x,
        )
        expected_z_matrix = np.array(
            [
                [0.4, 0.4, 0.2, 0.2],
                [0.3, 0.3, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2],
                [0.1, 0.1, 0.2, 0.2],
                [0.0, 0.0, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
        expected_is_benefit_z = [True, False, True, False]
        np.testing.assert_allclose(obtained_z_matrix, expected_z_matrix)
        self.assertEqual(obtained_z_matrix.dtype, expected_z_matrix.dtype)
        self.assertEqual(obtained_is_benefit_z, expected_is_benefit_z)

    def test_missing_element_exception(self):
        """
        Test the Linear3 normalization method with a missing element.
        """
        x_matrix = [
            [4.0, 4.0, 7.0, 7.0],
            [3.0, 3.0, 7.0, 7.0],
            [2.0, 2.0, 7.0, 7.0],
            [1.0, 1.0, 7.0],
            [0.0, 0.0, 7.0, 7.0],
        ]
        is_benefit_x = [True, False, True, False]
        self.assertRaises(
            ValueError,
            linear3,
            x_matrix,
            is_benefit_x,
        )

    def test_negative_exception(self):
        """
        Test the Linear3 normalization method with a negative value.
        """
        x_matrix = np.array(
            [
                [ 4.0, 4.0, 7.0, 7.0],  # noqa: E201
                [ 3.0, 3.0, 7.0, 7.0],  # noqa: E201
                [-2.0, 2.0, 7.0, 7.0],  # noqa: E201
                [ 1.0, 1.0, 7.0, 7.0],  # noqa: E201
                [ 0.0, 0.0, 7.0, 7.0],  # noqa: E201
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, False, True, False]
        self.assertRaises(
            ValueError,
            linear3,
            x_matrix,
            is_benefit_x,
        )

    def test_zero_constant_exception(self):
        """
        Test the Linear3 normalization method with a zero constant vector.
        """
        x_matrix = np.array(
            [
                [4.0, 4.0, 7.0, 0.0],
                [3.0, 3.0, 7.0, 0.0],
                [2.0, 2.0, 7.0, 0.0],
                [1.0, 1.0, 7.0, 0.0],
                [0.0, 0.0, 7.0, 0.0],
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, False, True, False]
        self.assertRaises(
            ValueError,
            linear3,
            x_matrix,
            is_benefit_x,
        )

    def test_is_benefit_x_exception(self):
        """
        Test the Linear3 normalization method with an invalid Boolean list.
        """
        x_matrix = np.array(
            [
                [4.0, 4.0, 7.0, 7.0],
                [3.0, 3.0, 7.0, 7.0],
                [2.0, 2.0, 7.0, 7.0],
                [1.0, 1.0, 7.0, 7.0],
                [0.0, 0.0, 7.0, 7.0],
            ],
            dtype=np.float64,
        )
        is_benefit_x = [True, False, True, False, True]
        self.assertRaises(
            ValueError,
            linear3,
            x_matrix,
            is_benefit_x,
        )


if __name__ == "__main__":
    unittest.main()

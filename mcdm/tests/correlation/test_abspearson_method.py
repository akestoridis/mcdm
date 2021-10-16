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
Test module for the ``correlation/abspearson_method.py`` file of the ``mcdm``
package.
"""

import unittest

import numpy as np

from mcdm.correlation import abspearson


class TestAbspearson(unittest.TestCase):
    """
    Test class for the ``abspearson`` function of the ``mcdm.correlation``
    package.
    """
    def test_linear(self):
        """
        Test the AbsPearson correlation method with a linear association.
        """
        z_matrix = np.array(
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
        obtained_corr_matrix = abspearson(z_matrix)
        expected_corr_matrix = np.array(
            [
                [1.0000000, 0.9314381, 0.9314381],
                [0.9314381, 1.0000000, 1.0000000],
                [0.9314381, 1.0000000, 1.0000000],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_corr_matrix, expected_corr_matrix)
        self.assertEqual(
            obtained_corr_matrix.dtype,
            expected_corr_matrix.dtype,
        )

    def test_nonlinear(self):
        """
        Test the AbsPearson correlation method with a non-linear association.
        """
        z_matrix = np.array(
            [
                [0.0, 0.0, 0.0],
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
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        obtained_corr_matrix = abspearson(z_matrix)
        expected_corr_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_corr_matrix, expected_corr_matrix)
        self.assertEqual(
            obtained_corr_matrix.dtype,
            expected_corr_matrix.dtype,
        )

    def test_float32(self):
        """
        Test the AbsPearson correlation method with a float32 NumPy array.
        """
        z_matrix = np.array(
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
        obtained_corr_matrix = abspearson(z_matrix)
        expected_corr_matrix = np.array(
            [
                [1.0000000, 0.9314381, 0.9314381],
                [0.9314381, 1.0000000, 1.0000000],
                [0.9314381, 1.0000000, 1.0000000],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_corr_matrix, expected_corr_matrix)
        self.assertEqual(
            obtained_corr_matrix.dtype,
            expected_corr_matrix.dtype,
        )

    def test_nested_list(self):
        """
        Test the AbsPearson correlation method with a nested list.
        """
        z_matrix = [
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 0.8],
            [0.2, 0.4, 0.6],
            [0.3, 0.7, 0.3],
            [0.6, 0.8, 0.2],
            [0.8, 0.9, 0.1],
            [1.0, 1.0, 0.0],
        ]
        obtained_corr_matrix = abspearson(z_matrix)
        expected_corr_matrix = np.array(
            [
                [1.0000000, 0.9314381, 0.9314381],
                [0.9314381, 1.0000000, 1.0000000],
                [0.9314381, 1.0000000, 1.0000000],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(obtained_corr_matrix, expected_corr_matrix)
        self.assertEqual(
            obtained_corr_matrix.dtype,
            expected_corr_matrix.dtype,
        )

    def test_missing_element_exception(self):
        """
        Test the AbsPearson correlation method with a missing element.
        """
        z_matrix = [
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 0.8],
            [0.2, 0.4, 0.6],
            [0.3, 0.7, 0.3],
            [0.6, 0.8, 0.2],
            [0.8, 0.9],
            [1.0, 1.0, 0.0],
        ]
        self.assertRaises(ValueError, abspearson, z_matrix)


if __name__ == "__main__":
    unittest.main()

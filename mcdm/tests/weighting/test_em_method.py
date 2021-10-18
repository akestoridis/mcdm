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
Test script for the ``weighting/em_method.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np
from mcdm.weighting import em

from ..helper_testing import (
    get_matrix41,
    get_matrix42,
    get_matrix43,
    get_matrix44,
    get_matrix45,
    get_matrix46,
)


class TestEm(unittest.TestCase):
    """
    Test class for the ``em`` function of the ``mcdm.weighting`` package.
    """
    def test_linear(self):
        """
        Test the EM weighting method with a linear association.
        """
        obtained_w_vector = em(np.array(get_matrix41(), dtype=np.float64))
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
        obtained_w_vector = em(np.array(get_matrix42(), dtype=np.float64))
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
        obtained_w_vector = em(np.array(get_matrix41(), dtype=np.float32))
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
        obtained_w_vector = em(get_matrix41())
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
        self.assertRaises(ValueError, em, get_matrix43())

    def test_over_exception(self):
        """
        Test the EM weighting method with a value greater than one.
        """
        self.assertRaises(
            ValueError,
            em,
            np.array(get_matrix44(), dtype=np.float64),
        )

    def test_under_exception(self):
        """
        Test the EM weighting method with a value less than zero.
        """
        self.assertRaises(
            ValueError,
            em,
            np.array(get_matrix45(), dtype=np.float64),
        )

    def test_sum_exception(self):
        """
        Test the EM weighting method with a column that does not sum to one.
        """
        self.assertRaises(
            ValueError,
            em,
            np.array(get_matrix46(), dtype=np.float64),
        )


if __name__ == "__main__":
    unittest.main()

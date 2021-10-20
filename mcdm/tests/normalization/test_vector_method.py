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
Test script for the ``normalization/vector_method.py`` file of the ``mcdm``
package.
"""

import unittest

import numpy as np
from mcdm.normalization import vector

from ..helper_testing import (
    ExtendedTestCase,
    get_matrix29,
    get_matrix30,
    get_matrix31,
    get_matrix32,
    get_matrix33,
)


class TestVector(ExtendedTestCase):
    """
    Test class for the ``vector`` function of the ``mcdm.normalization``
    package.
    """
    def test_calculations(self):
        """
        Test the calculations of the Vector normalization method.
        """
        obtained_z_matrix, obtained_is_benefit_z = vector(
            np.array(get_matrix29(), dtype=np.float64),
            [True, False, True, False],
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix30(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, False, True, False])

    def test_float32(self):
        """
        Test the Vector normalization method with a float32 NumPy array.
        """
        obtained_z_matrix, obtained_is_benefit_z = vector(
            np.array(get_matrix29(), dtype=np.float32),
            [True, False, True, False],
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix30(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, False, True, False])

    def test_nested_list(self):
        """
        Test the Vector normalization method with a nested list.
        """
        obtained_z_matrix, obtained_is_benefit_z = vector(
            get_matrix29(),
            [True, False, True, False],
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix30(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, False, True, False])

    def test_missing_element_exception(self):
        """
        Test the Vector normalization method with a missing element.
        """
        self.assertRaises(
            ValueError,
            vector,
            get_matrix31(),
            [True, False, True, False],
        )

    def test_negative_exception(self):
        """
        Test the Vector normalization method with a negative value.
        """
        self.assertRaises(
            ValueError,
            vector,
            np.array(get_matrix32(), dtype=np.float64),
            [True, False, True, False],
        )

    def test_zero_constant_exception(self):
        """
        Test the Vector normalization method with a zero constant vector.
        """
        self.assertRaises(
            ValueError,
            vector,
            np.array(get_matrix33(), dtype=np.float64),
            [True, False, True, False],
        )

    def test_is_benefit_x_exception(self):
        """
        Test the Vector normalization method with an invalid Boolean list.
        """
        self.assertRaises(
            ValueError,
            vector,
            np.array(get_matrix29(), dtype=np.float64),
            [True, False, True, False, True],
        )


if __name__ == "__main__":
    unittest.main()

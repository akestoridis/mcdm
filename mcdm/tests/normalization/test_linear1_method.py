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
Test script for the ``normalization/linear1_method.py`` file of the ``mcdm``
package.
"""

import unittest

import numpy as np
from mcdm.normalization import linear1

from ..helper_testing import (
    ExtendedTestCase,
    get_matrix04,
    get_matrix14,
    get_matrix15,
    get_matrix16,
    get_matrix17,
    get_matrix18,
)


class TestLinear1(ExtendedTestCase):
    """
    Test class for the ``linear1`` function of the ``mcdm.normalization``
    package.
    """
    def test_calculations(self):
        """
        Test the calculations of the Linear1 normalization method.
        """
        obtained_z_matrix, obtained_is_benefit_z = linear1(
            np.array(get_matrix04(), dtype=np.float64),
            [True, False, True, False],
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix14(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, True, True, True])

    def test_float32(self):
        """
        Test the Linear1 normalization method with a float32 NumPy array.
        """
        obtained_z_matrix, obtained_is_benefit_z = linear1(
            np.array(get_matrix04(), dtype=np.float32),
            [True, False, True, False],
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix14(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, True, True, True])

    def test_nested_list(self):
        """
        Test the Linear1 normalization method with a nested list.
        """
        obtained_z_matrix, obtained_is_benefit_z = linear1(
            get_matrix04(),
            [True, False, True, False],
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix14(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, True, True, True])

    def test_missing_element_exception(self):
        """
        Test the Linear1 normalization method with a missing element.
        """
        self.assertRaises(
            ValueError,
            linear1,
            get_matrix15(),
            [True, False, True, False],
        )

    def test_negative_exception(self):
        """
        Test the Linear1 normalization method with a negative value.
        """
        self.assertRaises(
            ValueError,
            linear1,
            np.array(get_matrix16(), dtype=np.float64),
            [True, False, True, False],
        )

    def test_zero_benefit_exception(self):
        """
        Test the Linear1 normalization method with a zero benefit vector.
        """
        self.assertRaises(
            ValueError,
            linear1,
            np.array(get_matrix17(), dtype=np.float64),
            [True, False, True, False],
        )

    def test_zero_cost_exception(self):
        """
        Test the Linear1 normalization method with a zero cost vector.
        """
        self.assertRaises(
            ValueError,
            linear1,
            np.array(get_matrix18(), dtype=np.float64),
            [True, False, True, False],
        )

    def test_is_benefit_x_exception(self):
        """
        Test the Linear1 normalization method with an invalid Boolean list.
        """
        self.assertRaises(
            ValueError,
            linear1,
            np.array(get_matrix04(), dtype=np.float64),
            [True, False, True, False, True],
        )


if __name__ == "__main__":
    unittest.main()

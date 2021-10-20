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
Test script for the ``helper_normalization.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np
from mcdm import normalize

from .helper_testing import (
    ExtendedTestCase,
    get_matrix01,
    get_matrix11,
    get_matrix12,
    get_matrix13,
)


class TestNormalize(ExtendedTestCase):
    """
    Test class for the ``normalize`` function of the ``mcdm`` package.
    """
    def test_none_calculations(self):
        """
        Test the processing of a decision matrix that is already normalized.
        """
        obtained_z_matrix, obtained_is_benefit_z = normalize(
            np.array(get_matrix01(), dtype=np.float64),
            [True, True, True],
            None,
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix01(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, True, True])

    def test_none_float32(self):
        """
        Test the processing of a float32 NumPy array that is already
        normalized.
        """
        obtained_z_matrix, obtained_is_benefit_z = normalize(
            np.array(get_matrix01(), dtype=np.float32),
            [True, True, True],
            None,
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix01(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, True, True])

    def test_none_nested_list(self):
        """
        Test the processing of a nested list that is already normalized.
        """
        obtained_z_matrix, obtained_is_benefit_z = normalize(
            get_matrix01(),
            [True, True, True],
            None,
        )
        self.assertAlmostEqualArrays(
            obtained_z_matrix,
            np.array(get_matrix01(), dtype=np.float64),
        )
        self.assertEqual(obtained_is_benefit_z, [True, True, True])

    def test_none_missing_element_exception(self):
        """
        Test the processing of a nested list with a missing element.
        """
        self.assertRaises(
            ValueError,
            normalize,
            get_matrix11(),
            [True, True, True],
            None,
        )

    def test_none_over_exception(self):
        """
        Test the processing of a decision matrix with a value greater than
        one.
        """
        self.assertRaises(
            ValueError,
            normalize,
            np.array(get_matrix12(), dtype=np.float64),
            [True, True, True],
            None,
        )

    def test_none_under_exception(self):
        """
        Test the processing of a decision matrix with a value less than zero.
        """
        self.assertRaises(
            ValueError,
            normalize,
            np.array(get_matrix13(), dtype=np.float64),
            [True, True, True],
            None,
        )

    def test_none_is_benefit_x_exception(self):
        """
        Test the processing of a decision matrix with an invalid Boolean list.
        """
        self.assertRaises(
            ValueError,
            normalize,
            np.array(get_matrix01(), dtype=np.float64),
            [True, True, True, True],
            None,
        )

    def test_unknown_selection_exception(self):
        """
        Test the selection of an unknown normalization method.
        """
        self.assertRaises(
            ValueError,
            normalize,
            np.array(get_matrix01(), dtype=np.float64),
            [True, True, True],
            "Unknown",
        )


if __name__ == "__main__":
    unittest.main()

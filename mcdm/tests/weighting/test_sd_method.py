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
Test script for the ``weighting/sd_method.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np
from mcdm.weighting import sd

from ..helper_testing import (
    ExtendedTestCase,
    get_matrix01,
    get_matrix02,
    get_matrix11,
    get_matrix12,
    get_matrix13,
    get_vector17,
    get_vector20,
)


class TestSd(ExtendedTestCase):
    """
    Test class for the ``sd`` function of the ``mcdm.weighting`` package.
    """
    def test_linear(self):
        """
        Test the SD weighting method with a linear association.
        """
        self.assertAlmostEqualArrays(
            sd(np.array(get_matrix01(), dtype=np.float64)),
            np.array(get_vector17(), dtype=np.float64),
        )

    def test_nonlinear(self):
        """
        Test the SD weighting method with a non-linear association.
        """
        self.assertAlmostEqualArrays(
            sd(np.array(get_matrix02(), dtype=np.float64)),
            np.array(get_vector20(), dtype=np.float64),
        )

    def test_float32(self):
        """
        Test the SD weighting method with a float32 NumPy array.
        """
        self.assertAlmostEqualArrays(
            sd(np.array(get_matrix01(), dtype=np.float32)),
            np.array(get_vector17(), dtype=np.float64),
        )

    def test_nested_list(self):
        """
        Test the SD weighting method with a nested list.
        """
        self.assertAlmostEqualArrays(
            sd(get_matrix01()),
            np.array(get_vector17(), dtype=np.float64),
        )

    def test_missing_element_exception(self):
        """
        Test the SD weighting method with a missing element.
        """
        self.assertRaises(ValueError, sd, get_matrix11())

    def test_over_exception(self):
        """
        Test the SD weighting method with a value greater than one.
        """
        self.assertRaises(
            ValueError,
            sd,
            np.array(get_matrix12(), dtype=np.float64),
        )

    def test_under_exception(self):
        """
        Test the SD weighting method with a value less than zero.
        """
        self.assertRaises(
            ValueError,
            sd,
            np.array(get_matrix13(), dtype=np.float64),
        )


if __name__ == "__main__":
    unittest.main()

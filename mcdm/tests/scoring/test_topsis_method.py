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
Test script for the ``scoring/topsis_method.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np
from mcdm.scoring import topsis

from ..helper_testing import (
    ExtendedTestCase,
    get_matrix03,
    get_matrix06,
    get_matrix10,
    get_matrix47,
    get_matrix48,
    get_vector03,
    get_vector04,
    get_vector05,
    get_vector06,
    get_vector07,
    get_vector11,
    get_vector12,
    get_vector13,
)


class TestTopsis(ExtendedTestCase):
    """
    Test class for the ``topsis`` function of the ``mcdm.scoring`` package.
    """
    def test_balanced(self):
        """
        Test the TOPSIS scoring method with a balanced decision matrix.
        """
        obtained_s_vector, obtained_desc_order = topsis(
            np.array(get_matrix03(), dtype=np.float64),
            np.array(get_vector05(), dtype=np.float64),
            [True, True],
        )
        self.assertAlmostEqualArrays(
            obtained_s_vector,
            np.array(get_vector06(), dtype=np.float64),
        )
        self.assertEqual(obtained_desc_order, True)

    def test_simple_benefit(self):
        """
        Test the TOPSIS scoring method with simple benefit criteria.
        """
        obtained_s_vector, obtained_desc_order = topsis(
            np.array(get_matrix06(), dtype=np.float64),
            np.array(get_vector07(), dtype=np.float64),
            [True, True, True, True, True],
        )
        self.assertAlmostEqualArrays(
            obtained_s_vector,
            np.array(get_vector11(), dtype=np.float64),
        )
        self.assertEqual(obtained_desc_order, True)

    def test_simple_cost(self):
        """
        Test the TOPSIS scoring method with simple cost criteria.
        """
        obtained_s_vector, obtained_desc_order = topsis(
            np.array(get_matrix06(), dtype=np.float64),
            np.array(get_vector07(), dtype=np.float64),
            [False, False, False, False, False],
        )
        self.assertAlmostEqualArrays(
            obtained_s_vector,
            np.array(get_vector12(), dtype=np.float64),
        )
        self.assertEqual(obtained_desc_order, True)

    def test_simple_mixture(self):
        """
        Test the TOPSIS scoring method with a mixture of benefit and cost
        criteria.
        """
        obtained_s_vector, obtained_desc_order = topsis(
            np.array(get_matrix06(), dtype=np.float64),
            np.array(get_vector07(), dtype=np.float64),
            [True, False, True, True, True],
        )
        self.assertAlmostEqualArrays(
            obtained_s_vector,
            np.array(get_vector13(), dtype=np.float64),
        )
        self.assertEqual(obtained_desc_order, True)

    def test_float32(self):
        """
        Test the TOPSIS scoring method with float32 NumPy arrays.
        """
        obtained_s_vector, obtained_desc_order = topsis(
            np.array(get_matrix03(), dtype=np.float32),
            np.array(get_vector05(), dtype=np.float32),
            [True, True],
        )
        self.assertAlmostEqualArrays(
            obtained_s_vector,
            np.array(get_vector06(), dtype=np.float64),
        )
        self.assertEqual(obtained_desc_order, True)

    def test_nested_list(self):
        """
        Test the TOPSIS scoring method with nested lists.
        """
        obtained_s_vector, obtained_desc_order = topsis(
            get_matrix03(),
            get_vector05(),
            [True, True],
        )
        self.assertAlmostEqualArrays(
            obtained_s_vector,
            np.array(get_vector06(), dtype=np.float64),
        )
        self.assertEqual(obtained_desc_order, True)

    def test_missing_element_exception(self):
        """
        Test the TOPSIS scoring method with a missing element.
        """
        self.assertRaises(
            ValueError,
            topsis,
            get_matrix10(),
            get_vector05(),
            [True, True],
        )

    def test_over_exception(self):
        """
        Test the TOPSIS scoring method with a value greater than one.
        """
        self.assertRaises(
            ValueError,
            topsis,
            np.array(get_matrix47(), dtype=np.float64),
            np.array(get_vector05(), dtype=np.float64),
            [True, True],
        )

    def test_under_exception(self):
        """
        Test the TOPSIS scoring method with a value less than zero.
        """
        self.assertRaises(
            ValueError,
            topsis,
            np.array(get_matrix48(), dtype=np.float64),
            np.array(get_vector05(), dtype=np.float64),
            [True, True],
        )

    def test_w_vector_length_exception(self):
        """
        Test the TOPSIS scoring method with an invalid weight vector length.
        """
        self.assertRaises(
            ValueError,
            topsis,
            np.array(get_matrix03(), dtype=np.float64),
            np.array(get_vector03(), dtype=np.float64),
            [True, True],
        )

    def test_w_vector_sum_exception(self):
        """
        Test the TOPSIS scoring method with an invalid weight vector sum.
        """
        self.assertRaises(
            ValueError,
            topsis,
            np.array(get_matrix03(), dtype=np.float64),
            np.array(get_vector04(), dtype=np.float64),
            [True, True],
        )

    def test_is_benefit_z_exception(self):
        """
        Test the TOPSIS scoring method with an invalid Boolean list.
        """
        self.assertRaises(
            ValueError,
            topsis,
            np.array(get_matrix03(), dtype=np.float64),
            np.array(get_vector05(), dtype=np.float64),
            [True, True, True],
        )


if __name__ == "__main__":
    unittest.main()

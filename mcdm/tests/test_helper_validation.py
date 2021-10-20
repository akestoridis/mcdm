#!/usr/bin/env python3

# Copyright (c) 2021 Dimitrios-Georgios Akestoridis
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
Test script for the ``helper_validation.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np
from mcdm.helper_validation import (
    check_normalization_input,
    check_scoring_input,
    check_weighting_input,
)

from .helper_testing import (
    ExtendedTestCase,
    get_matrix01,
    get_matrix03,
    get_vector05,
)


class TestCheckScoringInput(ExtendedTestCase):
    """
    Test class for the ``check_scoring_input`` function of the
    ``mcdm.helper_validation`` module.
    """
    def test_unknown_method_exception(self):
        """
        Test the validation of the input data for an unknown scoring method.
        """
        self.assertRaises(
            ValueError,
            check_scoring_input,
            np.array(get_matrix03(), dtype=np.float64),
            np.array(get_vector05(), dtype=np.float64),
            [True, True],
            "Unknown",
        )


class TestCheckWeightingInput(ExtendedTestCase):
    """
    Test class for the ``check_weighting_input`` function of the
    ``mcdm.helper_validation`` module.
    """
    def test_unknown_method_exception(self):
        """
        Test the validation of the input data for an unknown weighting method.
        """
        self.assertRaises(
            ValueError,
            check_weighting_input,
            np.array(get_matrix01(), dtype=np.float64),
            "",
            "Unknown",
        )


class TestCheckNormalizationInput(ExtendedTestCase):
    """
    Test class for the ``check_normalization_input`` function of the
    ``mcdm.helper_validation`` module.
    """
    def test_unknown_method_exception(self):
        """
        Test the validation of the input data for an unknown normalization
        method.
        """
        self.assertRaises(
            ValueError,
            check_normalization_input,
            np.array(get_matrix01(), dtype=np.float64),
            [True, True, True],
            "Unknown",
        )


if __name__ == "__main__":
    unittest.main()

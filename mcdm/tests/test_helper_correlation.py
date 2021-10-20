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
Test script for the ``helper_correlation.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np
from mcdm import correlate

from .helper_testing import (
    ExtendedTestCase,
    get_matrix01,
)


class TestCorrelate(ExtendedTestCase):
    """
    Test class for the ``correlate`` function of the ``mcdm`` package.
    """
    def test_unknown_selection_exception(self):
        """
        Test the selection of an unknown correlation method.
        """
        self.assertRaises(
            ValueError,
            correlate,
            np.array(get_matrix01(), dtype=np.float64),
            "Unknown",
        )


if __name__ == "__main__":
    unittest.main()

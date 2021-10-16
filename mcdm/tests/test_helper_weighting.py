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
Test module for the ``helper_weighting.py`` file of the ``mcdm`` package.
"""

import unittest

import numpy as np

from mcdm import weigh


class TestWeigh(unittest.TestCase):
    """
    Test class for the ``weigh`` function of the ``mcdm`` package.
    """
    def test_unknown_selection_exception(self):
        """
        Test the selection of an unknown weighting method.
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
        self.assertRaises(ValueError, weigh, z_matrix, "Unknown")


if __name__ == "__main__":
    unittest.main()

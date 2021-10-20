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
Test script for the integration of the ``mcdm`` package.
"""

import os
import unittest

import mcdm

from .helper_testing import (
    ExtendedTestCase,
    get_ranking01,
    get_ranking16,
    get_ranking17,
    get_ranking18,
    get_ranking19,
    get_ranking20,
    get_ranking21,
    get_ranking22,
    get_ranking23,
    get_vector01,
)


DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TestMcdm(ExtendedTestCase):
    """
    Test class for the integration of the ``mcdm`` package.
    """
    def test_default(self):
        """
        Test the integration with the default selections.
        """
        x_matrix, _alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example03.tsv"),
            delimiter="\t",
        )
        self.assertAlmostEqualRankings(mcdm.rank(x_matrix), get_ranking01())

    def test_mew(self):
        """
        Test the integration with the MEW scoring method.
        """
        x_matrix, _alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example03.tsv"),
            delimiter="\t",
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(x_matrix, s_method="MEW"),
            get_ranking17(),
        )

    def test_topsis_w(self):
        """
        Test the integration with the TOPSIS scoring method and predefined
        weights.
        """
        x_matrix, _alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example03.tsv"),
            delimiter="\t",
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(x_matrix, w_vector=get_vector01(), s_method="TOPSIS"),
            get_ranking18(),
        )

    def test_topsis_is_benefit_x(self):
        """
        Test the integration with the TOPSIS scoring method and a mixture of
        benefit and cost criteria.
        """
        x_matrix, _alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example03.tsv"),
            delimiter="\t",
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(
                x_matrix,
                is_benefit_x=[True, False],
                s_method="TOPSIS",
            ),
            get_ranking19(),
        )

    def test_topsis_sd_vector(self):
        """
        Test the integration with the TOPSIS scoring method, the SD weighting
        method, and the Vector normalization method.
        """
        x_matrix, alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example08.tsv"),
            delimiter="\t",
            labeled_rows=True,
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(
                x_matrix,
                alt_names=alt_names,
                n_method="Vector",
                w_method="SD",
                s_method="TOPSIS",
            ),
            get_ranking20(),
        )

    def test_saw_critic_linear2(self):
        """
        Test the integration with the SAW scoring method, the CRITIC weighting
        method, and the Linear2 normalization method.
        """
        x_matrix, alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example08.tsv"),
            delimiter="\t",
            labeled_rows=True,
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(
                x_matrix,
                alt_names=alt_names,
                n_method="Linear2",
                w_method="CRITIC",
                s_method="SAW",
            ),
            get_ranking21(),
        )

    def test_mtopsis_em_linear3(self):
        """
        Test the integration with the mTOPSIS scoring method, the EM weighting
        method, and the Linear3 normalization method.
        """
        x_matrix, alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example08.tsv"),
            delimiter="\t",
            labeled_rows=True,
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(
                x_matrix,
                alt_names=alt_names,
                n_method="Linear3",
                w_method="EM",
                s_method="mTOPSIS",
            ),
            get_ranking22(),
        )

    def test_mew_vic_linear1(self):
        """
        Test the integration with the MEW scoring method, the VIC weighting
        method, and the Linear1 normalization method.
        """
        x_matrix, alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example08.tsv"),
            delimiter="\t",
            labeled_rows=True,
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(
                x_matrix,
                alt_names=alt_names,
                n_method="Linear1",
                w_method="VIC",
                s_method="MEW",
            ),
            get_ranking23(),
        )

    def test_mew_vic(self):
        """
        Test the integration with the MEW scoring method and VIC weighting
        method.
        """
        x_matrix, alt_names = mcdm.load(
            os.path.join(DIR_PATH, "data", "example09.tsv"),
            delimiter="\t",
            skiprows=1,
            labeled_rows=True,
        )
        self.assertAlmostEqualRankings(
            mcdm.rank(
                x_matrix,
                alt_names=alt_names,
                w_method="VIC",
                s_method="MEW",
            ),
            get_ranking16(),
        )


if __name__ == "__main__":
    unittest.main()

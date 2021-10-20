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
Test script for the ``main.py`` file of the ``mcdm`` package.
"""

import os
import unittest

import numpy as np
from mcdm import (
    load,
    rank,
)

from .helper_testing import (
    ExtendedTestCase,
    get_labels01,
    get_labels02,
    get_labels03,
    get_labels04,
    get_labels05,
    get_matrix01,
    get_matrix02,
    get_matrix03,
    get_matrix04,
    get_matrix05,
    get_matrix06,
    get_matrix07,
    get_matrix08,
    get_matrix09,
    get_matrix10,
    get_ranking01,
    get_ranking02,
    get_ranking03,
    get_ranking04,
    get_ranking05,
    get_ranking06,
    get_ranking07,
    get_ranking08,
    get_ranking09,
    get_ranking10,
    get_ranking11,
    get_ranking12,
    get_ranking13,
    get_ranking14,
    get_ranking15,
    get_ranking16,
    get_vector01,
    get_vector02,
    get_vector03,
    get_vector04,
)


DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TestRank(ExtendedTestCase):
    """
    Test class for the ``rank`` function of the ``mcdm`` package.
    """
    def test_default(self):
        """
        Test the ranking of alternatives with the default selections.
        """
        self.assertAlmostEqualRankings(rank(get_matrix03()), get_ranking01())

    def test_default_float64(self):
        """
        Test the ranking of alternatives with the default selections and a
        float64 NumPy array.
        """
        self.assertAlmostEqualRankings(
            rank(np.array(get_matrix03(), dtype=np.float64)),
            get_ranking01(),
        )

    def test_default_float32(self):
        """
        Test the ranking of alternatives with the default selections and a
        float32 NumPy array.
        """
        self.assertAlmostEqualRankings(
            rank(np.array(get_matrix03(), dtype=np.float32)),
            get_ranking01(),
        )

    def test_default_w_desc_order(self):
        """
        Test the ranking of alternatives with the default selections,
        predefined weights, and benefit criteria.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix03(),
                is_benefit_x=[True, True],
                w_vector=get_vector01(),
            ),
            get_ranking02(),
        )

    def test_default_w_asc_order(self):
        """
        Test the ranking of alternatives with the default selections,
        predefined weights, and cost criteria.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix03(),
                is_benefit_x=[False, False],
                w_vector=get_vector01(),
            ),
            get_ranking03(),
        )

    def test_saw_mw_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the MW
        weighting method, and the Linear1 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear1",
                w_method="MW",
                s_method="SAW",
            ),
            get_ranking04(),
        )

    def test_saw_sd_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the SD
        weighting method, and the Linear1 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear1",
                w_method="SD",
                s_method="SAW",
            ),
            get_ranking05(),
        )

    def test_saw_vic_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the VIC
        weighting method, and the Linear1 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear1",
                w_method="VIC",
                s_method="SAW",
            ),
            get_ranking06(),
        )

    def test_saw_vic_abspearson_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the
        VIC.AbsPearson weighting method, and the Linear1 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear1",
                c_method="AbsPearson",
                w_method="VIC",
                s_method="SAW",
            ),
            get_ranking07(),
        )

    def test_mew_vic_linear1(self):
        """
        Test the ranking of alternatives with the MEW scoring method, the VIC
        weighting method, and the Linear1 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear1",
                w_method="VIC",
                s_method="MEW",
            ),
            get_ranking08(),
        )

    def test_saw_critic_linear2(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the
        CRITIC weighting method, and the Linear2 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear2",
                w_method="CRITIC",
                s_method="SAW",
            ),
            get_ranking09(),
        )

    def test_saw_critic_dcor_linear2(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the
        CRITIC.dCor weighting method, and the Linear2 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear2",
                c_method="dCor",
                w_method="CRITIC",
                s_method="SAW",
            ),
            get_ranking10(),
        )

    def test_topsis_em_linear3(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method, the
        EM weighting method, and the Linear3 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear3",
                w_method="EM",
                s_method="TOPSIS",
            ),
            get_ranking11(),
        )

    def test_mtopsis_em_linear3(self):
        """
        Test the ranking of alternatives with the mTOPSIS scoring method, the
        EM weighting method, and the Linear3 normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Linear3",
                w_method="EM",
                s_method="mTOPSIS",
            ),
            get_ranking12(),
        )

    def test_topsis_w_vector(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method,
        predefined weights, and the Vector normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Vector",
                w_vector=get_vector02(),
                s_method="TOPSIS",
            ),
            get_ranking13(),
        )

    def test_mtopsis_w_vector(self):
        """
        Test the ranking of alternatives with the mTOPSIS scoring method,
        predefined weights, and the Vector normalization method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix07(),
                is_benefit_x=[True, False, False, True],
                n_method="Vector",
                w_vector=get_vector02(),
                s_method="mTOPSIS",
            ),
            get_ranking14(),
        )

    def test_saw_critic(self):
        """
        Test the ranking of alternatives with the SAW scoring method and the
        CRITIC weighting method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix09(),
                alt_names=get_labels04(),
                is_benefit_x=[True, True, True],
                w_method="CRITIC",
                s_method="SAW",
            ),
            get_ranking15(),
        )

    def test_mew_vic(self):
        """
        Test the ranking of alternatives with the MEW scoring method and the
        VIC weighting method.
        """
        self.assertAlmostEqualRankings(
            rank(
                get_matrix09(),
                alt_names=get_labels04(),
                is_benefit_x=[True, True, True],
                w_method="VIC",
                s_method="MEW",
            ),
            get_ranking16(),
        )

    def test_topsis_w_vector_float64(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method,
        predefined weights, the Vector normalization method, and float64 NumPy
        arrays.
        """
        self.assertAlmostEqualRankings(
            rank(
                np.array(get_matrix07(), dtype=np.float64),
                is_benefit_x=[True, False, False, True],
                n_method="Vector",
                w_vector=np.array(get_vector02(), dtype=np.float64),
                s_method="TOPSIS",
            ),
            get_ranking13(),
        )

    def test_topsis_w_vector_float32(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method,
        predefined weights, the Vector normalization method, and float32 NumPy
        arrays.
        """
        self.assertAlmostEqualRankings(
            rank(
                np.array(get_matrix07(), dtype=np.float32),
                is_benefit_x=[True, False, False, True],
                n_method="Vector",
                w_vector=np.array(get_vector02(), dtype=np.float32),
                s_method="TOPSIS",
            ),
            get_ranking13(),
        )

    def test_missing_element_exception(self):
        """
        Test the ranking of alternatives with a missing element.
        """
        self.assertRaises(
            ValueError,
            rank,
            get_matrix10(),
            alt_names=get_labels05(),
            is_benefit_x=[True, True],
        )

    def test_alt_names_exception(self):
        """
        Test the ranking of alternatives with an invalid list of names.
        """
        self.assertRaises(
            ValueError,
            rank,
            get_matrix03(),
            alt_names=get_labels02(),
            is_benefit_x=[True, True],
        )

    def test_is_benefit_x_exception(self):
        """
        Test the ranking of alternatives with an invalid Boolean list.
        """
        self.assertRaises(
            ValueError,
            rank,
            get_matrix03(),
            alt_names=get_labels05(),
            is_benefit_x=[True, True, True],
        )

    def test_w_vector_length_exception(self):
        """
        Test the ranking of alternatives with an invalid weight vector length.
        """
        self.assertRaises(
            ValueError,
            rank,
            get_matrix03(),
            alt_names=get_labels05(),
            is_benefit_x=[True, True],
            w_vector=get_vector03(),
        )

    def test_w_vector_sum_exception(self):
        """
        Test the ranking of alternatives with an invalid weight vector sum.
        """
        self.assertRaises(
            ValueError,
            rank,
            get_matrix03(),
            alt_names=get_labels05(),
            is_benefit_x=[True, True],
            w_vector=get_vector04(),
        )


class TestLoad(ExtendedTestCase):
    """
    Test class for the ``load`` function of the ``mcdm`` package.
    """
    def test_example01(self):
        """
        Test the loading of a matrix from a CSV file with the default
        parameter values.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example01.csv"),
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix01(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, None)

    def test_example02(self):
        """
        Test the loading of a matrix from a CSV file that contains column and
        row labels.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example02.csv"),
            skiprows=1,
            labeled_rows=True,
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix02(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, get_labels01())

    def test_example03(self):
        """
        Test the loading of a matrix from a TSV file that does not contain any
        labels.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example03.tsv"),
            delimiter="\t",
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix03(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, None)

    def test_example04(self):
        """
        Test the loading of a matrix from a TSV file that contains a
        single-line comment.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example04.tsv"),
            delimiter="\t",
            skiprows=1,
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix04(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, None)

    def test_example05(self):
        """
        Test the loading of a matrix from a CSV file that contains row labels.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example05.csv"),
            labeled_rows=True,
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix05(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, get_labels02())

    def test_example06(self):
        """
        Test the loading of a matrix from a CSV file that contains a
        single-line comment.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example06.csv"),
            skiprows=1,
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix06(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, None)

    def test_example07(self):
        """
        Test the loading of a matrix from a CSV file that contains a
        multi-line comment.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example07.csv"),
            skiprows=3,
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix07(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, None)

    def test_example08(self):
        """
        Test the loading of a matrix from a TSV file that contains row labels.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example08.tsv"),
            delimiter="\t",
            labeled_rows=True,
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix08(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, get_labels03())

    def test_example09(self):
        """
        Test the loading of a matrix from a large TSV file.
        """
        obtained_matrix, obtained_row_labels = load(
            os.path.join(DIR_PATH, "data", "example09.tsv"),
            delimiter="\t",
            skiprows=1,
            labeled_rows=True,
        )
        self.assertAlmostEqualArrays(
            obtained_matrix,
            np.array(get_matrix09(), dtype=np.float64),
        )
        self.assertEqual(obtained_row_labels, get_labels04())

    def test_wrong_columns_exception(self):
        """
        Test the loading of a matrix from a TSV file that contains the wrong
        number of columns.
        """
        self.assertRaises(
            ValueError,
            load,
            os.path.join(DIR_PATH, "data", "failure01.tsv"),
            delimiter="\t",
            labeled_rows=True,
        )

    def test_no_columns_exception(self):
        """
        Test the loading of a matrix from a TSV file that does not contain any
        columns.
        """
        self.assertRaises(
            ValueError,
            load,
            os.path.join(DIR_PATH, "data", "failure02.tsv"),
            delimiter="\t",
            labeled_rows=True,
        )

    def test_wrong_type_exception(self):
        """
        Test the loading of a matrix from a CSV file that contains a wrong
        value type.
        """
        self.assertRaises(
            ValueError,
            load,
            os.path.join(DIR_PATH, "data", "failure03.csv"),
            skiprows=1,
            labeled_rows=True,
        )


if __name__ == "__main__":
    unittest.main()

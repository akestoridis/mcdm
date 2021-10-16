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
Test module for the integration of the ``mcdm`` package.
"""

import os
import unittest

import mcdm


DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TestMcdm(unittest.TestCase):
    """
    Test class for the integration of the ``mcdm`` package.
    """
    def test_default(self):
        """
        Test the integration with the default selections.
        """
        filepath = os.path.join(DIR_PATH, "data", "example03.tsv")
        x_matrix, _alt_names = mcdm.load(filepath, delimiter="\t")
        obtained_ranking = mcdm.rank(x_matrix)
        expected_ranking = [
            ("a1", 0.500000),
            ("a2", 0.500000),
            ("a3", 0.500000),
            ("a4", 0.500000),
            ("a5", 0.500000),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mew(self):
        """
        Test the integration with the MEW scoring method.
        """
        filepath = os.path.join(DIR_PATH, "data", "example03.tsv")
        x_matrix, _alt_names = mcdm.load(filepath, delimiter="\t")
        obtained_ranking = mcdm.rank(x_matrix, s_method="MEW")
        expected_ranking = [
            ("a3", 0.500000),
            ("a2", 0.433013),
            ("a4", 0.433013),
            ("a1", 0.000000),
            ("a5", 0.000000),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_topsis_w(self):
        """
        Test the integration with the TOPSIS scoring method and predefined
        weights.
        """
        filepath = os.path.join(DIR_PATH, "data", "example03.tsv")
        x_matrix, _alt_names = mcdm.load(filepath, delimiter="\t")
        obtained_ranking = mcdm.rank(
            x_matrix,
            w_vector=[0.7, 0.3],
            s_method="TOPSIS",
        )
        expected_ranking = [
            ("a5", 0.700000),
            ("a4", 0.650413),
            ("a3", 0.500000),
            ("a2", 0.349587),
            ("a1", 0.300000),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_topsis_is_benefit_x(self):
        """
        Test the integration with the TOPSIS scoring method and a mixture of
        benefit and cost criteria.
        """
        filepath = os.path.join(DIR_PATH, "data", "example03.tsv")
        x_matrix, _alt_names = mcdm.load(filepath, delimiter="\t")
        obtained_ranking = mcdm.rank(
            x_matrix,
            is_benefit_x=[True, False],
            s_method="TOPSIS",
        )
        expected_ranking = [
            ("a5", 1.000000),
            ("a4", 0.750000),
            ("a3", 0.500000),
            ("a2", 0.250000),
            ("a1", 0.000000),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_topsis_sd_vector(self):
        """
        Test the integration with the TOPSIS scoring method, the SD weighting
        method, and the Vector normalization method.
        """
        filepath = os.path.join(DIR_PATH, "data", "example08.tsv")
        x_matrix, alt_names = mcdm.load(
            filepath,
            delimiter="\t",
            labeled_rows=True,
        )
        obtained_ranking = mcdm.rank(
            x_matrix,
            alt_names=alt_names,
            n_method="Vector",
            w_method="SD",
            s_method="TOPSIS",
        )
        expected_ranking = [
            ("A", 0.562314),
            ("D", 0.472564),
            ("C", 0.447428),
            ("B", 0.438744),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_critic_linear2(self):
        """
        Test the integration with the SAW scoring method, the CRITIC weighting
        method, and the Linear2 normalization method.
        """
        filepath = os.path.join(DIR_PATH, "data", "example08.tsv")
        x_matrix, alt_names = mcdm.load(
            filepath,
            delimiter="\t",
            labeled_rows=True,
        )
        obtained_ranking = mcdm.rank(
            x_matrix,
            alt_names=alt_names,
            n_method="Linear2",
            w_method="CRITIC",
            s_method="SAW",
        )
        expected_ranking = [
            ("C", 0.586404),
            ("A", 0.536356),
            ("B", 0.422726),
            ("D", 0.418160),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mtopsis_em_linear3(self):
        """
        Test the integration with the mTOPSIS scoring method, the EM weighting
        method, and the Linear3 normalization method.
        """
        filepath = os.path.join(DIR_PATH, "data", "example08.tsv")
        x_matrix, alt_names = mcdm.load(
            filepath,
            delimiter="\t",
            labeled_rows=True,
        )
        obtained_ranking = mcdm.rank(
            x_matrix,
            alt_names=alt_names,
            n_method="Linear3",
            w_method="EM",
            s_method="mTOPSIS",
        )
        expected_ranking = [
            ("A", 0.567198),
            ("D", 0.473771),
            ("B", 0.440236),
            ("C", 0.439791),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mew_vic_linear1(self):
        """
        Test the integration with the MEW scoring method, the VIC weighting
        method, and the Linear1 normalization method.
        """
        filepath = os.path.join(DIR_PATH, "data", "example08.tsv")
        x_matrix, alt_names = mcdm.load(
            filepath,
            delimiter="\t",
            labeled_rows=True,
        )
        obtained_ranking = mcdm.rank(
            x_matrix,
            alt_names=alt_names,
            n_method="Linear1",
            w_method="VIC",
            s_method="MEW",
        )
        expected_ranking = [
            ("A", 0.596199),
            ("B", 0.592651),
            ("D", 0.581653),
            ("C", 0.507066),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mew_vic(self):
        """
        Test the integration with the MEW scoring method and VIC weighting
        method.
        """
        filepath = os.path.join(DIR_PATH, "data", "example09.tsv")
        x_matrix, alt_names = mcdm.load(
            filepath,
            delimiter="\t",
            skiprows=1,
            labeled_rows=True,
        )
        obtained_ranking = mcdm.rank(
            x_matrix,
            alt_names=alt_names,
            w_method="VIC",
            s_method="MEW",
        )
        expected_ranking = [
            ("COORD.PRoPHET", 0.475401),
            ("DF.PRoPHET",    0.472054),
            ("CnR.LTS",       0.380770),
            ("SimBetTS.L8",   0.380006),
            ("SimBetTS.L16",  0.379992),
            ("CnR.DestEnc",   0.379448),
            ("LSF-SnW.L16",   0.377400),
            ("DF.DestEnc",    0.373788),
            ("COORD.DestEnc", 0.373536),
            ("SimBetTS.L4",   0.372440),
            ("LSF-SnW.L8",    0.368945),
            ("DF.LTS",        0.366043),
            ("COORD.LTS",     0.365320),
            ("LSF-SnW.L4",    0.344986),
            ("CnF.PRoPHET",   0.344899),
            ("CnF.DestEnc",   0.340809),
            ("CnF.LTS",       0.336824),
            ("SnF.L8",        0.333813),
            ("SnF.L4",        0.331080),
            ("CnR.PRoPHET",   0.328371),
            ("SnF.L2",        0.328271),
            ("SnF.L16",       0.325965),
            ("SimBetTS.L2",   0.319820),
            ("LSF-SnW.L2",    0.283363),
            ("CnR.Enc",       0.253889),
            ("DF.Enc",        0.196428),
            ("COORD.Enc",     0.185271),
            ("Epidemic",      0.176182),
            ("Direct",        0.144637),
            ("EBR.L16",       0.144275),
            ("SnW.L16",       0.144196),
            ("EBR.L2",        0.139577),
            ("SnW.L2",        0.139347),
            ("SnW.L8",        0.137288),
            ("EBR.L8",        0.137283),
            ("EBR.L4",        0.136547),
            ("SnW.L4",        0.136425),
            ("CnF.Enc",       0.117134),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)


if __name__ == "__main__":
    unittest.main()

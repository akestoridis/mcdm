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
Test module for the ``main.py`` file of the ``mcdm`` package.
"""

import os
import unittest

import numpy as np

from mcdm import (
    load,
    rank,
)


DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TestRank(unittest.TestCase):
    """
    Test class for the ``rank`` function of the ``mcdm`` package.
    """
    def test_default(self):
        """
        Test the ranking of alternatives with the default selections.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        obtained_ranking = rank(x_matrix)
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

    def test_default_float64(self):
        """
        Test the ranking of alternatives with the default selections and a
        float64 NumPy array.
        """
        x_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float64,
        )
        obtained_ranking = rank(x_matrix)
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

    def test_default_float32(self):
        """
        Test the ranking of alternatives with the default selections and a
        float32 NumPy array.
        """
        x_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float32,
        )
        obtained_ranking = rank(x_matrix)
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

    def test_default_w_desc_order(self):
        """
        Test the ranking of alternatives with the default selections,
        predefined weights, and benefit criteria.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, True],
            w_vector=[0.7, 0.3],
        )
        expected_ranking = [
            ("a5", 0.700000),
            ("a4", 0.600000),
            ("a3", 0.500000),
            ("a2", 0.400000),
            ("a1", 0.300000),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_default_w_asc_order(self):
        """
        Test the ranking of alternatives with the default selections,
        predefined weights, and cost criteria.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[False, False],
            w_vector=[0.7, 0.3],
        )
        expected_ranking = [
            ("a1", 0.300000),
            ("a2", 0.400000),
            ("a3", 0.500000),
            ("a4", 0.600000),
            ("a5", 0.700000),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_mw_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the MW
        weighting method, and the Linear1 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear1",
            w_method="MW",
            s_method="SAW",
        )
        expected_ranking = [
            ("a2", 0.677778),
            ("a1", 0.669167),
            ("a3", 0.638889),
            ("a6", 0.625000),
            ("a5", 0.590278),
            ("a4", 0.588889),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_sd_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the SD
        weighting method, and the Linear1 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear1",
            w_method="SD",
            s_method="SAW",
        )
        expected_ranking = [
            ("a2", 0.653952),
            ("a3", 0.604472),
            ("a1", 0.601574),
            ("a6", 0.595749),
            ("a5", 0.539665),
            ("a4", 0.530537),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_vic_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the VIC
        weighting method, and the Linear1 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear1",
            w_method="VIC",
            s_method="SAW",
        )
        expected_ranking = [
            ("a2", 0.650527),
            ("a1", 0.612074),
            ("a3", 0.599994),
            ("a6", 0.594459),
            ("a5", 0.540496),
            ("a4", 0.537186),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_vic_abspearson_linear1(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the
        VIC.AbsPearson weighting method, and the Linear1 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear1",
            c_method="AbsPearson",
            w_method="VIC",
            s_method="SAW",
        )
        expected_ranking = [
            ("a2", 0.644440),
            ("a1", 0.623018),
            ("a3", 0.593228),
            ("a6", 0.591963),
            ("a4", 0.543750),
            ("a5", 0.540097),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mew_vic_linear1(self):
        """
        Test the ranking of alternatives with the MEW scoring method, the VIC
        weighting method, and the Linear1 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear1",
            w_method="VIC",
            s_method="MEW",
        )
        expected_ranking = [
            ("a6", 0.583347),
            ("a3", 0.574199),
            ("a5", 0.480220),
            ("a2", 0.469420),
            ("a4", 0.304194),
            ("a1", 0.192606),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_critic_linear2(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the
        CRITIC weighting method, and the Linear2 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear2",
            w_method="CRITIC",
            s_method="SAW",
        )
        expected_ranking = [
            ("a2", 0.669839),
            ("a5", 0.647361),
            ("a3", 0.645343),
            ("a6", 0.622660),
            ("a4", 0.587153),
            ("a1", 0.471261),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_critic_dcor_linear2(self):
        """
        Test the ranking of alternatives with the SAW scoring method, the
        CRITIC.dCor weighting method, and the Linear2 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear2",
            c_method="dCor",
            w_method="CRITIC",
            s_method="SAW",
        )
        expected_ranking = [
            ("a2", 0.677366),
            ("a5", 0.675493),
            ("a3", 0.658395),
            ("a6", 0.652317),
            ("a4", 0.622630),
            ("a1", 0.456501),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_topsis_em_linear3(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method, the
        EM weighting method, and the Linear3 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear3",
            w_method="EM",
            s_method="TOPSIS",
        )
        expected_ranking = [
            ("a6", 0.983188),
            ("a3", 0.980454),
            ("a5", 0.968182),
            ("a2", 0.967595),
            ("a4", 0.808142),
            ("a1", 0.033316),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mtopsis_em_linear3(self):
        """
        Test the ranking of alternatives with the mTOPSIS scoring method, the
        EM weighting method, and the Linear3 normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Linear3",
            w_method="EM",
            s_method="mTOPSIS",
        )
        expected_ranking = [
            ("a6", 0.955577),
            ("a5", 0.954078),
            ("a3", 0.938579),
            ("a2", 0.909531),
            ("a4", 0.808416),
            ("a1", 0.096521),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_topsis_w_vector(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method,
        predefined weights, and the Vector normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Vector",
            w_vector=[0.3, 0.2, 0.4, 0.1],
            s_method="TOPSIS",
        )
        expected_ranking = [
            ("a5", 0.868655),
            ("a6", 0.846338),
            ("a4", 0.812076),
            ("a3", 0.789327),
            ("a2", 0.718801),
            ("a1", 0.300742),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mtopsis_w_vector(self):
        """
        Test the ranking of alternatives with the mTOPSIS scoring method,
        predefined weights, and the Vector normalization method.
        """
        x_matrix = [
            [0.9, 30.0, 500.0, 4.0],
            [0.1, 50.0,   5.0, 6.0],
            [0.5, 80.0,   8.0, 6.0],
            [0.8, 40.0, 100.0, 4.0],
            [0.7, 60.0,  20.0, 5.0],
            [0.6, 60.0,  10.0, 5.0],
        ]
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Vector",
            w_vector=[0.3, 0.2, 0.4, 0.1],
            s_method="mTOPSIS",
        )
        expected_ranking = [
            ("a5", 0.836287),
            ("a6", 0.814430),
            ("a4", 0.805387),
            ("a3", 0.745801),
            ("a2", 0.688769),
            ("a1", 0.341532),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_saw_critic(self):
        """
        Test the ranking of alternatives with the SAW scoring method and the
        CRITIC weighting method.
        """
        x_matrix = [
            [1.000000, 1.000000, 0.017276],
            [0.046296, 0.022222, 1.000000],
            [0.259295, 0.106985, 0.783554],
            [0.260509, 0.107106, 0.801962],
            [0.090419, 0.044763, 0.245226],
            [0.563999, 0.239328, 0.288358],
            [0.320434, 0.147798, 0.738850],
            [0.314969, 0.144773, 0.751384],
            [0.714533, 0.364252, 0.092688],
            [0.972336, 0.706954, 0.091856],
            [0.283518, 0.127236, 0.805858],
            [0.296781, 0.132676, 0.797796],
            [0.265469, 0.122640, 0.202089],
            [0.839930, 0.461981, 0.304980],
            [0.282103, 0.126395, 0.808264],
            [0.296100, 0.132096, 0.799922],
            [0.212761, 0.104337, 0.229227],
            [0.798002, 0.429797, 0.335956],
            [0.068258, 0.035742, 0.519465],
            [0.102412, 0.055489, 0.281905],
            [0.155229, 0.085050, 0.163012],
            [0.238498, 0.128995, 0.103688],
            [0.177178, 0.075565, 0.854643],
            [0.257650, 0.112055, 0.811516],
            [0.294934, 0.131563, 0.781283],
            [0.310552, 0.140593, 0.762520],
            [0.368115, 0.159646, 0.449073],
            [0.498578, 0.228317, 0.296180],
            [0.635688, 0.310778, 0.210340],
            [0.759518, 0.402583, 0.149893],
            [0.499916, 0.188975, 0.302964],
            [0.717516, 0.306092, 0.249340],
            [0.790702, 0.359737, 0.221402],
            [0.848093, 0.415040, 0.193533],
            [0.068414, 0.035866, 0.519542],
            [0.102469, 0.055554, 0.282188],
            [0.155261, 0.085064, 0.162956],
            [0.238748, 0.129114, 0.103684],
        ]
        alt_names = [
            "Epidemic",
            "Direct",
            "CnF.LTS",
            "CnF.DestEnc",
            "CnF.Enc",
            "CnF.PRoPHET",
            "CnR.LTS",
            "CnR.DestEnc",
            "CnR.Enc",
            "CnR.PRoPHET",
            "DF.LTS",
            "DF.DestEnc",
            "DF.Enc",
            "DF.PRoPHET",
            "COORD.LTS",
            "COORD.DestEnc",
            "COORD.Enc",
            "COORD.PRoPHET",
            "SnW.L2",
            "SnW.L4",
            "SnW.L8",
            "SnW.L16",
            "LSF-SnW.L2",
            "LSF-SnW.L4",
            "LSF-SnW.L8",
            "LSF-SnW.L16",
            "SnF.L2",
            "SnF.L4",
            "SnF.L8",
            "SnF.L16",
            "SimBetTS.L2",
            "SimBetTS.L4",
            "SimBetTS.L8",
            "SimBetTS.L16",
            "EBR.L2",
            "EBR.L4",
            "EBR.L8",
            "EBR.L16",
        ]
        obtained_ranking = rank(
            x_matrix,
            alt_names=alt_names,
            is_benefit_x=[True, True, True],
            w_method="CRITIC",
            s_method="SAW",
        )
        expected_ranking = [
            ("Direct",        0.554250),
            ("COORD.DestEnc", 0.535107),
            ("COORD.LTS",     0.534726),
            ("DF.DestEnc",    0.534260),
            ("DF.LTS",        0.533976),
            ("LSF-SnW.L4",    0.527126),
            ("LSF-SnW.L8",    0.524672),
            ("CnF.DestEnc",   0.521799),
            ("LSF-SnW.L2",    0.521617),
            ("LSF-SnW.L16",   0.520533),
            ("CnR.DestEnc",   0.516544),
            ("CnR.LTS",       0.511861),
            ("CnF.LTS",       0.511555),
            ("DF.PRoPHET",    0.479107),
            ("COORD.PRoPHET", 0.478254),
            ("Epidemic",      0.471779),
            ("CnR.PRoPHET",   0.447615),
            ("SimBetTS.L16",  0.412294),
            ("SimBetTS.L8",   0.401135),
            ("SimBetTS.L4",   0.386093),
            ("SnF.L2",        0.371208),
            ("SnF.L16",       0.362631),
            ("CnF.PRoPHET",   0.352886),
            ("SnF.L8",        0.344061),
            ("SnF.L4",        0.337384),
            ("SimBetTS.L2",   0.333762),
            ("CnR.Enc",       0.312368),
            ("EBR.L2",        0.304587),
            ("SnW.L2",        0.304480),
            ("DF.Enc",        0.203707),
            ("COORD.Enc",     0.200588),
            ("EBR.L4",        0.189972),
            ("SnW.L4",        0.189792),
            ("CnF.Enc",       0.164776),
            ("SnW.L8",        0.145805),
            ("EBR.L8",        0.145786),
            ("EBR.L16",       0.144892),
            ("SnW.L16",       0.144804),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_mew_vic(self):
        """
        Test the ranking of alternatives with the MEW scoring method and the
        VIC weighting method.
        """
        x_matrix = [
            [1.000000, 1.000000, 0.017276],
            [0.046296, 0.022222, 1.000000],
            [0.259295, 0.106985, 0.783554],
            [0.260509, 0.107106, 0.801962],
            [0.090419, 0.044763, 0.245226],
            [0.563999, 0.239328, 0.288358],
            [0.320434, 0.147798, 0.738850],
            [0.314969, 0.144773, 0.751384],
            [0.714533, 0.364252, 0.092688],
            [0.972336, 0.706954, 0.091856],
            [0.283518, 0.127236, 0.805858],
            [0.296781, 0.132676, 0.797796],
            [0.265469, 0.122640, 0.202089],
            [0.839930, 0.461981, 0.304980],
            [0.282103, 0.126395, 0.808264],
            [0.296100, 0.132096, 0.799922],
            [0.212761, 0.104337, 0.229227],
            [0.798002, 0.429797, 0.335956],
            [0.068258, 0.035742, 0.519465],
            [0.102412, 0.055489, 0.281905],
            [0.155229, 0.085050, 0.163012],
            [0.238498, 0.128995, 0.103688],
            [0.177178, 0.075565, 0.854643],
            [0.257650, 0.112055, 0.811516],
            [0.294934, 0.131563, 0.781283],
            [0.310552, 0.140593, 0.762520],
            [0.368115, 0.159646, 0.449073],
            [0.498578, 0.228317, 0.296180],
            [0.635688, 0.310778, 0.210340],
            [0.759518, 0.402583, 0.149893],
            [0.499916, 0.188975, 0.302964],
            [0.717516, 0.306092, 0.249340],
            [0.790702, 0.359737, 0.221402],
            [0.848093, 0.415040, 0.193533],
            [0.068414, 0.035866, 0.519542],
            [0.102469, 0.055554, 0.282188],
            [0.155261, 0.085064, 0.162956],
            [0.238748, 0.129114, 0.103684],
        ]
        alt_names = [
            "Epidemic",
            "Direct",
            "CnF.LTS",
            "CnF.DestEnc",
            "CnF.Enc",
            "CnF.PRoPHET",
            "CnR.LTS",
            "CnR.DestEnc",
            "CnR.Enc",
            "CnR.PRoPHET",
            "DF.LTS",
            "DF.DestEnc",
            "DF.Enc",
            "DF.PRoPHET",
            "COORD.LTS",
            "COORD.DestEnc",
            "COORD.Enc",
            "COORD.PRoPHET",
            "SnW.L2",
            "SnW.L4",
            "SnW.L8",
            "SnW.L16",
            "LSF-SnW.L2",
            "LSF-SnW.L4",
            "LSF-SnW.L8",
            "LSF-SnW.L16",
            "SnF.L2",
            "SnF.L4",
            "SnF.L8",
            "SnF.L16",
            "SimBetTS.L2",
            "SimBetTS.L4",
            "SimBetTS.L8",
            "SimBetTS.L16",
            "EBR.L2",
            "EBR.L4",
            "EBR.L8",
            "EBR.L16",
        ]
        obtained_ranking = rank(
            x_matrix,
            alt_names=alt_names,
            is_benefit_x=[True, True, True],
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

    def test_topsis_w_vector_float64(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method,
        predefined weights, the Vector normalization method, and float64 NumPy
        arrays.
        """
        x_matrix = np.array(
            [
                [0.9, 30.0, 500.0, 4.0],
                [0.1, 50.0,   5.0, 6.0],
                [0.5, 80.0,   8.0, 6.0],
                [0.8, 40.0, 100.0, 4.0],
                [0.7, 60.0,  20.0, 5.0],
                [0.6, 60.0,  10.0, 5.0],
            ],
            dtype=np.float64,
        )
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Vector",
            w_vector=np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float64),
            s_method="TOPSIS",
        )
        expected_ranking = [
            ("a5", 0.868655),
            ("a6", 0.846338),
            ("a4", 0.812076),
            ("a3", 0.789327),
            ("a2", 0.718801),
            ("a1", 0.300742),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_topsis_w_vector_float32(self):
        """
        Test the ranking of alternatives with the TOPSIS scoring method,
        predefined weights, the Vector normalization method, and float32 NumPy
        arrays.
        """
        x_matrix = np.array(
            [
                [0.9, 30.0, 500.0, 4.0],
                [0.1, 50.0,   5.0, 6.0],
                [0.5, 80.0,   8.0, 6.0],
                [0.8, 40.0, 100.0, 4.0],
                [0.7, 60.0,  20.0, 5.0],
                [0.6, 60.0,  10.0, 5.0],
            ],
            dtype=np.float32,
        )
        obtained_ranking = rank(
            x_matrix,
            is_benefit_x=[True, False, False, True],
            n_method="Vector",
            w_vector=np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float32),
            s_method="TOPSIS",
        )
        expected_ranking = [
            ("a5", 0.868655),
            ("a6", 0.846338),
            ("a4", 0.812076),
            ("a3", 0.789327),
            ("a2", 0.718801),
            ("a1", 0.300742),
        ]
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)

    def test_missing_element_exception(self):
        """
        Test the ranking of alternatives with a missing element.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75],
            [1.00, 0.00],
        ]
        alt_names = ["A", "B", "C", "D", "E", "F"]
        self.assertRaises(
            ValueError,
            rank,
            x_matrix,
            alt_names,
            is_benefit_x=[True, True],
        )

    def test_alt_names_exception(self):
        """
        Test the ranking of alternatives with an invalid list of names.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        alt_names = ["A", "B", "C", "D", "E", "F"]
        self.assertRaises(
            ValueError,
            rank,
            x_matrix,
            alt_names,
            is_benefit_x=[True, True],
        )

    def test_is_benefit_x_exception(self):
        """
        Test the ranking of alternatives with an invalid Boolean list.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        alt_names = ["A", "B", "C", "D", "E"]
        self.assertRaises(
            ValueError,
            rank,
            x_matrix,
            alt_names,
            is_benefit_x=[True, True, True],
        )

    def test_w_vector_length_exception(self):
        """
        Test the ranking of alternatives with an invalid weight vector length.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        alt_names = ["A", "B", "C", "D", "E"]
        self.assertRaises(
            ValueError,
            rank,
            x_matrix,
            alt_names,
            is_benefit_x=[True, True],
            w_vector=[0.25, 0.25, 0.25, 0.25],
        )

    def test_w_vector_sum_exception(self):
        """
        Test the ranking of alternatives with an invalid weight vector sum.
        """
        x_matrix = [
            [0.00, 1.00],
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25],
            [1.00, 0.00],
        ]
        alt_names = ["A", "B", "C", "D", "E"]
        self.assertRaises(
            ValueError,
            rank,
            x_matrix,
            alt_names,
            is_benefit_x=[True, True],
            w_vector=[0.5, 0.4],
        )


class TestLoad(unittest.TestCase):
    """
    Test class for the ``load`` function of the ``mcdm`` package.
    """
    def test_example01(self):
        """
        Test the loading of a matrix from a CSV file with the default
        parameter values.
        """
        filepath = os.path.join(DIR_PATH, "data", "example01.csv")
        obtained_matrix, obtained_row_labels = load(filepath)
        expected_matrix = np.array(
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
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example02(self):
        """
        Test the loading of a matrix from a CSV file that contains column and
        row labels.
        """
        filepath = os.path.join(DIR_PATH, "data", "example02.csv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            skiprows=1,
            labeled_rows=True,
        )
        expected_matrix = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.5, 0.0],
                [0.2, 0.5, 1.0],
                [0.4, 1.0, 0.0],
                [0.4, 1.0, 1.0],
                [0.6, 1.0, 0.0],
                [0.6, 1.0, 1.0],
                [0.8, 0.5, 0.0],
                [0.8, 0.5, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_row_labels = [
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "a8",
            "a9",
            "a10",
            "a11",
            "a12",
        ]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example03(self):
        """
        Test the loading of a matrix from a TSV file that does not contain any
        labels.
        """
        filepath = os.path.join(DIR_PATH, "data", "example03.tsv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            delimiter="\t",
        )
        expected_matrix = np.array(
            [
                [0.00, 1.00],
                [0.25, 0.75],
                [0.50, 0.50],
                [0.75, 0.25],
                [1.00, 0.00],
            ],
            dtype=np.float64,
        )
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example04(self):
        """
        Test the loading of a matrix from a TSV file that contains a
        single-line comment.
        """
        filepath = os.path.join(DIR_PATH, "data", "example04.tsv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            delimiter="\t",
            skiprows=1,
        )
        expected_matrix = np.array(
            [
                [ 2.0,  12.0, 7.0, 7.0],  # noqa: E201
                [ 4.0, 100.0, 7.0, 7.0],  # noqa: E201
                [10.0, 200.0, 7.0, 7.0],  # noqa: E201
                [ 0.0, 300.0, 7.0, 7.0],  # noqa: E201
                [ 6.0, 400.0, 7.0, 7.0],  # noqa: E201
                [ 1.0, 600.0, 7.0, 7.0],  # noqa: E201
            ],
            dtype=np.float64,
        )
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example05(self):
        """
        Test the loading of a matrix from a CSV file that contains row labels.
        """
        filepath = os.path.join(DIR_PATH, "data", "example05.csv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            labeled_rows=True,
        )
        expected_matrix = np.array(
            [
                [ 8.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],  # noqa: E201
                [24.0, 24.0, -11.0, -11.0,   0.0,   0.0],  # noqa: E201
                [ 4.0,  4.0, -10.0, -10.0,  40.0,  40.0],  # noqa: E201
                [14.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],  # noqa: E201
                [ 6.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],  # noqa: E201
                [18.0, 18.0,  -5.0,  -5.0, -10.0, -10.0],  # noqa: E201
            ],
            dtype=np.float64,
        )
        expected_row_labels = ["A", "B", "C", "D", "E", "F"]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example06(self):
        """
        Test the loading of a matrix from a CSV file that contains a
        single-line comment.
        """
        filepath = os.path.join(DIR_PATH, "data", "example06.csv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            skiprows=1,
        )
        expected_matrix = np.array(
            [
                [0.5, 0.6, 0.3, 0.2, 0.9],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.4, 0.7, 0.8, 0.1],
            ],
            dtype=np.float64,
        )
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example07(self):
        """
        Test the loading of a matrix from a CSV file that contains a
        multi-line comment.
        """
        filepath = os.path.join(DIR_PATH, "data", "example07.csv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            skiprows=3,
        )
        expected_matrix = np.array(
            [
                [0.9, 30.0, 500.0, 4.0],
                [0.1, 50.0,   5.0, 6.0],
                [0.5, 80.0,   8.0, 6.0],
                [0.8, 40.0, 100.0, 4.0],
                [0.7, 60.0,  20.0, 5.0],
                [0.6, 60.0,  10.0, 5.0],
            ],
            dtype=np.float64,
        )
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example08(self):
        """
        Test the loading of a matrix from a TSV file that contains row labels.
        """
        filepath = os.path.join(DIR_PATH, "data", "example08.tsv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            delimiter="\t",
            labeled_rows=True,
        )
        expected_matrix = np.array(
            [
                [4.0,  5.0, 10.0],
                [3.0, 10.0,  6.0],
                [3.0, 20.0,  2.0],
                [2.0, 15.0,  5.0],
            ],
            dtype=np.float64,
        )
        expected_row_labels = ["A", "B", "C", "D"]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_example09(self):
        """
        Test the loading of a matrix from a large TSV file.
        """
        filepath = os.path.join(DIR_PATH, "data", "example09.tsv")
        obtained_matrix, obtained_row_labels = load(
            filepath,
            delimiter="\t",
            skiprows=1,
            labeled_rows=True,
        )
        expected_matrix = np.array(
            [
                [1.000000, 1.000000, 0.017276],
                [0.046296, 0.022222, 1.000000],
                [0.259295, 0.106985, 0.783554],
                [0.260509, 0.107106, 0.801962],
                [0.090419, 0.044763, 0.245226],
                [0.563999, 0.239328, 0.288358],
                [0.320434, 0.147798, 0.738850],
                [0.314969, 0.144773, 0.751384],
                [0.714533, 0.364252, 0.092688],
                [0.972336, 0.706954, 0.091856],
                [0.283518, 0.127236, 0.805858],
                [0.296781, 0.132676, 0.797796],
                [0.265469, 0.122640, 0.202089],
                [0.839930, 0.461981, 0.304980],
                [0.282103, 0.126395, 0.808264],
                [0.296100, 0.132096, 0.799922],
                [0.212761, 0.104337, 0.229227],
                [0.798002, 0.429797, 0.335956],
                [0.068258, 0.035742, 0.519465],
                [0.102412, 0.055489, 0.281905],
                [0.155229, 0.085050, 0.163012],
                [0.238498, 0.128995, 0.103688],
                [0.177178, 0.075565, 0.854643],
                [0.257650, 0.112055, 0.811516],
                [0.294934, 0.131563, 0.781283],
                [0.310552, 0.140593, 0.762520],
                [0.368115, 0.159646, 0.449073],
                [0.498578, 0.228317, 0.296180],
                [0.635688, 0.310778, 0.210340],
                [0.759518, 0.402583, 0.149893],
                [0.499916, 0.188975, 0.302964],
                [0.717516, 0.306092, 0.249340],
                [0.790702, 0.359737, 0.221402],
                [0.848093, 0.415040, 0.193533],
                [0.068414, 0.035866, 0.519542],
                [0.102469, 0.055554, 0.282188],
                [0.155261, 0.085064, 0.162956],
                [0.238748, 0.129114, 0.103684],
            ],
            dtype=np.float64,
        )
        expected_row_labels = [
            "Epidemic",
            "Direct",
            "CnF.LTS",
            "CnF.DestEnc",
            "CnF.Enc",
            "CnF.PRoPHET",
            "CnR.LTS",
            "CnR.DestEnc",
            "CnR.Enc",
            "CnR.PRoPHET",
            "DF.LTS",
            "DF.DestEnc",
            "DF.Enc",
            "DF.PRoPHET",
            "COORD.LTS",
            "COORD.DestEnc",
            "COORD.Enc",
            "COORD.PRoPHET",
            "SnW.L2",
            "SnW.L4",
            "SnW.L8",
            "SnW.L16",
            "LSF-SnW.L2",
            "LSF-SnW.L4",
            "LSF-SnW.L8",
            "LSF-SnW.L16",
            "SnF.L2",
            "SnF.L4",
            "SnF.L8",
            "SnF.L16",
            "SimBetTS.L2",
            "SimBetTS.L4",
            "SimBetTS.L8",
            "SimBetTS.L16",
            "EBR.L2",
            "EBR.L4",
            "EBR.L8",
            "EBR.L16",
        ]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_wrong_columns_exception(self):
        """
        Test the loading of a matrix from a TSV file that contains the wrong
        number of columns.
        """
        filepath = os.path.join(DIR_PATH, "data", "failure01.tsv")
        self.assertRaises(
            ValueError,
            load,
            filepath,
            delimiter="\t",
            labeled_rows=True,
        )

    def test_no_columns_exception(self):
        """
        Test the loading of a matrix from a TSV file that does not contain any
        columns.
        """
        filepath = os.path.join(DIR_PATH, "data", "failure02.tsv")
        self.assertRaises(
            ValueError,
            load,
            filepath,
            delimiter="\t",
            labeled_rows=True,
        )

    def test_wrong_type_exception(self):
        """
        Test the loading of a matrix from a CSV file that contains a wrong
        value type.
        """
        filepath = os.path.join(DIR_PATH, "data", "failure03.csv")
        self.assertRaises(
            ValueError,
            load,
            filepath,
            skiprows=1,
            labeled_rows=True,
        )


if __name__ == "__main__":
    unittest.main()

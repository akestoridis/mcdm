#!/usr/bin/env python3

# Copyright (c) 2020 Dimitrios-Georgios Akestoridis
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import mcdm
import numpy as np
import unittest


class TestWeigh(unittest.TestCase):
    def test_mw_linear(self):
        """Test the MW method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "MW")
        expected_w_vector = np.array(
            [0.33333333, 0.33333333, 0.33333333],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_mw_nonlinear(self):
        """Test the MW method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "MW")
        expected_w_vector = np.array(
            [0.33333333, 0.33333333, 0.33333333],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_mw_over_exception(self):
        """Test the MW method with a value greater than 1."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.1],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "MW")

    def test_mw_under_exception(self):
        """Test the MW method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.0, 0.0, 1.0],
             [-0.1, 0.2, 0.8],
             [ 0.2, 0.4, 0.6],
             [ 0.3, 0.7, 0.3],
             [ 0.6, 0.8, 0.2],
             [ 0.8, 0.9, 0.1],
             [ 1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "MW")

    def test_em_linear(self):
        """Test the EM method with a linear association."""
        z_matrix = np.array(
            [[0.000, 0.000, 0.333],
             [0.033, 0.050, 0.267],
             [0.067, 0.100, 0.200],
             [0.100, 0.175, 0.100],
             [0.200, 0.200, 0.067],
             [0.267, 0.225, 0.033],
             [0.333, 0.250, 0.000]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "EM")
        expected_w_vector = np.array(
            [0.37406776, 0.25186448, 0.37406776],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_em_nonlinear(self):
        """Test the EM method with a non-linear association."""
        z_matrix = np.array(
            [[0.00000000, 0.00000000, 0.00000000],
             [0.00000000, 0.00000000, 0.16666667],
             [0.03333333, 0.08333333, 0.00000000],
             [0.03333333, 0.08333333, 0.16666667],
             [0.06666667, 0.16666667, 0.00000000],
             [0.06666667, 0.16666667, 0.16666667],
             [0.10000000, 0.16666667, 0.00000000],
             [0.10000000, 0.16666667, 0.16666667],
             [0.13333333, 0.08333333, 0.00000000],
             [0.13333333, 0.08333333, 0.16666667],
             [0.16666667, 0.00000000, 0.00000000],
             [0.16666667, 0.00000000, 0.16666667]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "EM")
        expected_w_vector = np.array(
            [0.20724531, 0.31710188, 0.47565280],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_em_over_exception(self):
        """Test the EM method with a value greater than 1."""
        z_matrix = np.array(
            [[0.000, 0.000, 1.333],
             [0.033, 0.050, 0.267],
             [0.067, 0.100, 0.200],
             [0.100, 0.175, 0.100],
             [0.200, 0.200, 0.067],
             [0.267, 0.225, 0.033],
             [0.333, 0.250, 0.000]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "EM")

    def test_em_under_exception(self):
        """Test the EM method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.000, 0.000, 0.333],
             [-0.033, 0.050, 0.267],
             [ 0.067, 0.100, 0.200],
             [ 0.100, 0.175, 0.100],
             [ 0.200, 0.200, 0.067],
             [ 0.267, 0.225, 0.033],
             [ 0.333, 0.250, 0.000]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "EM")

    def test_em_sum_exception(self):
        """Test the EM method with a column that does not sum to 1."""
        z_matrix = np.array(
            [[0.000, 0.0, 0.333],
             [0.033, 0.2, 0.267],
             [0.067, 0.4, 0.200],
             [0.100, 0.7, 0.100],
             [0.200, 0.8, 0.067],
             [0.267, 0.9, 0.033],
             [0.333, 1.0, 0.000]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "EM")

    def test_sd_linear(self):
        """Test the SD method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "SD")
        expected_w_vector = np.array(
            [0.33333333, 0.33333333, 0.33333333],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_sd_nonlinear(self):
        """Test the SD method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "SD")
        expected_w_vector = np.array(
            [0.27329284, 0.32664742, 0.40005975],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_sd_over_exception(self):
        """Test the SD method with a value greater than 1."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.1],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "SD")

    def test_sd_under_exception(self):
        """Test the SD method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.0, 0.0, 1.0],
             [-0.1, 0.2, 0.8],
             [ 0.2, 0.4, 0.6],
             [ 0.3, 0.7, 0.3],
             [ 0.6, 0.8, 0.2],
             [ 0.8, 0.9, 0.1],
             [ 1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "SD")

    def test_critic_linear(self):
        """Test the CRITIC method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC")
        expected_w_vector = np.array(
            [0.25000000, 0.25857023, 0.49142977],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_nonlinear(self):
        """Test the CRITIC method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC")
        expected_w_vector = np.array(
            [0.27329284, 0.32664742, 0.40005975],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_pearson_linear(self):
        """Test the CRITIC.Pearson method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC", "Pearson")
        expected_w_vector = np.array(
            [0.25000000, 0.25857023, 0.49142977],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_pearson_nonlinear(self):
        """Test the CRITIC.Pearson method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC", "Pearson")
        expected_w_vector = np.array(
            [0.27329284, 0.32664742, 0.40005975],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_abspearson_linear(self):
        """Test the CRITIC.AbsPearson method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC", "AbsPearson")
        expected_w_vector = np.array(
            [0.50000000, 0.25000000, 0.25000000],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_abspearson_nonlinear(self):
        """Test the CRITIC.AbsPearson method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC", "AbsPearson")
        expected_w_vector = np.array(
            [0.27329284, 0.32664742, 0.40005975],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_dcor_linear(self):
        """Test the CRITIC.dCor method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC", "dCor")
        expected_w_vector = np.array(
            [0.50000000, 0.25000000, 0.25000000],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_dcor_nonlinear(self):
        """Test the CRITIC.dCor method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "CRITIC", "dCor")
        expected_w_vector = np.array(
            [0.23971980, 0.28651997, 0.47376023],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_critic_over_exception(self):
        """Test the CRITIC method with a value greater than 1."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.1],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "CRITIC")

    def test_critic_under_exception(self):
        """Test the CRITIC method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.0, 0.0, 1.0],
             [-0.1, 0.2, 0.8],
             [ 0.2, 0.4, 0.6],
             [ 0.3, 0.7, 0.3],
             [ 0.6, 0.8, 0.2],
             [ 0.8, 0.9, 0.1],
             [ 1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "CRITIC")

    def test_critic_unknown_exception(self):
        """Test the CRITIC method with an unknown correlation method."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh,
                          z_matrix, "CRITIC", "Unknown")

    def test_vic_linear(self):
        """Test the VIC method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "VIC")
        expected_w_vector = np.array(
            [0.33817571, 0.33091215, 0.33091215],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_vic_nonlinear(self):
        """Test the VIC method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "VIC")
        expected_w_vector = np.array(
            [0.22633480, 0.27052183, 0.50314336],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_vic_abspearson_linear(self):
        """Test the VIC.AbsPearson method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "VIC", "AbsPearson")
        expected_w_vector = np.array(
            [0.33861310, 0.33069345, 0.33069345],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_vic_abspearson_nonlinear(self):
        """Test the VIC.AbsPearson method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "VIC", "AbsPearson")
        expected_w_vector = np.array(
            [0.27329284, 0.32664742, 0.40005975],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_vic_dcor_linear(self):
        """Test the VIC.dCor method with a linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "VIC", "dCor")
        expected_w_vector = np.array(
            [0.33817571, 0.33091215, 0.33091215],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_vic_dcor_nonlinear(self):
        """Test the VIC.dCor method with a non-linear association."""
        z_matrix = np.array(
            [[0.0, 0.0, 0.0],
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
             [1.0, 0.0, 1.0]],
            dtype=np.float64)
        obtained_w_vector = mcdm.weigh(z_matrix, "VIC", "dCor")
        expected_w_vector = np.array(
            [0.22633480, 0.27052183, 0.50314336],
            dtype=np.float64)
        np.testing.assert_allclose(obtained_w_vector, expected_w_vector)
        self.assertEqual(obtained_w_vector.dtype, expected_w_vector.dtype)

    def test_vic_over_exception(self):
        """Test the VIC method with a value greater than 1."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.1],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "VIC")

    def test_vic_under_exception(self):
        """Test the VIC method with a value less than 0."""
        z_matrix = np.array(
            [[ 0.0, 0.0, 1.0],
             [-0.1, 0.2, 0.8],
             [ 0.2, 0.4, 0.6],
             [ 0.3, 0.7, 0.3],
             [ 0.6, 0.8, 0.2],
             [ 0.8, 0.9, 0.1],
             [ 1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "VIC")

    def test_vic_pearson_exception(self):
        """Test the VIC method with the Pearson correlation method."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "VIC", "Pearson")

    def test_vic_unknown_exception(self):
        """Test the VIC method with an unknown correlation method."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "VIC", "Unknown")

    def test_unknown_weigh_exception(self):
        """Test the selection of an unknown weighting method."""
        z_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        self.assertRaises(ValueError, mcdm.weigh, z_matrix, "Unknown")


if __name__ == "__main__":
    unittest.main()

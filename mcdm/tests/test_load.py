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
import os
import unittest


DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TestLoad(unittest.TestCase):
    def test_loading_example01(self):
        """Test loading a matrix with the default parameter values."""
        filepath = os.path.join(DIR_PATH, "data", "example01.csv")
        obtained_matrix, obtained_row_labels = mcdm.load(filepath)
        expected_matrix = np.array(
            [[0.0, 0.0, 1.0],
             [0.1, 0.2, 0.8],
             [0.2, 0.4, 0.6],
             [0.3, 0.7, 0.3],
             [0.6, 0.8, 0.2],
             [0.8, 0.9, 0.1],
             [1.0, 1.0, 0.0]],
            dtype=np.float64)
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example02(self):
        """Test loading a matrix with column and row labels."""
        filepath = os.path.join(DIR_PATH, "data", "example02.csv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, skiprows=1, labeled_rows=True)
        expected_matrix = np.array(
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
            "a12"
        ]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example03(self):
        """Test loading a matrix without any labels from a TSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example03.tsv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, delimiter="\t")
        expected_matrix = np.array(
            [[0.00, 1.00],
             [0.25, 0.75],
             [0.50, 0.50],
             [0.75, 0.25],
             [1.00, 0.00]],
            dtype=np.float64)
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example04(self):
        """Test loading a matrix with a comment line from a TSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example04.tsv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, delimiter="\t", skiprows=1)
        expected_matrix = np.array(
            [[ 2.0,  12.0, 7.0, 7.0],
             [ 4.0, 100.0, 7.0, 7.0],
             [10.0, 200.0, 7.0, 7.0],
             [ 0.0, 300.0, 7.0, 7.0],
             [ 6.0, 400.0, 7.0, 7.0],
             [ 1.0, 600.0, 7.0, 7.0]],
            dtype=np.float64)
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example05(self):
        """Test loading a matrix with row labels from a CSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example05.csv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, labeled_rows=True)
        expected_matrix = np.array(
            [[ 8.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
             [24.0, 24.0, -11.0, -11.0,   0.0,   0.0],
             [ 4.0,  4.0, -10.0, -10.0,  40.0,  40.0],
             [14.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
             [ 6.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
             [18.0, 18.0,  -5.0,  -5.0, -10.0, -10.0]],
            dtype=np.float64)
        expected_row_labels = ["A", "B", "C", "D", "E", "F"]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example06(self):
        """Test loading a matrix with a comment line from a CSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example06.csv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, skiprows=1)
        expected_matrix = np.array(
            [[0.5, 0.6, 0.3, 0.2, 0.9],
             [0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.4, 0.7, 0.8, 0.1]],
            dtype=np.float64)
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example07(self):
        """Test loading a matrix with a multi-line comment from a CSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example07.csv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, skiprows=3)
        expected_matrix = np.array(
            [[0.9, 30.0, 500.0, 4.0],
             [0.1, 50.0,   5.0, 6.0],
             [0.5, 80.0,   8.0, 6.0],
             [0.8, 40.0, 100.0, 4.0],
             [0.7, 60.0,  20.0, 5.0],
             [0.6, 60.0,  10.0, 5.0]],
            dtype=np.float64)
        expected_row_labels = None
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example08(self):
        """Test loading a matrix with row labels from a TSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example08.tsv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, delimiter="\t", labeled_rows=True)
        expected_matrix = np.array(
            [[4.0,  5.0, 10.0],
             [3.0, 10.0,  6.0],
             [3.0, 20.0,  2.0],
             [2.0, 15.0,  5.0]],
            dtype=np.float64)
        expected_row_labels = ["A", "B", "C", "D"]
        np.testing.assert_allclose(obtained_matrix, expected_matrix)
        self.assertEqual(obtained_matrix.dtype, expected_matrix.dtype)
        self.assertEqual(obtained_row_labels, expected_row_labels)

    def test_loading_example09(self):
        """Test loading a large matrix from a TSV file."""
        filepath = os.path.join(DIR_PATH, "data", "example09.tsv")
        obtained_matrix, obtained_row_labels = mcdm.load(
            filepath, delimiter="\t", skiprows=1, labeled_rows=True)
        expected_matrix = np.array(
            [[1.000000, 1.000000, 0.017276],
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
             [0.238748, 0.129114, 0.103684]],
            dtype=np.float64)
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

    def test_num_columns_exception(self):
        """Test loading a matrix with a wrong number of columns."""
        filepath = os.path.join(DIR_PATH, "data", "failure01.tsv")
        self.assertRaises(ValueError, mcdm.load, filepath,
                          delimiter="\t", labeled_rows=True)

    def test_no_columns_exception(self):
        """Test loading a matrix without any columns."""
        filepath = os.path.join(DIR_PATH, "data", "failure02.tsv")
        self.assertRaises(ValueError, mcdm.load, filepath,
                          delimiter="\t", labeled_rows=True)

    def test_wrong_type_exception(self):
        """Test loading a matrix with a wrong value type."""
        filepath = os.path.join(DIR_PATH, "data", "failure03.csv")
        self.assertRaises(ValueError, mcdm.load, filepath,
                          skiprows=1, labeled_rows=True)


if __name__ == "__main__":
    unittest.main()

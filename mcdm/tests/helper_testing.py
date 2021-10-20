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
Helper module for the testing of the ``mcdm`` package.
"""

import unittest

import numpy as np


class ExtendedTestCase(unittest.TestCase):
    # pylint: disable=invalid-name
    """
    Extended ``TestCase`` class of the ``unittest`` module.
    """
    def assertAlmostEqualArrays(self, obtained_array, expected_array):
        """
        Assert that two NumPy arrays are element-wise almost equal and use the
        same data type.
        """
        np.testing.assert_allclose(obtained_array, expected_array)
        self.assertEqual(obtained_array.dtype, expected_array.dtype)

    def assertAlmostEqualRankings(self, obtained_ranking, expected_ranking):
        """
        Assert that two lists of tuples contain the same alternatives in the
        same order with almost equal scores.
        """
        self.assertEqual(len(obtained_ranking), len(expected_ranking))
        for i, tmp in enumerate(obtained_ranking):
            self.assertEqual(tmp[0], expected_ranking[i][0])
            self.assertAlmostEqual(tmp[1], expected_ranking[i][1], places=6)


def get_labels01():
    """
    Return the labels with ID 01.
    """
    return [
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


def get_labels02():
    """
    Return the labels with ID 02.
    """
    return [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]


def get_labels03():
    """
    Return the labels with ID 03.
    """
    return [
        "A",
        "B",
        "C",
        "D",
    ]


def get_labels04():
    """
    Return the labels with ID 04.
    """
    return [
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


def get_labels05():
    """
    Return the labels with ID 05.
    """
    return [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]


def get_matrix01():
    """
    Return the matrix with ID 01.
    """
    return [
        [0.0, 0.0, 1.0],
        [0.1, 0.2, 0.8],
        [0.2, 0.4, 0.6],
        [0.3, 0.7, 0.3],
        [0.6, 0.8, 0.2],
        [0.8, 0.9, 0.1],
        [1.0, 1.0, 0.0],
    ]


def get_matrix02():
    """
    Return the matrix with ID 02.
    """
    return [
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
    ]


def get_matrix03():
    """
    Return the matrix with ID 03.
    """
    return [
        [0.00, 1.00],
        [0.25, 0.75],
        [0.50, 0.50],
        [0.75, 0.25],
        [1.00, 0.00],
    ]


def get_matrix04():
    """
    Return the matrix with ID 04.
    """
    return [
        [ 2.0,  12.0, 7.0, 7.0],  # noqa: E201
        [ 4.0, 100.0, 7.0, 7.0],  # noqa: E201
        [10.0, 200.0, 7.0, 7.0],  # noqa: E201
        [ 0.0, 300.0, 7.0, 7.0],  # noqa: E201
        [ 6.0, 400.0, 7.0, 7.0],  # noqa: E201
        [ 1.0, 600.0, 7.0, 7.0],  # noqa: E201
    ]


def get_matrix05():
    """
    Return the matrix with ID 05.
    """
    return [
        [ 8.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],  # noqa: E201
        [24.0, 24.0, -11.0, -11.0,   0.0,   0.0],  # noqa: E201
        [ 4.0,  4.0, -10.0, -10.0,  40.0,  40.0],  # noqa: E201
        [14.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],  # noqa: E201
        [ 6.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],  # noqa: E201
        [18.0, 18.0,  -5.0,  -5.0, -10.0, -10.0],  # noqa: E201
    ]


def get_matrix06():
    """
    Return the matrix with ID 06.
    """
    return [
        [0.5, 0.6, 0.3, 0.2, 0.9],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.4, 0.7, 0.8, 0.1],
    ]


def get_matrix07():
    """
    Return the matrix with ID 07.
    """
    return [
        [0.9, 30.0, 500.0, 4.0],
        [0.1, 50.0,   5.0, 6.0],
        [0.5, 80.0,   8.0, 6.0],
        [0.8, 40.0, 100.0, 4.0],
        [0.7, 60.0,  20.0, 5.0],
        [0.6, 60.0,  10.0, 5.0],
    ]


def get_matrix08():
    """
    Return the matrix with ID 08.
    """
    return [
        [4.0,  5.0, 10.0],
        [3.0, 10.0,  6.0],
        [3.0, 20.0,  2.0],
        [2.0, 15.0,  5.0],
    ]


def get_matrix09():
    """
    Return the matrix with ID 09.
    """
    return [
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


def get_matrix10():
    """
    Return the matrix with ID 10.
    """
    return [
        [0.00, 1.00],
        [0.25, 0.75],
        [0.50, 0.50],
        [0.75],
        [1.00, 0.00],
    ]


def get_matrix11():
    """
    Return the matrix with ID 11.
    """
    return [
        [0.0, 0.0, 1.0],
        [0.1, 0.2, 0.8],
        [0.2, 0.4, 0.6],
        [0.3, 0.7, 0.3],
        [0.6, 0.8, 0.2],
        [0.8, 0.9],
        [1.0, 1.0, 0.0],
    ]


def get_matrix12():
    """
    Return the matrix with ID 12.
    """
    return [
        [0.0, 0.0, 1.1],
        [0.1, 0.2, 0.8],
        [0.2, 0.4, 0.6],
        [0.3, 0.7, 0.3],
        [0.6, 0.8, 0.2],
        [0.8, 0.9, 0.1],
        [1.0, 1.0, 0.0],
    ]


def get_matrix13():
    """
    Return the matrix with ID 13.
    """
    return [
        [ 0.0, 0.0, 1.0],  # noqa: E201
        [-0.1, 0.2, 0.8],  # noqa: E201
        [ 0.2, 0.4, 0.6],  # noqa: E201
        [ 0.3, 0.7, 0.3],  # noqa: E201
        [ 0.6, 0.8, 0.2],  # noqa: E201
        [ 0.8, 0.9, 0.1],  # noqa: E201
        [ 1.0, 1.0, 0.0],  # noqa: E201
    ]


def get_matrix14():
    """
    Return the matrix with ID 14.
    """
    return [
        [0.2, 1.00, 1.0, 1.0],
        [0.4, 0.12, 1.0, 1.0],
        [1.0, 0.06, 1.0, 1.0],
        [0.0, 0.04, 1.0, 1.0],
        [0.6, 0.03, 1.0, 1.0],
        [0.1, 0.02, 1.0, 1.0],
    ]


def get_matrix15():
    """
    Return the matrix with ID 15.
    """
    return [
        [ 2.0,  12.0, 7.0, 7.0],  # noqa: E201
        [ 4.0, 100.0, 7.0, 7.0],  # noqa: E201
        [10.0, 200.0, 7.0, 7.0],  # noqa: E201
        [ 0.0, 300.0, 7.0, 7.0],  # noqa: E201
        [ 6.0, 400.0, 7.0],       # noqa: E201
        [ 1.0, 600.0, 7.0, 7.0],  # noqa: E201
    ]


def get_matrix16():
    """
    Return the matrix with ID 16.
    """
    return [
        [ 2.0,  12.0, 7.0, 7.0],  # noqa: E201
        [-4.0, 100.0, 7.0, 7.0],  # noqa: E201
        [10.0, 200.0, 7.0, 7.0],  # noqa: E201
        [ 0.0, 300.0, 7.0, 7.0],  # noqa: E201
        [ 6.0, 400.0, 7.0, 7.0],  # noqa: E201
        [ 1.0, 600.0, 7.0, 7.0],  # noqa: E201
    ]


def get_matrix17():
    """
    Return the matrix with ID 17.
    """
    return [
        [ 2.0,  12.0, 0.0, 7.0],  # noqa: E201
        [ 4.0, 100.0, 0.0, 7.0],  # noqa: E201
        [10.0, 200.0, 0.0, 7.0],  # noqa: E201
        [ 0.0, 300.0, 0.0, 7.0],  # noqa: E201
        [ 6.0, 400.0, 0.0, 7.0],  # noqa: E201
        [ 1.0, 600.0, 0.0, 7.0],  # noqa: E201
    ]


def get_matrix18():
    """
    Return the matrix with ID 18.
    """
    return [
        [ 2.0,  12.0, 7.0, 0.0],  # noqa: E201
        [ 4.0, 100.0, 7.0, 0.0],  # noqa: E201
        [10.0, 200.0, 7.0, 0.0],  # noqa: E201
        [ 0.0, 300.0, 7.0, 0.0],  # noqa: E201
        [ 6.0, 400.0, 7.0, 0.0],  # noqa: E201
        [ 1.0, 600.0, 7.0, 0.0],  # noqa: E201
    ]


def get_matrix19():
    """
    Return the matrix with ID 19.
    """
    return [
        [0.2, 0.8, 1.0, 0.0, 0.3, 0.7],
        [1.0, 0.0, 0.0, 1.0, 0.2, 0.8],
        [0.0, 1.0, 0.1, 0.9, 1.0, 0.0],
        [0.5, 0.5, 0.2, 0.8, 0.5, 0.5],
        [0.1, 0.9, 0.4, 0.6, 0.1, 0.9],
        [0.7, 0.3, 0.6, 0.4, 0.0, 1.0],
    ]


def get_matrix20():
    """
    Return the matrix with ID 20.
    """
    return [
        [ 8.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],  # noqa: E201
        [24.0, 24.0, -11.0, -11.0,   0.0,   0.0],  # noqa: E201
        [ 4.0,  4.0, -10.0, -10.0,  40.0,  40.0],  # noqa: E201
        [14.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],  # noqa: E201
        [ 6.0,  6.0,  -7.0,  -7.0,  -5.0],         # noqa: E201
        [18.0, 18.0,  -5.0,  -5.0, -10.0, -10.0],  # noqa: E201
    ]


def get_matrix21():
    """
    Return the matrix with ID 21.
    """
    return [
        [7.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
        [7.0, 24.0, -11.0, -11.0,   0.0,   0.0],
        [7.0,  4.0, -10.0, -10.0,  40.0,  40.0],
        [7.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
        [7.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
        [7.0, 18.0,  -5.0,  -5.0, -10.0, -10.0],
    ]


def get_matrix22():
    """
    Return the matrix with ID 22.
    """
    return [
        [-7.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
        [-7.0, 24.0, -11.0, -11.0,   0.0,   0.0],
        [-7.0,  4.0, -10.0, -10.0,  40.0,  40.0],
        [-7.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
        [-7.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
        [-7.0, 18.0,  -5.0,  -5.0, -10.0, -10.0],
    ]


def get_matrix23():
    """
    Return the matrix with ID 23.
    """
    return [
        [0.0,  8.0,  -1.0,  -1.0,   5.0,   5.0],
        [0.0, 24.0, -11.0, -11.0,   0.0,   0.0],
        [0.0,  4.0, -10.0, -10.0,  40.0,  40.0],
        [0.0, 14.0,  -9.0,  -9.0,  15.0,  15.0],
        [0.0,  6.0,  -7.0,  -7.0,  -5.0,  -5.0],
        [0.0, 18.0,  -5.0,  -5.0, -10.0, -10.0],
    ]


def get_matrix24():
    """
    Return the matrix with ID 24.
    """
    return [
        [4.0, 4.0, 7.0, 7.0],
        [3.0, 3.0, 7.0, 7.0],
        [2.0, 2.0, 7.0, 7.0],
        [1.0, 1.0, 7.0, 7.0],
        [0.0, 0.0, 7.0, 7.0],
    ]


def get_matrix25():
    """
    Return the matrix with ID 25.
    """
    return [
        [0.4, 0.4, 0.2, 0.2],
        [0.3, 0.3, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.2, 0.2],
        [0.0, 0.0, 0.2, 0.2],
    ]


def get_matrix26():
    """
    Return the matrix with ID 26.
    """
    return [
        [4.0, 4.0, 7.0, 7.0],
        [3.0, 3.0, 7.0, 7.0],
        [2.0, 2.0, 7.0, 7.0],
        [1.0, 1.0, 7.0],
        [0.0, 0.0, 7.0, 7.0],
    ]


def get_matrix27():
    """
    Return the matrix with ID 27.
    """
    return [
        [ 4.0, 4.0, 7.0, 7.0],  # noqa: E201
        [ 3.0, 3.0, 7.0, 7.0],  # noqa: E201
        [-2.0, 2.0, 7.0, 7.0],  # noqa: E201
        [ 1.0, 1.0, 7.0, 7.0],  # noqa: E201
        [ 0.0, 0.0, 7.0, 7.0],  # noqa: E201
    ]


def get_matrix28():
    """
    Return the matrix with ID 28.
    """
    return [
        [4.0, 4.0, 7.0, 0.0],
        [3.0, 3.0, 7.0, 0.0],
        [2.0, 2.0, 7.0, 0.0],
        [1.0, 1.0, 7.0, 0.0],
        [0.0, 0.0, 7.0, 0.0],
    ]


def get_matrix29():
    """
    Return the matrix with ID 29.
    """
    return [
        [0.0, 0.0, 5.0, 5.0],
        [6.0, 6.0, 5.0, 5.0],
        [0.0, 0.0, 5.0, 5.0],
        [8.0, 8.0, 5.0, 5.0],
    ]


def get_matrix30():
    """
    Return the matrix with ID 30.
    """
    return [
        [0.0, 0.0, 0.5, 0.5],
        [0.6, 0.6, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
        [0.8, 0.8, 0.5, 0.5],
    ]


def get_matrix31():
    """
    Return the matrix with ID 31.
    """
    return [
        [0.0, 0.0, 5.0, 5.0],
        [6.0, 6.0, 5.0, 5.0],
        [0.0, 0.0, 5.0],
        [8.0, 8.0, 5.0, 5.0],
    ]


def get_matrix32():
    """
    Return the matrix with ID 32.
    """
    return [
        [0.0,  0.0, 5.0, 5.0],
        [6.0, -6.0, 5.0, 5.0],
        [0.0,  0.0, 5.0, 5.0],
        [8.0,  8.0, 5.0, 5.0],
    ]


def get_matrix33():
    """
    Return the matrix with ID 33.
    """
    return [
        [0.0, 0.0, 5.0, 0.0],
        [6.0, 6.0, 5.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [8.0, 8.0, 5.0, 0.0],
    ]


def get_matrix34():
    """
    Return the matrix with ID 34.
    """
    return [
        [ 1.0000000,  0.9314381, -0.9314381],  # noqa: E201
        [ 0.9314381,  1.0000000, -1.0000000],  # noqa: E201
        [-0.9314381, -1.0000000,  1.0000000],  # noqa: E201
    ]


def get_matrix35():
    """
    Return the matrix with ID 35.
    """
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def get_matrix36():
    """
    Return the matrix with ID 36.
    """
    return [
        [1.0000000, 0.9314381, 0.9314381],
        [0.9314381, 1.0000000, 1.0000000],
        [0.9314381, 1.0000000, 1.0000000],
    ]


def get_matrix37():
    """
    Return the matrix with ID 37.
    """
    return [
        [1.0000000, 0.9369189, 0.9369189],
        [0.9369189, 1.0000000, 1.0000000],
        [0.9369189, 1.0000000, 1.0000000],
    ]


def get_matrix38():
    """
    Return the matrix with ID 38.
    """
    return [
        [1.0000000, 0.5186014, 0.0000000],
        [0.5186014, 1.0000000, 0.0000000],
        [0.0000000, 0.0000000, 1.0000000],
    ]


def get_matrix39():
    """
    Return the matrix with ID 39.
    """
    return [
        [0.0, 0.0],
        [0.0, 1.0],
    ]


def get_matrix40():
    """
    Return the matrix with ID 40.
    """
    return [
        [1.0000000, 0.0000000],
        [0.0000000, 1.0000000],
    ]


def get_matrix41():
    """
    Return the matrix with ID 41.
    """
    return [
        [0.000, 0.000, 0.333],
        [0.033, 0.050, 0.267],
        [0.067, 0.100, 0.200],
        [0.100, 0.175, 0.100],
        [0.200, 0.200, 0.067],
        [0.267, 0.225, 0.033],
        [0.333, 0.250, 0.000],
    ]


def get_matrix42():
    """
    Return the matrix with ID 42.
    """
    return [
        [0.00000000, 0.00000000, 0.00000000],
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
        [0.16666667, 0.00000000, 0.16666667],
    ]


def get_matrix43():
    """
    Return the matrix with ID 43.
    """
    return [
        [0.000, 0.000, 0.333],
        [0.033, 0.050, 0.267],
        [0.067, 0.100, 0.200],
        [0.100, 0.175, 0.100],
        [0.200, 0.200, 0.067],
        [0.267, 0.225],
        [0.333, 0.250, 0.000],
    ]


def get_matrix44():
    """
    Return the matrix with ID 44.
    """
    return [
        [0.000, 0.000, 1.333],
        [0.033, 0.050, 0.267],
        [0.067, 0.100, 0.200],
        [0.100, 0.175, 0.100],
        [0.200, 0.200, 0.067],
        [0.267, 0.225, 0.033],
        [0.333, 0.250, 0.000],
    ]


def get_matrix45():
    """
    Return the matrix with ID 45.
    """
    return [
        [ 0.000, 0.000, 0.333],  # noqa: E201
        [-0.033, 0.050, 0.267],  # noqa: E201
        [ 0.067, 0.100, 0.200],  # noqa: E201
        [ 0.100, 0.175, 0.100],  # noqa: E201
        [ 0.200, 0.200, 0.067],  # noqa: E201
        [ 0.267, 0.225, 0.033],  # noqa: E201
        [ 0.333, 0.250, 0.000],  # noqa: E201
    ]


def get_matrix46():
    """
    Return the matrix with ID 46.
    """
    return [
        [0.000, 0.0, 0.333],
        [0.033, 0.2, 0.267],
        [0.067, 0.4, 0.200],
        [0.100, 0.7, 0.100],
        [0.200, 0.8, 0.067],
        [0.267, 0.9, 0.033],
        [0.333, 1.0, 0.000],
    ]


def get_matrix47():
    """
    Return the matrix with ID 47.
    """
    return [
        [0.00, 1.01],
        [0.25, 0.75],
        [0.50, 0.50],
        [0.75, 0.25],
        [1.00, 0.00],
    ]


def get_matrix48():
    """
    Return the matrix with ID 48.
    """
    return [
        [ 0.00, 1.00],  # noqa: E201
        [-0.25, 0.75],  # noqa: E201
        [ 0.50, 0.50],  # noqa: E201
        [ 0.75, 0.25],  # noqa: E201
        [ 1.00, 0.00],  # noqa: E201
    ]


def get_ranking01():
    """
    Return the ranking with ID 01.
    """
    return [
        ("a1", 0.500000),
        ("a2", 0.500000),
        ("a3", 0.500000),
        ("a4", 0.500000),
        ("a5", 0.500000),
    ]


def get_ranking02():
    """
    Return the ranking with ID 02.
    """
    return [
        ("a5", 0.700000),
        ("a4", 0.600000),
        ("a3", 0.500000),
        ("a2", 0.400000),
        ("a1", 0.300000),
    ]


def get_ranking03():
    """
    Return the ranking with ID 03.
    """
    return [
        ("a1", 0.300000),
        ("a2", 0.400000),
        ("a3", 0.500000),
        ("a4", 0.600000),
        ("a5", 0.700000),
    ]


def get_ranking04():
    """
    Return the ranking with ID 04.
    """
    return [
        ("a2", 0.677778),
        ("a1", 0.669167),
        ("a3", 0.638889),
        ("a6", 0.625000),
        ("a5", 0.590278),
        ("a4", 0.588889),
    ]


def get_ranking05():
    """
    Return the ranking with ID 05.
    """
    return [
        ("a2", 0.653952),
        ("a3", 0.604472),
        ("a1", 0.601574),
        ("a6", 0.595749),
        ("a5", 0.539665),
        ("a4", 0.530537),
    ]


def get_ranking06():
    """
    Return the ranking with ID 06.
    """
    return [
        ("a2", 0.650527),
        ("a1", 0.612074),
        ("a3", 0.599994),
        ("a6", 0.594459),
        ("a5", 0.540496),
        ("a4", 0.537186),
    ]


def get_ranking07():
    """
    Return the ranking with ID 07.
    """
    return [
        ("a2", 0.644440),
        ("a1", 0.623018),
        ("a3", 0.593228),
        ("a6", 0.591963),
        ("a4", 0.543750),
        ("a5", 0.540097),
    ]


def get_ranking08():
    """
    Return the ranking with ID 08.
    """
    return [
        ("a6", 0.583347),
        ("a3", 0.574199),
        ("a5", 0.480220),
        ("a2", 0.469420),
        ("a4", 0.304194),
        ("a1", 0.192606),
    ]


def get_ranking09():
    """
    Return the ranking with ID 09.
    """
    return [
        ("a2", 0.669839),
        ("a5", 0.647361),
        ("a3", 0.645343),
        ("a6", 0.622660),
        ("a4", 0.587153),
        ("a1", 0.471261),
    ]


def get_ranking10():
    """
    Return the ranking with ID 10.
    """
    return [
        ("a2", 0.677366),
        ("a5", 0.675493),
        ("a3", 0.658395),
        ("a6", 0.652317),
        ("a4", 0.622630),
        ("a1", 0.456501),
    ]


def get_ranking11():
    """
    Return the ranking with ID 11.
    """
    return [
        ("a6", 0.983188),
        ("a3", 0.980454),
        ("a5", 0.968182),
        ("a2", 0.967595),
        ("a4", 0.808142),
        ("a1", 0.033316),
    ]


def get_ranking12():
    """
    Return the ranking with ID 12.
    """
    return [
        ("a6", 0.955577),
        ("a5", 0.954078),
        ("a3", 0.938579),
        ("a2", 0.909531),
        ("a4", 0.808416),
        ("a1", 0.096521),
    ]


def get_ranking13():
    """
    Return the ranking with ID 13.
    """
    return [
        ("a5", 0.868655),
        ("a6", 0.846338),
        ("a4", 0.812076),
        ("a3", 0.789327),
        ("a2", 0.718801),
        ("a1", 0.300742),
    ]


def get_ranking14():
    """
    Return the ranking with ID 14.
    """
    return [
        ("a5", 0.836287),
        ("a6", 0.814430),
        ("a4", 0.805387),
        ("a3", 0.745801),
        ("a2", 0.688769),
        ("a1", 0.341532),
    ]


def get_ranking15():
    """
    Return the ranking with ID 15.
    """
    return [
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


def get_ranking16():
    """
    Return the ranking with ID 16.
    """
    return [
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


def get_ranking17():
    """
    Return the ranking with ID 17.
    """
    return [
        ("a3", 0.500000),
        ("a2", 0.433013),
        ("a4", 0.433013),
        ("a1", 0.000000),
        ("a5", 0.000000),
    ]


def get_ranking18():
    """
    Return the ranking with ID 18.
    """
    return [
        ("a5", 0.700000),
        ("a4", 0.650413),
        ("a3", 0.500000),
        ("a2", 0.349587),
        ("a1", 0.300000),
    ]


def get_ranking19():
    """
    Return the ranking with ID 19.
    """
    return [
        ("a5", 1.000000),
        ("a4", 0.750000),
        ("a3", 0.500000),
        ("a2", 0.250000),
        ("a1", 0.000000),
    ]


def get_ranking20():
    """
    Return the ranking with ID 20.
    """
    return [
        ("A", 0.562314),
        ("D", 0.472564),
        ("C", 0.447428),
        ("B", 0.438744),
    ]


def get_ranking21():
    """
    Return the ranking with ID 21.
    """
    return [
        ("C", 0.586404),
        ("A", 0.536356),
        ("B", 0.422726),
        ("D", 0.418160),
    ]


def get_ranking22():
    """
    Return the ranking with ID 22.
    """
    return [
        ("A", 0.567198),
        ("D", 0.473771),
        ("B", 0.440236),
        ("C", 0.439791),
    ]


def get_ranking23():
    """
    Return the ranking with ID 23.
    """
    return [
        ("A", 0.596199),
        ("B", 0.592651),
        ("D", 0.581653),
        ("C", 0.507066),
    ]


def get_vector01():
    """
    Return the vector with ID 01.
    """
    return [
        0.7,
        0.3,
    ]


def get_vector02():
    """
    Return the vector with ID 02.
    """
    return [
        0.3,
        0.2,
        0.4,
        0.1,
    ]


def get_vector03():
    """
    Return the vector with ID 03.
    """
    return [
        0.25,
        0.25,
        0.25,
        0.25,
    ]


def get_vector04():
    """
    Return the vector with ID 04.
    """
    return [
        0.5,
        0.4,
    ]


def get_vector05():
    """
    Return the vector with ID 05.
    """
    return [
        0.5,
        0.5,
    ]


def get_vector06():
    """
    Return the vector with ID 06.
    """
    return [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ]


def get_vector07():
    """
    Return the vector with ID 07.
    """
    return [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
    ]


def get_vector08():
    """
    Return the vector with ID 07.
    """
    return [
        0.54,
        0.5,
        0.46,
    ]


def get_vector09():
    """
    Return the vector with ID 09.
    """
    return [
        0.0000000,
        0.4330127,
        0.5000000,
        0.4330127,
        0.0000000,
    ]


def get_vector10():
    """
    Return the vector with ID 10.
    """
    return [
        0.4418200,
        0.5000000,
        0.3163389,
    ]


def get_vector11():
    """
    Return the vector with ID 11.
    """
    return [
        0.6194425,
        0.5000000,
        0.3805575,
    ]


def get_vector12():
    """
    Return the vector with ID 12.
    """
    return [
        0.3805575,
        0.5000000,
        0.6194425,
    ]


def get_vector13():
    """
    Return the vector with ID 13.
    """
    return [
        0.6177727,
        0.5000000,
        0.3822273,
    ]


def get_vector14():
    """
    Return the vector with ID 14.
    """
    return [
        0.5767680,
        0.5000000,
        0.4232320,
    ]


def get_vector15():
    """
    Return the vector with ID 15.
    """
    return [
        0.4232320,
        0.5000000,
        0.5767680,
    ]


def get_vector16():
    """
    Return the vector with ID 16.
    """
    return [
        0.5714286,
        0.5000000,
        0.4285714,
    ]


def get_vector17():
    """
    Return the vector with ID 17.
    """
    return [
        0.33333333,
        0.33333333,
        0.33333333,
    ]


def get_vector18():
    """
    Return the vector with ID 18.
    """
    return [
        0.37406776,
        0.25186448,
        0.37406776,
    ]


def get_vector19():
    """
    Return the vector with ID 19.
    """
    return [
        0.20724531,
        0.31710188,
        0.47565280,
    ]


def get_vector20():
    """
    Return the vector with ID 20.
    """
    return [
        0.27329284,
        0.32664742,
        0.40005975,
    ]


def get_vector21():
    """
    Return the vector with ID 21.
    """
    return [
        0.25000000,
        0.25857023,
        0.49142977,
    ]


def get_vector22():
    """
    Return the vector with ID 22.
    """
    return [
        0.50000000,
        0.25000000,
        0.25000000,
    ]


def get_vector23():
    """
    Return the vector with ID 23.
    """
    return [
        0.23971980,
        0.28651997,
        0.47376023,
    ]


def get_vector24():
    """
    Return the vector with ID 24.
    """
    return [
        0.33817571,
        0.33091215,
        0.33091215,
    ]


def get_vector25():
    """
    Return the vector with ID 25.
    """
    return [
        0.22633480,
        0.27052183,
        0.50314336,
    ]


def get_vector26():
    """
    Return the vector with ID 26.
    """
    return [
        0.33861310,
        0.33069345,
        0.33069345,
    ]

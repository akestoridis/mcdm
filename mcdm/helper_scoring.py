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
Helper module for the scoring methods of the ``mcdm`` package.
"""

from . import scoring


def score(z_matrix, is_benefit_z, w_vector, s_method):
    """
    Return the selected scores of the provided decision matrix with the
    provided weight vector.
    """
    # Use the selected scoring method
    if s_method.upper() == "SAW":
        return scoring.saw(z_matrix, w_vector, is_benefit_z)
    elif s_method.upper() == "MEW":
        return scoring.mew(z_matrix, w_vector, is_benefit_z)
    elif s_method.upper() == "TOPSIS":
        return scoring.topsis(z_matrix, w_vector, is_benefit_z)
    elif s_method.upper() == "MTOPSIS":
        return scoring.mtopsis(z_matrix, w_vector, is_benefit_z)
    else:
        raise ValueError("Unknown scoring method ({})".format(s_method))

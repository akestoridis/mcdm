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
Helper module for the weighting methods of the ``mcdm`` package.
"""

from . import weighting


def weigh(z_matrix, w_method, c_method=None):
    """
    Return the weight vector of the provided decision matrix using the
    selected weighting method.
    """
    # Use the selected weighting method
    if w_method.upper() == "MW":
        return weighting.mw(z_matrix)
    elif w_method.upper() == "EM":
        return weighting.em(z_matrix)
    elif w_method.upper() == "SD":
        return weighting.sd(z_matrix)
    elif w_method.upper() == "CRITIC":
        return weighting.critic(z_matrix, c_method)
    elif w_method.upper() == "VIC":
        return weighting.vic(z_matrix, c_method)
    else:
        raise ValueError("Unknown weighting method ({})".format(w_method))

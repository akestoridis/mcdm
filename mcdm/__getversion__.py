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

"""
Derivation of the version number for the mcdm package
"""

import os
import re
import subprocess


def getversion(pkg_dirpath):
    version_filepath = os.path.join(pkg_dirpath, "VERSION.txt")
    git_dirpath = os.path.join(os.path.dirname(pkg_dirpath), ".git")
    try:
        cmd = "git --git-dir {} describe --tags".format(git_dirpath)
        cp = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode == 0:
            match = re.search(
                r"^v([0-9]+\.[0-9]+)(\-[0-9]+\-g[0-9a-f]{7})?$",
                cp.stdout.decode().rstrip())
            if match:
                version = match.group(1)
                if match.group(2) is not None:
                    version += "+" + re.search(
                        r"^\-[0-9]+\-g([0-9a-f]{7})$",
                        match.group(2)).group(1)
                with open(version_filepath, "w") as fp:
                    fp.write("{}\n".format(version))
                return version

        cmd = "git --git-dir {} rev-parse --short HEAD".format(git_dirpath)
        cp = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode == 0:
            match = re.search(r"^[0-9a-f]{7}$", cp.stdout.decode().rstrip())
            if match:
                version = "0+" + match.group(0)
                with open(version_filepath, "w") as fp:
                    fp.write("{}\n".format(version))
                return version
    except Exception:
        pass

    if os.path.isfile(version_filepath):
        with open(version_filepath, "r") as fp:
            match = re.search(
                r"^(0\+[0-9a-f]{7}|[0-9]+\.[0-9]+(\+[0-9a-f]{7})?)$",
                fp.read().rstrip())
        if match:
            return match.group(0)

    match = re.search(r"tag: v([0-9]+\.[0-9]+)(,|$)", "$Format:%D$")
    if match:
        return match.group(1)

    match = re.search(r"^[0-9a-f]{7}$", "$Format:%h$")
    if match:
        return "0+" + match.group(0)

    return "0+unknown"

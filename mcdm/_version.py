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
Version number derivation module for the ``mcdm`` package.
"""

import os
import re
import subprocess


def get_version(pkg_dirpath):
    """
    Return the derived version number of the ``mcdm`` package.
    """
    version_filepath = os.path.join(pkg_dirpath, "VERSION.txt")
    git_dirpath = os.path.join(os.path.dirname(pkg_dirpath), ".git")

    version = get_version_from_git(version_filepath, git_dirpath)
    if version is None:
        version = get_version_from_file(version_filepath)
        if version is None:
            version = "0+unknown"

    return version


def get_version_from_git(version_filepath, git_dirpath):
    """
    Try to derive and then return the version number of the ``mcdm`` package
    according to a Git command. If the derivation process succeeds, write the
    derived version number in the provided filepath before returning it.
    """
    try:
        args = [
            "git",
            "--git-dir",
            git_dirpath,
            "describe",
            "--tags",
        ]
        cp = subprocess.run(args, capture_output=True, check=False)
        if cp.returncode == 0:
            match = re.search(
                r"^v([0-9]+\.[0-9]+)(\-[0-9]+\-g[0-9a-f]{7})?$",
                cp.stdout.decode().rstrip(),
            )
            if match:
                version = match.group(1)
                if match.group(2) is not None:
                    version += "+" + re.search(
                        r"^\-[0-9]+\-g([0-9a-f]{7})$",
                        match.group(2),
                    ).group(1)
                write_version_to_file(version_filepath, version)
                return version

        args = [
            "git",
            "--git-dir",
            git_dirpath,
            "rev-parse",
            "--short",
            "HEAD",
        ]
        cp = subprocess.run(args, capture_output=True, check=False)
        if cp.returncode == 0:
            match = re.search(r"^[0-9a-f]{7}$", cp.stdout.decode().rstrip())
            if match:
                version = "0+" + match.group(0)
                write_version_to_file(version_filepath, version)
                return version
    except Exception:  # pylint: disable=broad-except
        return None

    return None


def get_version_from_file(version_filepath):
    """
    Try to derive and then return the version number of the ``mcdm`` package
    according to an existing file.
    """
    if os.path.isfile(version_filepath):
        with open(version_filepath, mode="r", encoding="utf-8") as fp:
            match = re.search(
                r"^(0\+[0-9a-f]{7}|[0-9]+\.[0-9]+(\+[0-9a-f]{7})?)$",
                fp.read().rstrip(),
            )
        if match:
            return match.group(0)

    match = re.search(r"tag: v([0-9]+\.[0-9]+)(,|$)", "$Format:%D$")
    if match:
        return match.group(1)

    match = re.search(r"^[0-9a-f]{7}$", "$Format:%h$")
    if match:
        return "0+" + match.group(0)

    return None


def write_version_to_file(version_filepath, version):
    """
    Write the derived version number of the ``mcdm`` package to a file.
    """
    with open(version_filepath, mode="w", encoding="utf-8") as fp:
        fp.write("{}\n".format(version))

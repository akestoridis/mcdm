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
Setup script for the ``mcdm`` package.
"""

import importlib
import os
import sys

import setuptools


def setup():
    """
    Customize the setup process of the ``mcdm`` package.
    """
    top_dirpath = os.path.dirname(os.path.abspath(__file__))
    pkg_dirpath = os.path.join(top_dirpath, "mcdm")

    metadata = {}
    with open(
        os.path.join(pkg_dirpath, "_metadata.py"),
        mode="r",
        encoding="utf-8",
    ) as fp:
        exec(fp.read(), metadata)  # nosec

    long_description = ""
    with open(
        os.path.join(top_dirpath, "README.md"),
        mode="r",
        encoding="utf-8",
    ) as fp:
        comment_counter = 0
        for line in fp:
            if line == "<!-- START OF BADGES -->\n":
                comment_counter += 1
            elif line == "<!-- END OF BADGES -->\n":
                comment_counter -= 1
            elif comment_counter == 0:
                long_description += line

    version_spec = importlib.util.spec_from_file_location(
        "_version",
        os.path.join(pkg_dirpath, "_version.py"),
    )
    version_module = importlib.util.module_from_spec(version_spec)
    sys.modules["_version"] = version_module
    version_spec.loader.exec_module(version_module)

    setuptools.setup(
        name=metadata["__title__"],
        version=version_module.get_version(pkg_dirpath),
        author=metadata["__author__"],
        author_email=metadata["__author_email__"],
        description=metadata["__description__"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        license=metadata["__license__"],
        url=metadata["__url__"],
        keywords=metadata["__keywords__"],
        classifiers=metadata["__classifiers__"],
        install_requires=metadata["__install_requires__"],
        python_requires=metadata["__python_requires__"],
        include_package_data=True,
        zip_safe=False,
        packages=setuptools.find_packages(),
    )


if __name__ == "__main__":
    setup()

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
Setup script for the mcdm package
"""

import importlib
import os
import setuptools
import sys


top_dirpath = os.path.dirname(os.path.abspath(__file__))
pkg_dirpath = os.path.join(top_dirpath, "mcdm")

about = {}
with open(os.path.join(pkg_dirpath, "__about__.py"), "r") as fp:
    exec(fp.read(), about)

long_description = ""
with open(os.path.join(top_dirpath, "README.md"), "r") as fp:
    long_description = fp.read()

getversion_spec = importlib.util.spec_from_file_location(
    "__getversion__", os.path.join(pkg_dirpath, "__getversion__.py"))
getversion_module = importlib.util.module_from_spec(getversion_spec)
sys.modules["__getversion__"] = getversion_module
getversion_spec.loader.exec_module(getversion_module)

setuptools.setup(
    name=about["__title__"],
    version=getversion_module.getversion(pkg_dirpath),
    author=about["__author__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=about["__license__"],
    url=about["__url__"],
    keywords=about["__keywords__"],
    classifiers=about["__classifiers__"],
    install_requires=about["__install_requires__"],
    python_requires=about["__python_requires__"],
    include_package_data=True,
    zip_safe=False,
    packages=setuptools.find_packages()
)

# Copyright (c) 2020 Dimitrios-Georgios Akestoridis
# This project is licensed under the terms of the MIT license.

import setuptools


about = {}
with open("mcdm/__about__.py", "r") as fp:
    exec(fp.read(), about)

with open("README.md", "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=about["__license__"],
    url=about["__url__"],
    keywords=about["__keywords__"],
    classifiers=about["__classifiers__"],
    install_requires=about["__requires__"],
    python_requires=about["__python_requires__"],
    packages=setuptools.find_packages()
)

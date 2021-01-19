#!/usr/bin/env python

import platform
from setuptools import setup, find_packages


setup(
    name="sparse-update",
    description="",
    version="master",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pytorch-lightning==1.1.4",
        "transformers==4.2.1",
        "datasets==1.2.1",
        "sklearn",
        "scipy==1.5.4",
    ],
    extras_require={
        "test": [
            "coverage",
            "pytest",
            "flake8",
            "pre-commit",
            "codecov",
            "pytest-cov",
            "pytest-flake8",
            "flake8-black",
            "black",
        ]
    },
)
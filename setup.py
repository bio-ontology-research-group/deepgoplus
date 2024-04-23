#!/usr/bin/env python3
import os
import sys

import setuptools.command.egg_info as egg_info_cmd
from setuptools import setup

SETUP_DIR = os.path.dirname(__file__)
README = os.path.join(SETUP_DIR, "README.md")

try:
    import gittaggers

    tagger = gittaggers.EggInfoFromGit
except ImportError:
    tagger = egg_info_cmd.egg_info

install_requires = [
    "click < 8", "scikit-learn", "pandas", "tensorflow < 2.4",
    "numpy < 2.0", "scipy" 
]

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest < 6", "pytest-runner < 5"] if needs_pytest else []

setup(
    name="deepgoplus",
    version="1.0.1",
    description="DeepGOPlus function predictor",
    long_description=open(README).read(),
    long_description_content_type="text/markdown",
    author="Maxat Kulmanov",
    author_email="maxat.kulmanov@kaust.edu.sa",
    download_url="https://github.com/bio-ontology-research-group/deepgoplus/archive/v1.0.1.tar.gz",
    license="Apache 2.0",
    packages=["deepgoplus",],
    package_data={"deepgoplus": [],},
    install_requires=install_requires,
    extras_require={},
    setup_requires=[] + pytest_runner,
    tests_require=["pytest<5"],
    entry_points={
        "console_scripts": [
            "deepgoplus=deepgoplus.main:main",
        ]
    },
    zip_safe=True,
    cmdclass={"egg_info": tagger},
    python_requires=">=3.7, <3.8",
)

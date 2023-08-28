from setuptools import find_packages, setup


# Package meta-data.
NAME = "poisnet"
DESCRIPTION = "Poisoning Network Flow Classifiers"
URL = ""
AUTHOR = "Giorgio Severi"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.1.0"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
)

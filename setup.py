"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
import glob

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = None
with open(path.join(here, 'tensorCache', '__init__.py'), encoding='utf-8') as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version

setup(
    name='tensorCache',
    version=version,
    description='An LRU cache implemented for torch tensors',
    long_description=long_description,  # Optional
    url='https://github.com/rsanchezgarc/tensorCache',
    author='Ruben Sanchez Garcia',  # Optional
    author_email='ruben.sanchez-garcia@stats.ox.ac.uk',  # Optional
    keywords='PyToch torch tensor cache',  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True,  # This line is important to read MANIFEST.in
)

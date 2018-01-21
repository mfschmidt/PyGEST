#!/usr/bin/env python3

""" setup.py

This file is based on the setuptools-based Python Packaging Authority example
found at https://github.com/pypa/sampleproject

See also https://packaging.python.org/en/latest/distributing.html


"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='AllenHumanBrainGeneExpression',  # Required
    # $ pip3 install AllenHumanBrainGeneExpression
    # https://pypi.org/project/AllenHumanBrainGeneExpression/

    version='0.1.0',
    # 0.1.0:
    #        the initial empty version number

    description='A pythonic wrapper to treat Allen Brain data as pandas dataframes and numpy arrays.',

    # The long description is read in from README.rst above.
    # long_description=long_description,

    url='https://github.com/mfschmidt/AllenHumanBrainGeneExpression',

    author='Mike Schmidt',
    author_email='mikeschmidt@schmidtgracen.com',

    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 1 - Planning',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='genetics neuroscience brain',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['pandas', 'numpy', ],

    # Users will be able to install these using the "extras" syntax, e.g.:
    #
    #   $ pip3 install AllenHumanBrainGeneExpression[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={
        'dev': [],
        'test': [],
    },

    # The idea is that this project pulls its own data files from the cloud.
    # It will need to be initialized with a path with ample space available.
    # But it will not need to include files in its distribution.
    package_data={
        # 'sample': ['package_data.dat'],
    },

)
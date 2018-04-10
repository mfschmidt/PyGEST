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
    name='pygest',  # Required
    # $ pip3 install pygest
    # https://pypi.org/project/pygest/

    version='0.5.3',
    # 0.5.0: added shell scripts for easy command-line shortcuts
    # 0.4.0: handle missing files from a bare git data pull
    # 0.3.0: added nipype interface, not yet functional
    # 0.2.0: changed the directory structure to match bids 1.0.2
    #        and make results easier to peruse by a human (I hope)
    # 0.1.0: the initial empty version number

    description='Python Gene Expression Spatial Toolkit.',

    # The long description is read in from README.rst above.
    # long_description=long_description,

    url='https://github.com/mfschmidt/PyGEST',

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
        'Programming Language :: Python :: 3.7',
    ],

    keywords='genetics neuroscience brain microarray',

    packages=find_packages(include=['pygest', 'utility'],
                           exclude=['contrib', 'docs', 'tests']),

    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['pandas',
                      'numpy',
                      'humanize',
                      'requests',
                      'scipy',
                      'matplotlib',
                      'seaborn',
                      'reportlab',
                      'nipype',
                      ],

    # Users will be able to install these using the "extras" syntax, e.g.:
    #
    #   $ pip3 install PyGEST[dev]
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

    # Useful portions of this library are available in a command-line script
    scripts = ['bin/pygest', 'bin/push_all_for_donor', 'bin/null_all_for_donor', ],

)

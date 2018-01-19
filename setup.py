#!/usr/bin/python3

from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gnpy',
    version='0.0.1',
    description='TIP optical network modeling library',
    long_description=long_description,
    url='https://github.com/Telecominfraproject/gnpy',
    author='Telecom Infra Project',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='optics network fiber communication',
    packages=find_packages(exclude=['examples', 'docs', 'tests']),  # Required
    install_requires=['cycler',
                      'decorator',
                      'matplotlib',
                      'networkx',
                      'numpy',
                      'scipy',
                      'pyparsing',
                      'python-dateutil',
                      'pytz',
                      'six',
                      'xlrd']
)

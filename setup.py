#!/usr/bin/python3

from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gnpy',
    version='1.2.0',
    description='route planning and optimization tool for mesh optical networks',
    long_description=long_description,
    long_description_content_type='text/x-rst; charset=UTF-8',
    url='https://github.com/Telecominfraproject/gnpy',
    author='Telecom Infra Project',
    author_email='jan.kundrat@telecominfraproject.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='optics network fiber communication route planning optimization',
    #packages=find_packages(exclude=['examples', 'docs', 'tests']),  # Required
    packages=find_packages(exclude=['docs', 'tests']),  # Required
    install_requires=list(open('requirements.txt'))
)

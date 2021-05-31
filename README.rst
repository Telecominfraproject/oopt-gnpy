.. image:: docs/images/GNPy-banner.png
   :width: 100%
   :align: left
   :alt: GNPy with an OLS system

====================================================================
`gnpy`: mesh optical network route planning and optimization library
====================================================================

|pypi| |pypi-python| |docs| |travis| |contributors| |codacy-quality| |codecov| |doi|

GNPy is an open-source, community-developed library for building route planning and optimization tools in real-world mesh optical networks.
We are a consortium of operators, vendors, and academic researchers sponsored via the `Telecom Infra Project <http://telecominfraproject.com>`_'s `OOPT/PSE <https://telecominfraproject.com/open-optical-packet-transport/>`_ working group.
Together, we are building this tool for rapid development of production-grade route planning tools which is easily extensible to include custom network elements and performant to the scale of real-world mesh optical networks.

Quick Start
-----------

Install either via `Docker <docs/install.rst#install-docker>`__, or as a `Python package <docs/install.rst#install-pip>`__.
Read our `documentation <https://gnpy.readthedocs.io>`__, learn from the demos, and `get in touch with us <https://github.com/Telecominfraproject/oopt-gnpy/discussions>`__.

This example demonstrates how GNPy can be used to check the expected SNR at the end of the line by varying the channel input power:

.. image:: https://telecominfraproject.github.io/oopt-gnpy/docs/images/transmission_main_example.svg
   :width: 100%
   :align: left
   :alt: Running a simple simulation example
   :target: https://asciinema.org/a/252295

GNPy can do much more, including acting as a Path Computation Engine, tracking bandwidth requests, or advising the SDN controller about a best possible path through a large DWDM network.
Learn more about this `in the documentation <https://gnpy.readthedocs.io>`__.

.. |docs| image:: https://readthedocs.org/projects/gnpy/badge/?version=master
  :target: http://gnpy.readthedocs.io/en/master/?badge=master
  :alt: Documentation Status
  :scale: 100%

.. |travis| image:: https://travis-ci.com/Telecominfraproject/oopt-gnpy.svg?branch=master
  :target: https://travis-ci.com/Telecominfraproject/oopt-gnpy
  :alt: Build Status via Travis CI
  :scale: 100%

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3458319.svg
  :target: https://doi.org/10.5281/zenodo.3458319
  :alt: DOI
  :scale: 100%

.. |contributors| image:: https://img.shields.io/github/contributors-anon/Telecominfraproject/oopt-gnpy
  :target: https://github.com/Telecominfraproject/oopt-gnpy/graphs/contributors
  :alt: Code Contributors via GitHub
  :scale: 100%

.. |codacy-quality| image:: https://img.shields.io/lgtm/grade/python/github/Telecominfraproject/oopt-gnpy
  :target: https://lgtm.com/projects/g/Telecominfraproject/oopt-gnpy/
  :alt: Code Quality via LGTM.com
  :scale: 100%

.. |codecov| image:: https://img.shields.io/codecov/c/github/Telecominfraproject/oopt-gnpy
  :target: https://codecov.io/gh/Telecominfraproject/oopt-gnpy
  :alt: Code Coverage via codecov
  :scale: 100%

.. |pypi| image:: https://img.shields.io/pypi/v/gnpy
  :target: https://pypi.org/project/gnpy/
  :alt: Install via PyPI
  :scale: 100%

.. |pypi-python| image:: https://img.shields.io/pypi/pyversions/gnpy
  :target: https://pypi.org/project/gnpy/
  :alt: Python versions
  :scale: 100%

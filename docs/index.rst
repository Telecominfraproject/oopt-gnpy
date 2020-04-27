.. gnpy documentation master file, created by
   sphinx-quickstart on Mon Dec 18 14:41:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gnpy's documentation!
================================

**gnpy is an open-source, community-developed library for building route planning
and optimization tools in real-world mesh optical networks.**

`gnpy <http://github.com/telecominfraproject/gnpy>`_ is:

- a sponsored project of the `OOPT/PSE <http://telecominfraproject.com/project-groups-2/backhaul-projects/open-optical-packet-transport/>`_ working group of the `Telecom Infra Project <http://telecominfraproject.com>`_.
- fully community-driven, fully open source library
- driven by a consortium of operators, vendors, and academic researchers
- intended for rapid development of production-grade route planning tools
- easily extensible to include custom network elements
- performant to the scale of real-world mesh optical networks

Physical Model
==============

- Goal is to build an end-to-end simulation environment which defines the
  network models of the optical device transfer functions and their parameters.
  This environment will provide validation of the optical performance
  requirements for the TIP OLS building blocks.
- The model may be approximate or complete depending on the network complexity.
  Each model shall be validated against the proposed network scenario.
- The environment must be able to process network models from multiple vendors,
  and also allow users to pick any implementation in an open source framework.
- The PSE will influence and benefit from the innovation of the DTC, API, and
  OLS working groups.
- The PSE represents a step along the journey towards multi-layer optimization.

The following pages are meant to describe specific implementation details and
modeling assumptions behind gnpy. 

.. toctree::
   :maxdepth: 2

   model

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


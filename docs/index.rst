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

Documentation
=============

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

Contributors in alphabetical order
==================================
+----------+------------+-----------------------+--------------------------------------+
| Name     | Surname    | Affiliation           | Contact                              |
+==========+============+=======================+======================================+
| Alessio  | Ferrari    | Politecnico di Torino | alessio.ferrari@polito.it            |
+----------+------------+-----------------------+--------------------------------------+
| Anders   | Lindgren   | Telia Company         | Anders.X.Lindgren@teliacompany.com   |
+----------+------------+-----------------------+--------------------------------------+
| Andrea   | d'Amico    | Politecnico di Torino | andrea.damico@polito.it              |
+----------+------------+-----------------------+--------------------------------------+
| Brian    | Taylor     | Facebook              | briantaylor@fb.com                   |
+----------+------------+-----------------------+--------------------------------------+
| David    | Boertjes   | Ciena                 | dboertje@ciena.com                   |
+----------+------------+-----------------------+--------------------------------------+
| Diego    | Landa      | Facebook              | dlanda@fb.com                        |
+----------+------------+-----------------------+--------------------------------------+
| Esther   | Le Rouzic  | Orange                | esther.lerouzic@orange.com           |
+----------+------------+-----------------------+--------------------------------------+
| Gabriele | Galimberti | Cisco                 | ggalimbe@cisco.com                   |
+----------+------------+-----------------------+--------------------------------------+
| Gert     | Grammel    | Juniper Networks      | ggrammel@juniper.net                 |
+----------+------------+-----------------------+--------------------------------------+
| Gilad    | Goldfarb   | Facebook              | giladg@fb.com                        |
+----------+------------+-----------------------+--------------------------------------+
| James    | Powell     | Telecom Infra Project | james.powell@telecominfraproject.com |
+----------+------------+-----------------------+--------------------------------------+
| Jan      | Kundrát    | Telecom Infra Project | jan.kundrat@telecominfraproject.com  |
+----------+------------+-----------------------+--------------------------------------+
| Jeanluc  | Augé       | Orange                | jeanluc.auge@orange.com              |
+----------+------------+-----------------------+--------------------------------------+
| Jonas    | Mårtensson | RISE Research Sweden  | jonas.martensson@ri.se               |
+----------+------------+-----------------------+--------------------------------------+
| Mattia   | Cantono    | Politecnico di Torino | mattia.cantono@polito.it             |
+----------+------------+-----------------------+--------------------------------------+
| Miguel   | Garrich    | University Catalunya  | miquel.garrich@upct.es               |
+----------+------------+-----------------------+--------------------------------------+
| Raj      | Nagarajan  | Lumentum              | raj.nagarajan@lumentum.com           |
+----------+------------+-----------------------+--------------------------------------+
| Roberts  | Miculens   | Lattelecom            | roberts.miculens@lattelecom.lv       |
+----------+------------+-----------------------+--------------------------------------+
| Shengxiang | Zhu      | University of Arizona | szhu@email.arizona.edu               |
+----------+------------+-----------------------+--------------------------------------+
| Stefan   | Melin      | Telia Company         | Stefan.Melin@teliacompany.com        |
+----------+------------+-----------------------+--------------------------------------+
| Vittorio | Curri      | Politecnico di Torino | vittorio.curri@polito.it             |
+----------+------------+-----------------------+--------------------------------------+
| Xufeng   | Liu        | Jabil                 | xufeng_liu@jabil.com                 |
+----------+------------+-----------------------+--------------------------------------+

--------------

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


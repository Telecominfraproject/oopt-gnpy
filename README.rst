====
gnpy
====


|docs| |build|                

Gaussian Noise (GN) modeling library


* Free software: BSD license
* Documentation: https://gnpy.readthedocs.io.


Summary
--------

We believe that openly sharing ideas, specifications, and other intellectual property is the key to maximizing innovation and reducing complexity

PSE WG Charter
--------------

- Goal is to build an end-to-end simulation environment which defines the network models of the optical device transfer functions and their parameters. This environment will provide validation of the optical performance requirements for the TIP OLS building blocks.   
- The model may be approximate or complete depending on the network complexity. Each model shall be validated against the proposed network scenario. 
- The environment must be able to process network models from multiple vendors, and also allow users to pick any implementation in an open source framework. 
- The PSE will influence and benefit from the innovation of the DTC, API, and OLS working groups.
- The PSE represents a step along the journey towards multi-layer optimization.

Features
--------

* GNPY simulation of an amplified optical link


============
Background
============

Data Centers are built upon interchangeable, highly standardized node and network architectures rather than a sum of isolated solutions. This also translates to optical networking. It leads to a push in enabling multi-vendor optical network by disaggregating HW and SW functions and focussing on interoperability. In this paradigm, the burden of responsability for ensuring the performance of such disaggregated open optical systems falls on the operators. Consequently, operators and vendors are collaborating in defining control models that can be readily used by off-the-shelf controllers. However, node and network models are only part of the answer. To take reasonable decisions, controllers need to incorporate logic to simulate and assess optical performance. Hence, a vendor-independent optical quality estimator is required. Given its vendor-agnostic nature, such an estimator needs to be driven by a consortium of operators, system and component suppliers. 


Founded in February 2016, the Telcominfraproject (TIP)is an engineering-focused initiative which is operator driven, but features collaboration across operators, suppliers, developers, integrators, and startups with the goal of disaggregating the traditional network deployment approach. The group’s ultimate goal is to help provide better connectivity for communities all over the world as more people come on-line and demand more bandwidth- intensive experiences like video, virtual reality and augmented reality. 
Within TIP, the Open Optical Packet Transport (OOPT) project group is chartered with unbundling monolithic packet-optical network technologies in order to unlock innovation and support new, more flexible connectivity paradigms. 
The key to unbundling is the ability to accurately plan and predict the performance of optical line systems based on an accurate simulation of optical parameters. Under that OOPT umbrella, the Physical Simulation Environment (PSE) working group set out to disrupt the planning landscape by providing an open source simulation model which can be used freely across multiple vendor implementations.





.. |docs| image:: https://readthedocs.org/projects/gnpy/badge/?version=develop
  :target: http://gnpy.readthedocs.io/en/develop/?badge=develop
  :alt: Documentation Status
  :scale: 100%

.. |build| image:: https://travis-ci.org/mcantono/gnpy.svg?branch=develop
  :target: https://travis-ci.org/mcantono/gnpy
  :alt: Build Status
  :scale: 100%

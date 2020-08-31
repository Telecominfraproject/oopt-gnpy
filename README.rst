.. image:: docs/images/GNPy-banner.png
   :width: 100%
   :align: left
   :alt: GNPy with an OLS system

====================================================================
`gnpy`: mesh optical network route planning and optimization library
====================================================================

|pypi| |docs| |travis| |doi| |contributors| |codacy-quality| |codecov|

**`gnpy` is an open-source, community-developed library for building route
planning and optimization tools in real-world mesh optical networks.**

`gnpy <http://github.com/telecominfraproject/oopt-gnpy>`__ is:
--------------------------------------------------------------

- a sponsored project of the `OOPT/PSE <https://telecominfraproject.com/open-optical-packet-transport/>`_ working group of the `Telecom Infra Project <http://telecominfraproject.com>`_
- fully community-driven, fully open source library
- driven by a consortium of operators, vendors, and academic researchers
- intended for rapid development of production-grade route planning tools
- easily extensible to include custom network elements
- performant to the scale of real-world mesh optical networks

Documentation: https://gnpy.readthedocs.io

Get In Touch
~~~~~~~~~~~~

There are `weekly calls <https://telecominfraproject.workplace.com/events/702894886867547/>`__ about our progress.
Newcomers, users and telecom operators are especially welcome there.
We encourage all interested people outside the TIP to `join the project <https://telecominfraproject.com/apply-for-membership/>`__.

How to Install
--------------

Install either via `Docker <docs/install.rst#install-docker>`__, or as a `Python package <docs/install.rst#install-pip>`__.

Instructions for First Use
--------------------------

``gnpy`` is a library for building route planning and optimization tools.

It ships with a number of example programs. Release versions will ship with
fully-functional programs.

    **Note**: *If you are a network operator or involved in route planning and
    optimization for your organization, please contact project maintainer Jan
    Kundrát <jan.kundrat@telecominfraproject.com>. gnpy is looking for users with
    specific, delineated use cases to drive requirements for future
    development.*

This example demonstrates how GNPy can be used to check the expected SNR at the end of the line by varying the channel input power:

.. image:: https://telecominfraproject.github.io/oopt-gnpy/docs/images/transmission_main_example.svg
   :width: 100%
   :align: left
   :alt: Running a simple simulation example
   :target: https://asciinema.org/a/252295

By default, this script operates on a single span network defined in
`gnpy/example-data/edfa_example_network.json <gnpy/example-data/edfa_example_network.json>`_

You can specify a different network at the command line as follows. For
example, to use the CORONET Global network defined in
`gnpy/example-data/CORONET_Global_Topology.json <gnpy/example-data/CORONET_Global_Topology.json>`_:

.. code-block:: shell-session

    $ gnpy-transmission-example $(gnpy-example-data)/CORONET_Global_Topology.json

It is also possible to use an Excel file input (for example
`gnpy/example-data/CORONET_Global_Topology.xlsx <gnpy/example-data/CORONET_Global_Topology.xlsx>`_).
The Excel file will be processed into a JSON file with the same prefix.
Further details about the Excel data structure are available `in the documentation <docs/excel.rst>`__.

The main transmission example will calculate the average signal OSNR and SNR
across network elements (transceiver, ROADMs, fibers, and amplifiers)
between two transceivers selected by the user. Additional details are provided by doing ``gnpy-transmission-example -h``. (By default, for the CORONET Global
network, it will show the transmission of spectral information between Abilene and Albany)

This script calculates the average signal OSNR = |OSNR| and SNR = |SNR|.

.. |OSNR| replace:: P\ :sub:`ch`\ /P\ :sub:`ase`
.. |SNR| replace:: P\ :sub:`ch`\ /(P\ :sub:`nli`\ +\ P\ :sub:`ase`)

|Pase| is the amplified spontaneous emission noise, and |Pnli| the non-linear
interference noise.

.. |Pase| replace:: P\ :sub:`ase`
.. |Pnli| replace:: P\ :sub:`nli`

Further Instructions for Use
----------------------------

Simulations are driven by a set of `JSON <docs/json.rst>`__ or `XLS <docs/excel.rst>`__ files.

The ``gnpy-transmission-example`` script propagates a spectrum of channels at 32 Gbaud, 50 GHz spacing and 0 dBm/channel. 
Launch power can be overridden by using the ``--power`` argument.
Spectrum information is not yet parametrized but can be modified directly in the ``eqpt_config.json`` (via the ``SpectralInformation`` -SI- structure) to accommodate any baud rate or spacing.
The number of channel is computed based on ``spacing`` and ``f_min``, ``f_max`` values.

An experimental support for Raman amplification is available:

.. code-block:: shell-session

     $ gnpy-transmission-example \
       $(gnpy-example-data)/raman_edfa_example_network.json \
       --sim $(gnpy-example-data)/sim_params.json --show-channels

Configuration of Raman pumps (their frequencies, power and pumping direction) is done via the `RamanFiber element in the network topology <gnpy/example-data/raman_edfa_example_network.json>`_.
General numeric parameters for simulaiton control are provided in the `gnpy/example-data/sim_params.json <gnpy/example-data/sim_params.json>`_.

Use ``gnpy-path-request`` to request several paths at once:

.. code-block:: shell-session

     $ cd $(gnpy-example-data)
     $ gnpy-path-request -o output_file.json \
       meshTopologyExampleV2.xls meshTopologyExampleV2_services.json

This program operates on a network topology (`JSON <docs/json.rst>`__ or `Excel <docs/excel.rst>`__ format), processing the list of service requests (JSON or XLS again).
The service requests and reply formats are based on the `draft-ietf-teas-yang-path-computation-01 <https://tools.ietf.org/html/draft-ietf-teas-yang-path-computation-01>`__ with custom extensions (e.g., for transponder modes).
An example of the JSON input is provided in file `service-template.json`, while results are shown in `path_result_template.json`.

Important note: ``gnpy-path-request`` is not a network dimensionning tool: each service does not reserve spectrum, or occupy ressources such as transponders. It only computes path feasibility assuming the spectrum (between defined frequencies) is loaded with "nb of channels" spaced by "spacing" values as specified in the system parameters input in the service file, each cannel having the same characteristics in terms of baudrate, format,... as the service transponder. The transceiver element acts as a "logical starting/stopping point" for the spectral information propagation. At that point it is not meant to represent the capacity of add drop ports.
As a result transponder type is not part of the network info. it is related to the list of services requests.

The current version includes a spectrum assigment features that enables to compute a candidate spectrum assignment for each service based on a first fit policy. Spectrum is assigned based on service specified spacing value, path_bandwidth value and selected mode for the transceiver. This spectrum assignment includes a basic capacity planning capability so that the spectrum resource is limited by the frequency min and max values defined for the links. If the requested services reach the link spectrum capacity, additional services feasibility are computed but marked as blocked due to spectrum reason.


Contributing
------------

``gnpy`` is looking for additional contributors, especially those with experience
planning and maintaining large-scale, real-world mesh optical networks.

To get involved, please contact Jan Kundrát
<jan.kundrat@telecominfraproject.com> or Gert Grammel <ggrammel@juniper.net>.

``gnpy`` contributions are currently limited to members of `TIP
<http://telecominfraproject.com>`_. Membership is free and open to all.

See the `Onboarding Guide
<https://github.com/Telecominfraproject/gnpy/wiki/Onboarding-Guide>`_ for
specific details on code contributions, or just `upload patches to our Gerrit
<https://review.gerrithub.io/Documentation/intro-gerrit-walkthrough-github.html>`_.

See `AUTHORS.rst <AUTHORS.rst>`_ for past and present contributors.

Project Background
------------------

Data Centers are built upon interchangeable, highly standardized node and
network architectures rather than a sum of isolated solutions. This also
translates to optical networking. It leads to a push in enabling multi-vendor
optical network by disaggregating HW and SW functions and focusing on
interoperability. In this paradigm, the burden of responsibility for ensuring
the performance of such disaggregated open optical systems falls on the
operators. Consequently, operators and vendors are collaborating in defining
control models that can be readily used by off-the-shelf controllers. However,
node and network models are only part of the answer. To take reasonable
decisions, controllers need to incorporate logic to simulate and assess optical
performance. Hence, a vendor-independent optical quality estimator is required.
Given its vendor-agnostic nature, such an estimator needs to be driven by a
consortium of operators, system and component suppliers.

Founded in February 2016, the Telecom Infra Project (TIP) is an
engineering-focused initiative which is operator driven, but features
collaboration across operators, suppliers, developers, integrators, and
startups with the goal of disaggregating the traditional network deployment
approach. The group’s ultimate goal is to help provide better connectivity for
communities all over the world as more people come on-line and demand more
bandwidth- intensive experiences like video, virtual reality and augmented
reality.

Within TIP, the Open Optical Packet Transport (OOPT) project group is chartered
with unbundling monolithic packet-optical network technologies in order to
unlock innovation and support new, more flexible connectivity paradigms.

The key to unbundling is the ability to accurately plan and predict the
performance of optical line systems based on an accurate simulation of optical
parameters. Under that OOPT umbrella, the Physical Simulation Environment (PSE)
working group set out to disrupt the planning landscape by providing an open
source simulation model which can be used freely across multiple vendor
implementations.

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


TIP OOPT/PSE & PSE WG Charter
-----------------------------

We believe that openly sharing ideas, specifications, and other intellectual
property is the key to maximizing innovation and reducing complexity

TIP OOPT/PSE's goal is to build an end-to-end simulation environment which
defines the network models of the optical device transfer functions and their
parameters.  This environment will provide validation of the optical
performance requirements for the TIP OLS building blocks.

- The model may be approximate or complete depending on the network complexity.
  Each model shall be validated against the proposed network scenario.
- The environment must be able to process network models from multiple vendors,
  and also allow users to pick any implementation in an open source framework.
- The PSE will influence and benefit from the innovation of the DTC, API, and
  OLS working groups.
- The PSE represents a step along the journey towards multi-layer optimization.

License
-------

``gnpy`` is distributed under a standard BSD 3-Clause License.

See `LICENSE <LICENSE>`__ for more details.

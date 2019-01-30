====================================================================
`gnpy`: mesh optical network route planning and optimization library
====================================================================

|docs| |build|

**`gnpy` is an open-source, community-developed library for building route
planning and optimization tools in real-world mesh optical networks.**

`gnpy <http://github.com/telecominfraproject/oopt-gnpy>`__ is:

- a sponsored project of the `OOPT/PSE <https://telecominfraproject.com/open-optical-packet-transport/>`_ working group of the `Telecom Infra Project <http://telecominfraproject.com>`_
- fully community-driven, fully open source library
- driven by a consortium of operators, vendors, and academic researchers
- intended for rapid development of production-grade route planning tools
- easily extensible to include custom network elements
- performant to the scale of real-world mesh optical networks

Documentation: https://gnpy.readthedocs.io

Branches and Tagged Releases
----------------------------

- the `master <https://github.com/Telecominfraproject/oopt-gnpy/tree/master>`_ branch contains stable, validated code. It is updated from develop on a release schedule determined by the OOPT-PSE Working Group. For more information about the validation process, see: https://github.com/Telecominfraproject/oopt-gnpy/wiki/Testing-for-Quality
- the `develop <https://github.com/Telecominfraproject/oopt-gnpy/tree/develop>`_ branch contains the latest code under active development, which may not be fully validated and tested.
- the `phase-1 <https://github.com/Telecominfraproject/oopt-gnpy/tree/phase-1>`_ branch contains code for Phase I of the OOPT-PSE efforts and is kept only for reference. This branch is unmaintained.

A brief outline of major (tagged) `gnpy` releases:

+---------------+-------------+-----------------------------------------------+
| release date  | version tag | notes                                         |
+===============+=============+===============================================+
| Jan 30, 2019  | v1.1        | - XLS parser enhancements                     |
|               |             | - carrier probe feature                       |
|               |             | - bug fixes                                   |
+---------------+-------------+-----------------------------------------------+
| Oct 16, 2018  | v1.0        | - first "production"-ready release            |
|               |             | - open network element model (EDFA, GN-model) |
|               |             | - auto-design functionality                   |
|               |             | - path request functionality                  |
+---------------+-------------+-----------------------------------------------+

How to Install
--------------

   **Note**: `gnpy` supports Python 3 only. Python 2 is not supported.
   `gnpy` requires Python ≥3.6

   **Note**: the `gnpy` maintainers strongly recommend the use of Anaconda for
   managing dependencies.

It is recommended that you use a "virtual environment" when installing `gnpy`.
Do not install `gnpy` on your system Python.

We recommend the use of the Anaconda Python distribution
(https://www.anaconda.com/download) which comes with many scientific computing
dependencies pre-installed. Anaconda creates a base "virtual environment" for
you automatically. You can also create and manage your conda "virtual
environments" yourself (see:
https://conda.io/docs/user-guide/tasks/manage-environments.html)

To activate your Anaconda virtual environment, you may need to do the
following:

.. code-block:: shell

    $ source /path/to/anaconda/bin/activate # activate Anaconda base environment
    (base) $                                # note the change to the prompt

You can check which Anaconda environment you are using with:

.. code-block:: shell

    (base) $ conda env list                          # list all environments
    # conda environments:
    #
    base                  *  /src/install/anaconda3

    (base) $ echo $CONDA_DEFAULT_ENV                 # show default environment
    base

You can check your version of Python with the following. If you are using
Anaconda's Python 3, you should see similar output as below. Your results may
be slightly different depending on your Anaconda installation path and the
exact version of Python you are using.

.. code-block:: shell

    $ which python                   # check which Python executable is used
    /path/to/anaconda/bin/python
    $ python -V                      # check your Python version
    Python 3.6.5 :: Anaconda, Inc.

From within your Anaconda Python 3 environment, you can clone the master branch
of the `gnpy` repo and install it with:

.. code-block:: shell

    $ git clone https://github.com/Telecominfraproject/oopt-gnpy # clone the repo
    $ cd oopt-gnpy
    $ python setup.py install                                    # install

To test that `gnpy` was successfully installed, you can run this command. If it
executes without a `ModuleNotFoundError`, you have successfully installed
`gnpy`.

.. code-block:: shell

    $ python -c 'import gnpy' # attempt to import gnpy

    $ cd oopt-gnpy
    $ pytest                  # run tests

Instructions for First Use
--------------------------

``gnpy`` is a library for building route planning and optimization tools.

It ships with a number of example programs. Release versions will ship with
fully-functional programs.

    **Note**: *If you are a network operator or involved in route planning and
    optimization for your organization, please contact project maintainer James
    Powell <james.powell@telecominfraproject>. gnpy is looking for users with
    specific, delineated use cases to drive requirements for future
    development.*

**To get started, run the main transmission example:**

    **Note**: *Examples should be run from the examples/ folder.*

.. code-block:: shell
    $ pwd
    /path/to/oopt-gnpy
    $ cd examples
    $ python transmission_main_example.py

By default, this script operates on a single span network defined in
`examples/edfa_example_network.json <examples/edfa_example_network.json>`_

You can specify a different network at the command line as follows. For
example, to use the CORONET Global network defined in
`examples/CORONET_Global_Topology.json <examples/CORONET_Global_Topology.json>`_:

.. code-block:: shell

    $ cd examples
    $ python transmission_main_example.py CORONET_Global_Topology.json

It is also possible to use an Excel file input (for example
`examples/CORONET_Global_Topology.xls <examples/CORONET_Global_Topology.xls>`_).
The Excel file will be processed into a JSON file with the same prefix. For
further instructions on how to prepare the Excel input file, see
`Excel_userguide.rst <Excel_userguide.rst>`_.

The main transmission example will calculate the average signal OSNR and SNR
across network elements (transceiver, ROADMs, fibers, and amplifiers)
between two transceivers selected by the user. Additional details are provided by doing ``transmission_main_example.py -h``. (By default, for the CORONET Global
network, it will show the transmission of spectral information between Abilene and Albany)

This script calculates the average signal OSNR = |OSNR| and SNR = |SNR|.

.. |OSNR| replace:: P\ :sub:`ch`\ /P\ :sub:`ase`
.. |SNR| replace:: P\ :sub:`ch`\ /(P\ :sub:`nli`\ +\ P\ :sub:`ase`)

|Pase| is the amplified spontaneous emission noise, and |Pnli| the non-linear
interference noise.

.. |Pase| replace:: P\ :sub:`ase`
.. |Pnli| replace:: P\ :sub:`nli`

Further Instructions for Use (`transmission_main_example.py`, `path_requests_run.py`)
-------------------------------------------------------------------------------------

Design and transmission parameters are defined in a dedicated json file. By
default, this information is read from `examples/eqpt_config.json
<examples/eqpt_config.json>`_. This file defines the equipment libraries that
can be customized (EDFAs, fibers, and transceivers).

It also defines the simulation parameters (spans, ROADMs, and the spectral
information to transmit.)

The EDFA equipment library is a list of supported amplifiers. New amplifiers
can be added and existing ones removed. Three different noise models are available:

1. `'type_def': 'variable_gain'` is a simplified model simulating a 2-coil EDFA with internal, input and output VOAs. The NF vs gain response is calculated accordingly based on the input parameters: `nf_min`, `nf_max`, and `gain_flatmax`. It is not a simple interpolation but a 2-stage NF calculation.
2. `'type_def': 'fixed_gain'` is a fixed gain model.  `NF == Cte == nf0` if `gain_min < gain < gain_flatmax`
3. `'type_def': None` is an advanced model. A detailed json configuration file is required (by default `examples/std_medium_gain_advanced_config.json <examples/std_medium_gain_advanced_config.json>`_.) It uses a 3rd order polynomial where NF = f(gain), NF_ripple = f(frequency), gain_ripple = f(frequency), N-array dgt = f(frequency). Compared to the previous models, NF ripple and gain ripple are modelled.

For all amplifier models:

+----------------------+-----------+-----------------------------------------+
| field                |   type    | description                             |
+======================+===========+=========================================+
| `type_variety`       | (string)  | a unique name to ID the amplifier in the|
|                      |           | JSON/Excel template topology input file |
+----------------------+-----------+-----------------------------------------+
| `out_voa_auto`       | (boolean) | auto_design feature to optimize the     |
|                      |           | amplifier output VOA. If true, output   |
|                      |           | VOA is present and will be used to push |
|                      |           | amplifier gain to its maximum, within   |
|                      |           | EOL power margins.                      |
+----------------------+-----------+-----------------------------------------+
| `allowed_for_design` | (boolean) | If false, the amplifier will not be     |
|                      |           | picked by auto-design but it can still  |
|                      |           | be used as a manual input (from JSON or |
|                      |           | Excel template topology files.)         |
+----------------------+-----------+-----------------------------------------+

The fiber library currently describes SSMF and NZDF but additional fiber types can be entered by the user following the same model:

+----------------------+-----------+-----------------------------------------+
| field                | type      | description                             |
+======================+===========+=========================================+
| `type_variety`       | (string)  | a unique name to ID the fiber in the    |
|                      |           | JSON or Excel template topology input   |
|                      |           | file                                    |
+----------------------+-----------+-----------------------------------------+
| `dispersion`         | (number)  | (s.m-1.m-1)                             |
+----------------------+-----------+-----------------------------------------+
| `gamma`              | (number)  | 2pi.n2/(lambda*Aeff) (w-2.m-1)          |
+----------------------+-----------+-----------------------------------------+

The transceiver equipment library is a list of supported transceivers. New
transceivers can be added and existing ones removed at will by the user. It is
used to determine the service list path feasibility when running the
path_request_run.py routine.

+----------------------+-----------+-----------------------------------------+
| field                | type      | description                             |
+======================+===========+=========================================+
|  `type_variety`      | (string)  | a unique name to ID the transceiver in  |
|                      |           | the JSON or Excel template topology     |
|                      |           | input file                              |
+----------------------+-----------+-----------------------------------------+
|  `frequency`         | (number)  | Min/max as below.                       |
+----------------------+-----------+-----------------------------------------+
|  `mode`              | (number)  | a list of modes supported by the        |
|                      |           | transponder. New modes can be added at  |
|                      |           | will by the user. The modes are specific|
|                      |           | to each transponder type_variety.       |
|                      |           | Each mode is described as below.        |
+----------------------+-----------+-----------------------------------------+

The modes are defined as follows:

+----------------------+-----------+-----------------------------------------+
| field                | type      | description                             |
+======================+===========+=========================================+
| `format`             | (string)  | a unique name to ID the mode.           |
+----------------------+-----------+-----------------------------------------+
| `baud_rate`          | (number)  | in Hz                                   |
+----------------------+-----------+-----------------------------------------+
| `OSNR`               | (number)  | min required OSNR in 0.1nm (dB)         |
+----------------------+-----------+-----------------------------------------+
| `bit_rate`           | (number)  | in bit/s                                |
+----------------------+-----------+-----------------------------------------+
| `roll_off`           | (number)  | Not used.                               |
+----------------------+-----------+-----------------------------------------+
| `tx_osnr`            | (number)  | In dB. OSNR out from transponder.       |
+----------------------+-----------+-----------------------------------------+
| `cost`               | (number)  | Arbitrary unit                          |
+----------------------+-----------+-----------------------------------------+

Simulation parameters are defined as follows.

Auto-design automatically creates EDFA amplifier network elements when they are
missing, after a fiber, or between a ROADM and a fiber. This auto-design
functionality can be manually and locally deactivated by introducing a `Fused`
network element after a `Fiber` or a `Roadm` that doesn't need amplification.
The amplifier is chosen in the EDFA list of the equipment library based on
gain, power, and NF criteria. Only the EDFA that are marked
`'allowed_for_design': true` are considered.

For amplifiers defined in the topology JSON input but whose gain = 0
(placeholder), auto-design will set its gain automatically: see `power_mode` in
the `Spans` library to find out how the gain is calculated.

Span configuration is performed as follows. It is not a list (which may change
in later releases) and the user can only modify the value of existing
parameters:

+------------------------+-----------+---------------------------------------------+
| field                  | type      | description                                 |
+========================+===========+=============================================+
| `power_mode`           | (boolean) | If false, gain mode. Auto-design sets       |
|                        |           | amplifier gain = preceding span loss,      |
|                        |           | unless the amplifier exists and its         |
|                        |           | gain > 0 in the topology input json.        |
|                        |           | If true, power mode (recommended for        |
|                        |           | auto-design and power sweep.)               |
|                        |           | Auto-design sets amplifier power            |
|                        |           | according to delta_power_range. If the      |
|                        |           | amplifier exists with gain > 0 in the       |
|                        |           | topology json input, then its gain is       |
|                        |           | translated into a power target/channel.     |
|                        |           | Moreover, when performing a power sweep     |
|                        |           | (see power_range_db in the SI               |
|                        |           | configuration library) the power sweep      |
|                        |           | is performed w/r/t this power target,       |
|                        |           | regardless of preceding amplifiers         |
|                        |           | power saturation/limitations.               |
+------------------------+-----------+---------------------------------------------+
| `delta_power_range_db` | (number)  | Auto-design only, power-mode                |
|                        |           | only. Specifies the [min, max, step]        |
|                        |           | power excursion/span. It is a relative      |
|                        |           | power excursion w/r/t the                   |
|                        |           | power_dbm + power_range_db                  |
|                        |           | (power sweep if applicable) defined in      |
|                        |           | the SI configuration library. This          |
|                        |           | relative power excursion is = 1/3 of        |
|                        |           | the span loss difference with the           |
|                        |           | reference 20 dB span. The 1/3 slope is      |
|                        |           | derived from the GN model equations.        |
|                        |           | For example, a 23 dB span loss will be      |
|                        |           | set to 1 dB more power than a 20 dB         |
|                        |           | span loss. The 20 dB reference spans        |
|                        |           | will *always* be set to                     |
|                        |           | power = power_dbm + power_range_db.         |
|                        |           | To configure the same power in all          |
|                        |           | spans, use `[0, 0, 0]`. All spans will      |
|                        |           | be set to                                   |
|                        |           | power = power_dbm + power_range_db.         |
|                        |           | To configure the same power in all spans    |
|                        |           | and 3 dB more power just for the longest    |
|                        |           | spans: `[0, 3, 3]`. The longest spans are   |
|                        |           | set to                                      |
|                        |           | power = power_dbm + power_range_db + 3.     |
|                        |           | To configure a 4 dB power range across      |
|                        |           | all spans in 0.5 dB steps: `[-2, 2, 0.5]`.  |
|                        |           | A 17 dB span is set to                      |
|                        |           | power = power_dbm + power_range_db - 1,     |
|                        |           | a 20 dB span to                             |
|                        |           | power = power_dbm + power_range_db and      |
|                        |           | a 23 dB span to                             |
|                        |           | power = power_dbm + power_range_db + 1      |
+------------------------+-----------+---------------------------------------------+
| `max_length`           | (number)  | Split fiber lengths > max_length.           |
|                        |           | Interest to support high level              |
|                        |           | topologies that do not specify in line      |
|                        |           | amplification sites. For example the        |
|                        |           | CORONET_Global_Topology.xls defines         |
|                        |           | links > 1000km between 2 sites: it          |
|                        |           | couldn't be simulated if these links        |
|                        |           | were not splitted in shorter span           |
|                        |           | lengths.                                    |
+------------------------+-----------+---------------------------------------------+
| `length_unit`          | "m"/"km"  | Unit for max_length.                        |
+------------------------+-----------+---------------------------------------------+
| `max_loss`             | (number)  | Not used in the current code                |
|                        |           | implementation.                             |
+------------------------+-----------+---------------------------------------------+
| `padding`              | (number)  | In dB. Min span loss before putting an      |
|                        |           | attenuator before fiber. Attenuator         |
|                        |           | value                                       |
|                        |           | Fiber.att_in = max(0, padding - span_loss). |
|                        |           | Padding can be set manually to reach a      |
|                        |           | higher padding value for a given fiber      |
|                        |           | by filling in the Fiber/params/att_in       |
|                        |           | field in the topology json input [1]        |
|                        |           | but if span_loss = length * loss_coef       |
|                        |           | + att_in + con_in + con_out < padding,      |
|                        |           | the specified att_in value will be          |
|                        |           | completed to have span_loss = padding.      |
|                        |           | Therefore it is not possible to set         |
|                        |           | span_loss < padding.                        |
+------------------------+-----------+---------------------------------------------+
| `EOL`                  | (number)  | All fiber span loss ageing. The value       |
|                        |           | is added to the con_out (fiber output       |
|                        |           | connector). So the design and the path      |
|                        |           | feasibility are performed with              |
|                        |           | span_loss + EOL. EOL cannot be set          |
|                        |           | manually for a given fiber span             |
|                        |           | (workaround is to specify higher con_out    |
|                        |           | loss for this fiber).                       |
+------------------------+-----------+---------------------------------------------+
| `con_in`, `con_out`    | (number)  | Default values if Fiber/params/con_in/out   |
|                        |           | is None in the topology input               |
|                        |           | description. This default value is          |
|                        |           | ignored if a Fiber/params/con_in/out        |
|                        |           | value is input in the topology for a        |
|                        |           | given Fiber.                                |
+------------------------+-----------+---------------------------------------------+

**[1]**

.. code-block:: json

    {
        "uid": "fiber (A1->A2)",
        "type": "Fiber",
        "type_variety": "SSMF",
        "params":
        {
              "type_variety": "SSMF",
              "length": 120.0,
              "loss_coef": 0.2,
              "length_units": "km",
              "att_in": 0,
              "con_in": 0,
              "con_out": 0
        }
    }

ROADMs can be configured as follows. The user can only modify the value of
existing parameters:

+-------------------------+-----------+---------------------------------------------+
| field                   |   type    | description                                 |
+=========================+===========+=============================================+
|`gain_mode_default_loss` | (number)  | Default value if Roadm/params/loss is       |
|                         |           | None in the topology input description.     |
|                         |           | This default value is ignored if a          |
|                         |           | params/loss value is input in the           |
|                         |           | topology for a given ROADM.                 |
+-------------------------+-----------+---------------------------------------------+
|`power_mode_pref`        | (number)  | Power mode only. Auto-design sets the       |
|                         |           | power of ROADM ingress amplifiers to        |
|                         |           | power_dbm + power_range_db,                 |
|                         |           | regardless of existing gain settings        |
|                         |           | from the topology JSON input.               |
|                         |           | Auto-design sets the Roadm loss so that     |
|                         |           | its egress channel power = power_mode_pref, |
|                         |           | regardless of existing loss settings        |
|                         |           | from the topology JSON input. It means      |
|                         |           | that the output power from a ROADM (and      |
|                         |           | therefore its OSNR contribution) is Cte     |
|                         |           | and not depending from power_dbm and        |
|                         |           | power_range_db sweep settings. This         |
|                         |           | choice is meant to reflect some typical     |
|                         |           | control loop algorithms.                    |
+-------------------------+-----------+---------------------------------------------+

The `SpectralInformation` object can be configured as follows. The user can
only modify the value of existing parameters. It defines a spectrum of N
identical carriers. While the code libraries allow for different carriers and
power levels, the current user parametrization only allows one carrier type and
one power/channel definition.

+----------------------+-----------+-------------------------------------------+
| field                |   type    | description                               |
+======================+===========+===========================================+
| `f_min/max`          | (number)  | In Hz. Carrier min max excursion          |
+----------------------+-----------+-------------------------------------------+
| `baud_rate`          | (number)  | In Hz. Simulated baud rate.               |
+----------------------+-----------+-------------------------------------------+
| `spacing`            | (number)  | In Hz. Carrier spacing.                   |
+----------------------+-----------+-------------------------------------------+
| `roll_off`           | (number)  | Not used.                                 |
+----------------------+-----------+-------------------------------------------+
| `OSNR`               | (number)  | Not used.                                 |
+----------------------+-----------+-------------------------------------------+
| `bit_rate`           | (number)  | Not used.                                 |
+----------------------+-----------+-------------------------------------------+
| `tx_osnr`            | (number)  | In dB. OSNR out from transponder.         |
+----------------------+-----------+-------------------------------------------+
| `power_dbm`          | (number)  | Reference channel power. In gain mode     |
|                      |           | (see spans/power_mode = false), all gain  |
|                      |           | settings are offset w/r/t this reference  |
|                      |           | power. In power mode, it is the           |
|                      |           | reference power for                       |
|                      |           | Spans/delta_power_range_db. For example,  |
|                      |           | if delta_power_range_db = `[0,0,0]`, the  |
|                      |           | same power=power_dbm is launched in every |
|                      |           | spans. The network design is performed    |
|                      |           | with the power_dbm value: even if a       |
|                      |           | power sweep is defined (see after) the    |
|                      |           | design is not repeated.                   |
+----------------------+-----------+-------------------------------------------+
| `power_range_db`     | (number)  | Power sweep excursion around power_dbm.   |
|                      |           | It is not the min and max channel power   |
|                      |           | values! The reference power becomes:      |
|                      |           | power_range_db + power_dbm.               |
+----------------------+-----------+-------------------------------------------+

The `transmission_main_example.py <examples/transmission_main_example.py>`_
script propagates a spectrum of channels at 32 Gbaud, 50 GHz spacing and 0
dBm/channel. These are not yet parametrized but can be modified directly in the
script (via the SpectralInformation structure) to accommodate any baud rate,
spacing, power or channel count demand.

Use `examples/path_requests_run.py <examples/path_requests_run.py>`_ to run multiple optimizations as follows:

.. code-block:: shell

     $ python path_requests_run.py -h
     Usage: path_requests_run.py [-h] [-v] [-o OUTPUT] [network_filename] [service_filename] [eqpt_filename]

The `network_filename` and `service_filename` can be an XLS or JSON file. The `eqpt_filename` must be a JSON file.

To see an example of it, run:

.. code-block:: shell

    $ cd examples
    $ python path_requests_run.py meshTopologyExampleV2.xls meshTopologyExampleV2_services.json eqpt_config.json -o output_file.json

This program requires a list of connections to be estimated and the equipment
library. The program computes performances for the list of services (accepts
json or excel format) using the same spectrum propagation modules as
transmission_main_example.py. Explanation on the Excel template is provided in
the `Excel_userguide.rst <Excel_userguide.rst#service-sheet>`_. Template for
the json format can be found here: `service-template.json
<service-template.json>`_.

Contributing
------------

``gnpy`` is looking for additional contributors, especially those with experience
planning and maintaining large-scale, real-world mesh optical networks.

To get involved, please contact James Powell
<james.powell@telecominfraproject.com> or Gert Grammel <ggrammel@juniper.net>.

``gnpy`` contributions are currently limited to members of `TIP
<http://telecominfraproject.com>`_. Membership is free and open to all.

See the `Onboarding Guide
<https://github.com/Telecominfraproject/gnpy/wiki/Onboarding-Guide>`_ for
specific details on code contributions.

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

.. |docs| image:: https://readthedocs.org/projects/gnpy/badge/?version=develop
  :target: http://gnpy.readthedocs.io/en/develop/?badge=develop
  :alt: Documentation Status
  :scale: 100%

.. |build| image:: https://travis-ci.com/Telecominfraproject/oopt-gnpy.svg?branch=develop
  :target: https://travis-ci.com/Telecominfraproject/oopt-gnpy
  :alt: Build Status
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

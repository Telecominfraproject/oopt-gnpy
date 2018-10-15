====================================================================
`gnpy`: mesh optical network route planning and optimization library
====================================================================

|docs| |build|

**gnpy is an open-source, community-developed library for building route planning
and optimization tools in real-world mesh optical networks.**

`gnpy <http://github.com/telecominfraproject/oopt-gnpy>`__ is:

- a sponsored project of the `OOPT/PSE <https://telecominfraproject.com/open-optical-packet-transport/>`_ working group of the `Telecom Infra Project <http://telecominfraproject.com>`_.
- fully community-driven, fully open source library
- driven by a consortium of operators, vendors, and academic researchers
- intended for rapid development of production-grade route planning tools
- easily extensible to include custom network elements
- performant to the scale of real-world mesh optical networks

Documentation: https://gnpy.readthedocs.io

Installation
------------

``gnpy`` is hosted in the `Python Package Index <http://pypi.org/>`_ (`gnpy
<https://pypi.org/project/gnpy/>`__). It can be installed via:

.. code-block:: shell

    $ pip3 install gnpy

This will install the current (tagged) release version.

This will make the library available for your use. To test, try importing `gnpy` from your Python console

.. code-block:: python

    >>> import gnpy

It can also be installed directly from the repo.

.. code-block:: shell

    $ git clone https://github.com/telecominfraproject/oopt-gnpy
    $ cd oopt-gnpy
    $ python3 setup.py install

Both approaches above will handle installing any additional software dependencies.

It is recommended that you use a virtual environment when installing from the
repo.  In Python 3, you can use the `venv module <https://docs.python.org/3/library/venv.html>`_.

    **Note**: *Alternatively, we recommend the use of the Anaconda Python
    distribution (https://www.anaconda.com/download) which comes with many
    scientific computing dependencies pre-installed.*

Instructions for Use
--------------------

``gnpy`` is a library for building route planning and optimization tools.

It ships with a number of example programs. Release versions will ship with
fully-functional programs.


    **Note**: *If you are a network operator or involved in route planning and
    optimization for your organization, please contact project maintainer James
    Powell <james.powell@telecominfraproject>. gnpy is looking for users with
    specific, delineated use cases to drive requirements for future
    development.*

**To get started, run the transmission example:**

    **Note**: *Examples should be run from the examples/ folder.*

.. code-block:: shell

    $ cd examples
    $ python3 transmission_main_example.py

By default, this script operates on a single span network defined in `examples/edfa_example_network.json <examples/edfa_example_network.json>`_

You may need to set PYTHONPATH variable. For example on Ubuntu, add your workspace path to PYTHONPATH in your .bashrc file:

.. code-block:: shell

    export PYTHONPATH=$PYTHONPATH:~/<workspace path>/oopt-gnpy/

You may need to set PYTHONPATH variable. For example on Ubuntu, add your workspace path to PYTHONPATH in your .bashrc file:

.. code-block:: shell

    export PYTHONPATH=$PYTHONPATH:~/<workspace path>/gnpy/

You can specify a different network at the command line as follows. For
example, to use the CORONET Continental US (CONUS) network defined in `examples/coronet_conus_example.json <examples/coronet_conus_example.json>`_:

.. code-block:: shell

    $ cd examples
    $ python3 transmission_main_example.py CORONET_Global_Topology.json

It is also possible to use an Excel file input (for example CORONET_Global_Topology.xls). The excel file will be parsed automatically into a json file with the same name prefix. How to prepare the Excel input file is explained `here <Excel_userguide.rst>`_.

This script will calculate the average signal osnr and snr across 93 network
elements (transceiver, ROADMs, fibers, and amplifiers) between Abilene, Texas
and Albany, New York.

This script calculates the average signal OSNR = |OSNR| and SNR = |SNR|.

.. |OSNR| replace:: P\ :sub:`ch`\ /P\ :sub:`ase`
.. |SNR| replace:: P\ :sub:`ch`\ /(P\ :sub:`nli`\ +\ P\ :sub:`ase`)

|Pase| is the amplified spontaneous emission noise, and |Pnli| the non-linear
interference noise.

.. |Pase| replace:: P\ :sub:`ase`
.. |Pnli| replace:: P\ :sub:`nli`

Design and transmission parameters are defined in a dedicated json file : examples/eqpt_config.json. This file defines the equipement librairies that can be customized at will:
* Edfa:[]
* Fiber:[]
* Transceiver:[]
It also defines the simulation parameters: 
* Spans:[]
* Roadms:[]
* SI:[]

**EQUIPMENT LIBRARY**
* The Edfa equipment library is a list of supported amplifiers. New amplifiers can be added and existing ones removed at will by the user. It implements 3 different noise models:
1- 'type_def' : 'variable_gain'
  => simplified model simulating a 2 coils edfa with internal, input and output VOAs. The NF vs gain response is calculated accordingly based on the input parameters: nf_min, nf_max and gain_flatmax. It is not a simple interpolation but a 2 stages NF calculation.
2- 'type_def' : 'fixed_gain'
  fixed gain model: NF = Cte = nf0 if gain_min < gain < gain_flatmax
3- 'type_def' : None
  => advanced model: a detailed json configuration file is required 'advanced_config_from_json': 3rd order polynomial NF = f(gain), N-array NF_ripple = f(frequency), N-array gain_ripple = f(frequency), N-array dgt = f(frequency). Compared to the previous models, NF ripple and gain ripple are modelled.
For all amplifier models:
- 'type_variety' : a unique name to id the amplifier in the json or excel template topology input file.
- 'out_voa_auto' : true/false 
  => auto_design feature to optimize the amplifier output VOA. True: output VOA is present and will be used to push amplifier gain to its maximum, within EOL power margins. 
- 'allowed_for_design' : toggle true/false. If False, the amplifier will not be picked by auto-design but it can still be used as a manual input (from json or excel template topology files).

* The Fiber library currently describes SSMF but additional fiber types can be entered by the user, following the same model:
- 'type_variety' : a unique name to id the fiber type in the json or excel template topology input file.
- 'dispersion'  (s.m-1.m-1)
- 'gamma' : 2pi.n2/(lambda*Aeff) (w-2.m-1)

* The Transceiver equipment library is a list of supported transceivers. New transceivers can be added and existing ones removed at will by the user. It is used to determine the service list path feasibility when running the path_request_run.py routine.
- 'type_variety': a unique name to id the transponder in the json or excel template service list input file. 
- 'frequency' : min max excursion
- 'mode' : a list of modes supported by the transponder. New modes can be added at will by the user. The modes are specific to each transponder type_variety. Each mode is described with:
    - 'format' : a unique name to id the mode
    - 'baud_rate' (Hz)
    - 'OSNR' : min required OSNR in 0.1nm (dB)
    - 'bit_rate' (bit/s)
    - 'roll_off'

**SIMULATION PARAMETERS**
* Foreword (about auto_design): 
- auto_design automatically creates Edfa amplifier network elements when they are missing: after a fiber, or between a ROADM and a fiber. This auto_design functionality can be manually and locally deactivated by introducing a Fused network elements after a Fiber or a Roadm that doesn't need amplification. The amplifier is chosen in the Edfa list of the equipment library based on gain, power and NF criteria. Only the Edfa with the toogle 'allowed_for_design' = true are considered.
- For amplifier defined in the topology json input but whose gain = 0 (placeholder), auto_design will set its gain automatically: see power_mode in the Spans library to find out how the gain is calculated.

* Spans configuration library. It is not a list (in the current code version) and the user can only modify the value of existing parameters:
- 'power_mode': true/false
    => false = gain mode: auto_design sets amplifier gain = preceeding span loss, unless the amplifier exists and its gain>0 in the topology input json.
    => true = power mode (recommended for auto-design and power sweep): auto_design sets amplifier power according to delta_power_range (see after). If the amplifier exists with gain>0 in the topology json input, then its gain is translated into a power target/channel. Moreover, when performing a power sweep (see power_range_db in the SI configuration library) the power sweep is performed wrto this power target, regardless of preceeding amplifiers power saturation/limitations.
- 'delta_power_range_db': auto-design only, power mode only, specifies the [min, max, step] power excursion / span. It is a relative power excursion wrto the power_dbm + power_range_db (power sweep if applicable) defined in the SI configuration library. This relative power excursion is = 1/3 of the span loss difference with the reference 20dB span. The 1/3 slope is derived from the GN model equations. For example :
    => a 23dB span loss will be set to 1dB more power than a 20dB span loss. The 20dB reference spans will ALWAYS be set to power = power_dbm + power_range_db. 
    => to configure the same power in all spans : [0,0,0]. All spans will be set to power = power_dbm + power_range_db
    => to configure the same power in all spans and 3dB more power just for the longest spans: [0,3,3]. The longest spans are set to power = power_dbm + power_range_db + 3
    => to configure a 4dB power range across all spans in 0.5dB steps: [-2,2,0.5]. A 17dB span is set to power=power_dbm+power_range_db-1, a 20dB span to power=power_dbm+power_range_db and a 23dB span to power=power_dbm+power_range_db+1
- 'max_length': (length_units) split fiber lengths > max_length. Interest to support high level topologies that do not specify in line amplification sites. For example the CORONET_Global_Topology.xls defines links > 1000km between 2 sites: it couldn't be simulated if these links were not splitted in shorter span lengths.
- 'length_unit': unit for max_length
- 'max_loss' : not used in the current code implementation
- 'padding' (dB) : min span loss before putting an attenuator before fiber. Attenuator value Fiber.att_in = max(0, padding-span_loss). Padding can be set manually to reach a higher padding value for a given fiber by filling in the Fiber/params/att_in field in the topology json input (or excel template):
    =>   {"uid": "fiber (A1->A2)",
          "type": "Fiber",
          "type_variety": "SSMF",
          "params": {
            "type_variety": "SSMF",
            "length": 120.0,
            "loss_coef": 0.2,
            "length_units": "km",
            "att_in": 0,
            "con_in": 0,
            "con_out": 0 } }
    => but if span_loss = length * loss_coef + att_in + con_in + con_out < padding, the specified att_in value will be completed to have span_loss = padding. Therefore it is not possible to set span_loss < padding.
- 'EOL': all fiber span loss ageing. The value is added to the con_out (fiber output connector). So the design and the path feasibility are performed with span_loss + EOL. EOL cannot be set manually for a given fiber span (workaround is to specify higher con_out loss for this fiber).
- 'con_in/out' : default values if Fiber/params/con_in/out is None in the topology input description. This default value is ignored if a Fiber/params/con_in/out value is input in the topology for a given Fiber.

* Roadms configuration library. It is not a list of possible Roadm implementations (in the current code version) and the user can only modify the value of existing parmeters:
- 'gain_mode_default_loss' : default value if Roadm/params/loss is None in the topology input description. This default value is ignored if a params/loss value is input in the topology for a given Roadm.
- 'power_mode_pref' : power mode only,
    => auto_design sets the power of Roadm ingress amplifiers to power_dbm + power_range_db, REGARDLESS OF EXISTING GAIN SETTINGS from the topology json input. 
    => auto_design sets the Roadm loss so that its egress channel power = power_mode_pref, REGARDLESS OF EXISTINIG LOSS SETTINGS from the topology json input. It means that the ouput power from a ROADM (and therefore its OSNR contribution) is Cte and not depending from power_dbm and power_range_db sweep settings. This choice is meant to reflect some typical control loop algorithms.

*SI (Spectrum Information) configuration library: it is not a list and the user can only modify the value of existing parameters. It defines a spectrum of N identical carriers. While the code libraries allow for different carriers and power levels, the current user parametrization only allows one carrier type and one power/channel definition:
- 'f_min/max' (Hz): carrier min max excursion
- 'baud_rate' (Hz): simulated baud rate
- 'spacing' (Hz): carrier spacing
- 'roll_off'
- 'OSNR' : not used
- 'bit_rate' : not used
- 'power_dbm' : reference channel power,
    => In gain mode (see Spans/power_mode = false), all gain settings are offset wrto this reference power. 
    => In power mode, it is the reference power for Spans/delta_power_range_db: for example if delta_power_range_db = [0,0,0], the same power=power_dbm is launched in every spans. 
    => The network design is performed with the power_dbm value: even if a power sweep is defined (see after) the design is not repeated.
- 'power_range_db' : power sweep excursion around power_dbm. It is not the min and max channel power values! The reference power becomes : power_range_db + power_dbm.


The `transmission_main_example.py <examples/transmission_main_example.py>`_
script propagates a specrum of 96 channels at 32 Gbaud, 50 GHz spacing and 0
dBm/channel. These are not yet parametrized but can be modified directly in the
script (via the SpectralInformation tuple) to accomodate any baud rate,
spacing, power or channel count demand.

The amplifier's gain is set to exactly compensate for the loss in each network
element. The amplifier is currently defined with gain range of 15 dB to 25 dB
and 21 dBm max output power. Ripple and NF models are defined in
`examples/std_medium_gain_advanced_config.json <examples/std_medium_gain_advanced_config.json>`_


**Run multiple optimisation with path_requests_run.py**

**Usage**: path_requests_run.py [-h] [-v] [-o OUTPUT]
                            [network_filename xls or json] [service_filename xls or json] [eqpt_filename json]

.. code-block:: shell

    $ cd examples
    $ python path_requests_run.py meshTopologyExampleV2.xls meshTopologyExampleV2_services.json eqpt_file -o output_file.json


Additionally to the json or excel topology input, the program requires a list of connection to be estimated and the equipment library. The program computes performances for the list of services (accepts json or excel format) using the same spectrum propagation modules as transmission_main_example.py. Explanation on the Excel template is provided in the `Excel_userguide.rst <Excel_userguide.rst#service-sheet>`_ ; template for the json format can be found here:  `service_template.json <https://github.com/Telecominfraproject/oopt-gnpy/blob/8f8fc13dedee83532ff5bf83defb5fcb15b46f9f/service-template.json#L1>`_.


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
specific details on code contribtions.

See `AUTHORS.rst <AUTHORS.rst>`_ for past and present contributors.

Project Background
------------------

Data Centers are built upon interchangeable, highly standardized node and
network architectures rather than a sum of isolated solutions. This also
translates to optical networking. It leads to a push in enabling multi-vendor
optical network by disaggregating HW and SW functions and focussing on
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

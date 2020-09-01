JSON Input Files
================

GNPy uses a set of JSON files for modeling the network.
Some data (such as network topology or the service requests) can be also passed via :ref:`XLS files<excel-service-sheet>`.

Equipment Library
-----------------

Design and transmission parameters are defined in a dedicated json file. By
default, this information is read from `gnpy/example-data/eqpt_config.json
<gnpy/example-data/eqpt_config.json>`_. This file defines the equipment libraries that
can be customized (EDFAs, fibers, and transceivers).

It also defines the simulation parameters (spans, ROADMs, and the spectral
information to transmit.)

EDFA
~~~~

The EDFA equipment library is a list of supported amplifiers. New amplifiers
can be added and existing ones removed. Three different noise models are available:

1. ``'type_def': 'variable_gain'`` is a simplified model simulating a 2-coil EDFA with internal, input and output VOAs. The NF vs gain response is calculated accordingly based on the input parameters: ``nf_min``, ``nf_max``, and ``gain_flatmax``. It is not a simple interpolation but a 2-stage NF calculation.
2. ``'type_def': 'fixed_gain'`` is a fixed gain model.  `NF == Cte == nf0` if `gain_min < gain < gain_flatmax`
3. ``'type_def': None`` is an advanced model. A detailed JSON configuration file is required (by default `gnpy/example-data/std_medium_gain_advanced_config.json <gnpy/example-data/std_medium_gain_advanced_config.json>`_). It uses a 3rd order polynomial where NF = f(gain), NF_ripple = f(frequency), gain_ripple = f(frequency), N-array dgt = f(frequency). Compared to the previous models, NF ripple and gain ripple are modelled.

For all amplifier models:

+------------------------+-----------+-----------------------------------------+
| field                  |   type    | description                             |
+========================+===========+=========================================+
| ``type_variety``       | (string)  | a unique name to ID the amplifier in the|
|                        |           | JSON/Excel template topology input file |
+------------------------+-----------+-----------------------------------------+
| ``out_voa_auto``       | (boolean) | auto_design feature to optimize the     |
|                        |           | amplifier output VOA. If true, output   |
|                        |           | VOA is present and will be used to push |
|                        |           | amplifier gain to its maximum, within   |
|                        |           | EOL power margins.                      |
+------------------------+-----------+-----------------------------------------+
| ``allowed_for_design`` | (boolean) | If false, the amplifier will not be     |
|                        |           | picked by auto-design but it can still  |
|                        |           | be used as a manual input (from JSON or |
|                        |           | Excel template topology files.)         |
+------------------------+-----------+-----------------------------------------+

Fiber
~~~~~

The fiber library currently describes SSMF and NZDF but additional fiber types can be entered by the user following the same model:

+----------------------+-----------+-----------------------------------------+
| field                | type      | description                             |
+======================+===========+=========================================+
| ``type_variety``     | (string)  | a unique name to ID the fiber in the    |
|                      |           | JSON or Excel template topology input   |
|                      |           | file                                    |
+----------------------+-----------+-----------------------------------------+
| ``dispersion``       | (number)  | (s.m-1.m-1)                             |
+----------------------+-----------+-----------------------------------------+
| ``dispersion_slope`` | (number)  | (s.m-1.m-1.m-1)                         |
+----------------------+-----------+-----------------------------------------+
| ``gamma``            | (number)  | 2pi.n2/(lambda*Aeff) (w-1.m-1)          |
+----------------------+-----------+-----------------------------------------+
| ``pmd_coef``         | (number)  | Polarization mode dispersion (PMD)      |
|                      |           | coefficient. (s.sqrt(m)-1)              |
+----------------------+-----------+-----------------------------------------+

Transceiver
~~~~~~~~~~~

The transceiver equipment library is a list of supported transceivers. New
transceivers can be added and existing ones removed at will by the user. It is
used to determine the service list path feasibility when running the
``gnpy-path-request`` script.

+----------------------+-----------+-----------------------------------------+
| field                | type      | description                             |
+======================+===========+=========================================+
| ``type_variety``     | (string)  | A unique name to ID the transceiver in  |
|                      |           | the JSON or Excel template topology     |
|                      |           | input file                              |
+----------------------+-----------+-----------------------------------------+
| ``frequency``        | (number)  | Min/max as below.                       |
+----------------------+-----------+-----------------------------------------+
| ``mode``             | (number)  | A list of modes supported by the        |
|                      |           | transponder. New modes can be added at  |
|                      |           | will by the user. The modes are specific|
|                      |           | to each transponder type_variety.       |
|                      |           | Each mode is described as below.        |
+----------------------+-----------+-----------------------------------------+

The modes are defined as follows:

+----------------------+-----------+-----------------------------------------+
| field                | type      | description                             |
+======================+===========+=========================================+
| ``format``           | (string)  | a unique name to ID the mode            |
+----------------------+-----------+-----------------------------------------+
| ``baud_rate``        | (number)  | in Hz                                   |
+----------------------+-----------+-----------------------------------------+
| ``OSNR``             | (number)  | min required OSNR in 0.1nm (dB)         |
+----------------------+-----------+-----------------------------------------+
| ``bit_rate``         | (number)  | in bit/s                                |
+----------------------+-----------+-----------------------------------------+
| ``roll_off``         | (number)  | Pure number between 0 and 1. TX signal  |
|                      |           | roll-off shape. Used by Raman-aware     |
|                      |           | simulation code.                        |
+----------------------+-----------+-----------------------------------------+
| ``tx_osnr``          | (number)  | In dB. OSNR out from transponder.       |
+----------------------+-----------+-----------------------------------------+
| ``cost``             | (number)  | Arbitrary unit                          |
+----------------------+-----------+-----------------------------------------+

ROADM
~~~~~

The user can only modify the value of existing parameters:

+--------------------------+-----------+---------------------------------------------+
| field                    |   type    | description                                 |
+==========================+===========+=============================================+
| ``target_pch_out_db``    | (number)  | Auto-design sets the ROADM egress channel   |
|                          |           | power. This reflects typical control loop   |
|                          |           | algorithms that adjust ROADM losses to      |
|                          |           | equalize channels (eg coming from different |
|                          |           | ingress direction or add ports)             |
|                          |           | This is the default value                   |
|                          |           | Roadm/params/target_pch_out_db if no value  |
|                          |           | is given in the ``Roadm`` element in the    |
|                          |           | topology input description.                 |
|                          |           | This default value is ignored if a          |
|                          |           | params/target_pch_out_db value is input in  |
|                          |           | the topology for a given ROADM.             |
+--------------------------+-----------+---------------------------------------------+
| ``add_drop_osnr``        | (number)  | OSNR contribution from the add/drop ports   |
+--------------------------+-----------+---------------------------------------------+
| ``pmd``                  | (number)  | Polarization mode dispersion (PMD). (s)     |
+--------------------------+-----------+---------------------------------------------+
| ``restrictions``         | (dict of  | If non-empty, keys ``preamp_variety_list``  |
|                          |  strings) | and ``booster_variety_list`` represent      |
|                          |           | list of ``type_variety`` amplifiers which   |
|                          |           | are allowed for auto-design within ROADM's  |
|                          |           | line degrees.                               |
|                          |           |                                             |
|                          |           | If no booster should be placed on a degree, |
|                          |           | insert a ``Fused`` node on the degree       |
|                          |           | output.                                     |
+--------------------------+-----------+---------------------------------------------+

Global parameters
-----------------

The following options are still defined in ``eqpt_config.json`` for legacy reasons, but
they do not correspond to tangible network devices.

Auto-design automatically creates EDFA amplifier network elements when they are
missing, after a fiber, or between a ROADM and a fiber. This auto-design
functionality can be manually and locally deactivated by introducing a ``Fused``
network element after a ``Fiber`` or a ``Roadm`` that doesn't need amplification.
The amplifier is chosen in the EDFA list of the equipment library based on
gain, power, and NF criteria. Only the EDFA that are marked
``'allowed_for_design': true`` are considered.

For amplifiers defined in the topology JSON input but whose ``gain = 0``
(placeholder), auto-design will set its gain automatically: see ``power_mode`` in
the ``Spans`` library to find out how the gain is calculated.

Span
~~~~

Span configuration is not a list (which may change
in later releases) and the user can only modify the value of existing
parameters:

+-------------------------------------+-----------+---------------------------------------------+
| field                               | type      | description                                 |
+=====================================+===========+=============================================+
| ``power_mode``                      | (boolean) | If false, gain mode. Auto-design sets       |
|                                     |           | amplifier gain = preceding span loss,       |
|                                     |           | unless the amplifier exists and its         |
|                                     |           | gain > 0 in the topology input JSON.        |
|                                     |           | If true, power mode (recommended for        |
|                                     |           | auto-design and power sweep.)               |
|                                     |           | Auto-design sets amplifier power            |
|                                     |           | according to delta_power_range. If the      |
|                                     |           | amplifier exists with gain > 0 in the       |
|                                     |           | topology JSON input, then its gain is       |
|                                     |           | translated into a power target/channel.     |
|                                     |           | Moreover, when performing a power sweep     |
|                                     |           | (see ``power_range_db`` in the SI           |
|                                     |           | configuration library) the power sweep      |
|                                     |           | is performed w/r/t this power target,       |
|                                     |           | regardless of preceding amplifiers          |
|                                     |           | power saturation/limitations.               |
+-------------------------------------+-----------+---------------------------------------------+
| ``delta_power_range_db``            | (number)  | Auto-design only, power-mode                |
|                                     |           | only. Specifies the [min, max, step]        |
|                                     |           | power excursion/span. It is a relative      |
|                                     |           | power excursion w/r/t the                   |
|                                     |           | power_dbm + power_range_db                  |
|                                     |           | (power sweep if applicable) defined in      |
|                                     |           | the SI configuration library. This          |
|                                     |           | relative power excursion is = 1/3 of        |
|                                     |           | the span loss difference with the           |
|                                     |           | reference 20 dB span. The 1/3 slope is      |
|                                     |           | derived from the GN model equations.        |
|                                     |           | For example, a 23 dB span loss will be      |
|                                     |           | set to 1 dB more power than a 20 dB         |
|                                     |           | span loss. The 20 dB reference spans        |
|                                     |           | will *always* be set to                     |
|                                     |           | power = power_dbm + power_range_db.         |
|                                     |           | To configure the same power in all          |
|                                     |           | spans, use `[0, 0, 0]`. All spans will      |
|                                     |           | be set to                                   |
|                                     |           | power = power_dbm + power_range_db.         |
|                                     |           | To configure the same power in all spans    |
|                                     |           | and 3 dB more power just for the longest    |
|                                     |           | spans: `[0, 3, 3]`. The longest spans are   |
|                                     |           | set to                                      |
|                                     |           | power = power_dbm + power_range_db + 3.     |
|                                     |           | To configure a 4 dB power range across      |
|                                     |           | all spans in 0.5 dB steps: `[-2, 2, 0.5]`.  |
|                                     |           | A 17 dB span is set to                      |
|                                     |           | power = power_dbm + power_range_db - 1,     |
|                                     |           | a 20 dB span to                             |
|                                     |           | power = power_dbm + power_range_db and      |
|                                     |           | a 23 dB span to                             |
|                                     |           | power = power_dbm + power_range_db + 1      |
+-------------------------------------+-----------+---------------------------------------------+
| ``max_fiber_lineic_loss_for_raman`` | (number)  | Maximum linear fiber loss for Raman         |
|                                     |           | amplification use.                          |
+-------------------------------------+-----------+---------------------------------------------+
| ``max_length``                      | (number)  | Split fiber lengths > max_length.           |
|                                     |           | Interest to support high level              |
|                                     |           | topologies that do not specify in line      |
|                                     |           | amplification sites. For example the        |
|                                     |           | CORONET_Global_Topology.xlsx defines        |
|                                     |           | links > 1000km between 2 sites: it          |
|                                     |           | couldn't be simulated if these links        |
|                                     |           | were not split in shorter span lengths.     |
+-------------------------------------+-----------+---------------------------------------------+
| ``length_unit``                     | "m"/"km"  | Unit for ``max_length``.                    |
+-------------------------------------+-----------+---------------------------------------------+
| ``max_loss``                        | (number)  | Not used in the current code                |
|                                     |           | implementation.                             |
+-------------------------------------+-----------+---------------------------------------------+
| ``padding``                         | (number)  | In dB. Min span loss before putting an      |
|                                     |           | attenuator before fiber. Attenuator         |
|                                     |           | value                                       |
|                                     |           | Fiber.att_in = max(0, padding - span_loss). |
|                                     |           | Padding can be set manually to reach a      |
|                                     |           | higher padding value for a given fiber      |
|                                     |           | by filling in the Fiber/params/att_in       |
|                                     |           | field in the topology json input [1]        |
|                                     |           | but if span_loss = length * loss_coef       |
|                                     |           | + att_in + con_in + con_out < padding,      |
|                                     |           | the specified att_in value will be          |
|                                     |           | completed to have span_loss = padding.      |
|                                     |           | Therefore it is not possible to set         |
|                                     |           | span_loss < padding.                        |
+-------------------------------------+-----------+---------------------------------------------+
| ``EOL``                             | (number)  | All fiber span loss ageing. The value       |
|                                     |           | is added to the con_out (fiber output       |
|                                     |           | connector). So the design and the path      |
|                                     |           | feasibility are performed with              |
|                                     |           | span_loss + EOL. EOL cannot be set          |
|                                     |           | manually for a given fiber span             |
|                                     |           | (workaround is to specify higher            |
|                                     |           | ``con_out`` loss for this fiber).           |
+-------------------------------------+-----------+---------------------------------------------+
| ``con_in``,                         | (number)  | Default values if Fiber/params/con_in/out   |
| ``con_out``                         |           | is None in the topology input               |
|                                     |           | description. This default value is          |
|                                     |           | ignored if a Fiber/params/con_in/out        |
|                                     |           | value is input in the topology for a        |
|                                     |           | given Fiber.                                |
+-------------------------------------+-----------+---------------------------------------------+

.. code-block:: json

    {
        "uid": "fiber (A1->A2)",
        "type": "Fiber",
        "type_variety": "SSMF",
        "params":
        {
              "length": 120.0,
              "loss_coef": 0.2,
              "length_units": "km",
              "att_in": 0,
              "con_in": 0,
              "con_out": 0
        }
    }

SpectralInformation
~~~~~~~~~~~~~~~~~~~

The user can only modify the value of existing parameters. It defines a spectrum of N
identical carriers. While the code libraries allow for different carriers and
power levels, the current user parametrization only allows one carrier type and
one power/channel definition.

+----------------------+-----------+-------------------------------------------+
| field                |   type    | description                               |
+======================+===========+===========================================+
| ``f_min``,           | (number)  | In Hz. Carrier min max excursion.         |
| ``f_max``            |           |                                           |
+----------------------+-----------+-------------------------------------------+
| ``baud_rate``        | (number)  | In Hz. Simulated baud rate.               |
+----------------------+-----------+-------------------------------------------+
| ``spacing``          | (number)  | In Hz. Carrier spacing.                   |
+----------------------+-----------+-------------------------------------------+
| ``roll_off``         | (number)  | Pure number between 0 and 1. TX signal    |
|                      |           | roll-off shape. Used by Raman-aware       |
|                      |           | simulation code.                          |
+----------------------+-----------+-------------------------------------------+
| ``tx_osnr``          | (number)  | In dB. OSNR out from transponder.         |
+----------------------+-----------+-------------------------------------------+
| ``power_dbm``        | (number)  | Reference channel power. In gain mode     |
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
| ``power_range_db``   | (number)  | Power sweep excursion around power_dbm.   |
|                      |           | It is not the min and max channel power   |
|                      |           | values! The reference power becomes:      |
|                      |           | power_range_db + power_dbm.               |
+----------------------+-----------+-------------------------------------------+
| ``sys_margins``      | (number)  | In dB. Added margin on min required       |
|                      |           | transceiver OSNR.                         |
+----------------------+-----------+-------------------------------------------+

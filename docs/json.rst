.. _legacy-json:

JSON Input Files
================

GNPy uses a set of JSON files for modeling the network.
Some data (such as network topology or the service requests) can be also passed via :ref:`XLS files<excel-service-sheet>`.

Equipment Library
-----------------

Design and transmission parameters are defined in a dedicated json file.
By default, this information is read from `gnpy/example-data/eqpt_config.json <https://github.com/Telecominfraproject/oopt-gnpy/blob/master/gnpy/example-data/eqpt_config.json>`_.
This file defines the equipment libraries that can be customized (EDFAs, fibers, and transceivers).

It also defines the simulation parameters (spans, ROADMs, and the spectral information to transmit.)

EDFA
~~~~

The EDFA equipment library is a list of supported amplifiers. New amplifiers
can be added and existing ones removed. Three different noise models are available:

1. ``'type_def': 'variable_gain'`` is a simplified model simulating a 2-coil EDFA with internal, input and output VOAs.
   The NF vs gain response is calculated accordingly based on the input parameters: ``nf_min``, ``nf_max``, and ``gain_flatmax``.
   It is not a simple interpolation but a 2-stage NF calculation.
2. ``'type_def': 'fixed_gain'`` is a fixed gain model.
   `NF == Cte == nf0` if `gain_min < gain < gain_flatmax`
3. ``'type_def': 'openroadm'`` models the incremental OSNR contribution as a function of input power.
   It is suitable for inline amplifiers that conform to the OpenROADM specification.
   The input parameters are coefficients of the :ref:`third-degree polynomial<ext-nf-model-polynomial-OSNR-OpenROADM>`.
4. ``'type_def': 'openroadm_preamp'`` and ``openroadm_booster`` approximate the :ref:`preamp and booster within an OpenROADM network<ext-nf-model-noise-mask-OpenROADM>`.
   No extra parameters specific to the NF model are accepted.
5. ``'type_def': 'advanced_model'`` is an advanced model.
   A detailed JSON configuration file is required (by default `gnpy/example-data/std_medium_gain_advanced_config.json <https://github.com/Telecominfraproject/oopt-gnpy/blob/master/gnpy/example-data/std_medium_gain_advanced_config.json>`_).
   It uses a 3rd order polynomial where NF = f(gain), NF_ripple = f(frequency), gain_ripple = f(frequency), N-array dgt = f(frequency).
   Compared to the previous models, NF ripple and gain ripple are modelled.

For all amplifier models:

+------------------------+-----------+-----------------------------------------+
| field                  |   type    | description                             |
+========================+===========+=========================================+
| ``type_variety``       | (string)  | a unique name to ID the amplifier in the|
|                        |           | JSON/Excel template topology input file |
+------------------------+-----------+-----------------------------------------+
| ``out_voa_auto``       | (boolean) | auto-design feature to optimize the     |
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

+----------------------+-----------+------------------------------------------+
| field                | type      | description                              |
+======================+===========+==========================================+
| ``type_variety``     | (string)  | a unique name to ID the fiber in the     |
|                      |           | JSON or Excel template topology input    |
|                      |           | file                                     |
+----------------------+-----------+------------------------------------------+
| ``dispersion``       | (number)  | In :math:`s \times m^{-1} \times m^{-1}`.|
+----------------------+-----------+------------------------------------------+
| ``dispersion_slope`` | (number)  | In :math:`s \times m^{-1} \times m^{-1}  |
|                      |           | \times m^{-1}`                           |
+----------------------+-----------+------------------------------------------+
| ``effective_area``   | (number)  | Effective area of the fiber (not just    |
|                      |           | the MFD circle). This is the             |
|                      |           | :math:`A_{eff}`, see e.g., the           |
|                      |           | `Corning whitepaper on MFD/EA`_.         |
|                      |           | Specified in :math:`m^{2}`.              |
+----------------------+-----------+------------------------------------------+
| ``gamma``            | (number)  | Coefficient :math:`\gamma = 2\pi\times   |
|                      |           | n^2/(\lambda*A_{eff})`.                  |
|                      |           | If not provided, this will be derived    |
|                      |           | from the ``effective_area``              |
|                      |           | :math:`A_{eff}`.                         |
|                      |           | In :math:`w^{-1} \times m^{-1}`.         |
+----------------------+-----------+------------------------------------------+
| ``pmd_coef``         | (number)  | Polarization mode dispersion (PMD)       |
|                      |           | coefficient. In                          |
|                      |           | :math:`s\times\sqrt{m}^{-1}`.            |
+----------------------+-----------+------------------------------------------+
| ``lumped_losses``    | (array)   | Places along the fiber length with extra |
|                      |           | losses. Specified as a loss in dB at     |
|                      |           | each relevant position (in km):          |
|                      |           | ``{"position": 10, "loss": 1.5}``)       |
+----------------------+-----------+------------------------------------------+

.. _Corning whitepaper on MFD/EA: https://www.corning.com/microsites/coc/oem/documents/specialty-fiber/WP7071-Mode-Field-Diam-and-Eff-Area.pdf

RamanFiber
~~~~~~~~~~

The RamanFiber can be used to simulate Raman amplification through dedicated Raman pumps. The Raman pumps must be listed
in the key ``raman_pumps`` within the RamanFiber ``operational`` dictionary. The description of each Raman pump must
contain the following:

+---------------------------+-----------+------------------------------------------------------------+
| field                     | type      | description                                                |
+===========================+===========+============================================================+
| ``power``                 | (number)  | Total pump power in :math:`W`                              |
|                           |           | considering a depolarized pump                             |
+---------------------------+-----------+------------------------------------------------------------+
| ``frequency``             | (number)  | Pump central frequency in :math:`Hz`                       |
+---------------------------+-----------+------------------------------------------------------------+
| ``propagation_direction`` | (number)  | The pumps can propagate in the same or opposite direction  |
|                           |           | with respect the signal. Valid choices are ``coprop`` and  |
|                           |           | ``counterprop``, respectively                              |
+---------------------------+-----------+------------------------------------------------------------+

Beside the list of Raman pumps, the RamanFiber ``operational`` dictionary must include the ``temperature`` that affects
the amplified spontaneous emission noise generated by the Raman amplification.
As the loss coefficient significantly varies outside the C-band, where the Raman pumps are usually placed,
it is suggested to include an estimation of the loss coefficient for the Raman pump central frequencies within
a dictionary-like definition of the ``RamanFiber.params.loss_coef``
(e.g. ``loss_coef = {"value": [0.18, 0.18, 0.20, 0.20], "frequency": [191e12, 196e12, 200e12, 210e12]}``).

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
| ``frequency``        | (number)  | Min/max central channel frequency.      |
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

+-------------------------------+-----------+----------------------------------------------------+
| field                         |   type    | description                                        |
+===============================+===========+====================================================+
| ``target_pch_out_db``         | (number)  | Default :ref:`equalization strategy<equalization>` |
| or                            |           | for this ROADM type.                               |
| ``target_psd_out_mWperGHz``   |           |                                                    |
| or                            |           | Auto-design sets the ROADM egress channel          |
| ``target_out_mWperSlotWidth`` |           | power. This reflects typical control loop          |
| (mutually exclusive)          |           | algorithms that adjust ROADM losses to             |
|                               |           | equalize channels (e.g., coming from               |
|                               |           | different ingress direction or add ports).         |
|                               |           |                                                    |
|                               |           | These values are used as defaults when no          |
|                               |           | overrides are set per each ``Roadm``               |
|                               |           | element in the network topology.                   |
+-------------------------------+-----------+----------------------------------------------------+
| ``add_drop_osnr``             | (number)  | OSNR contribution from the add/drop ports          |
+-------------------------------+-----------+----------------------------------------------------+
| ``pmd``                       | (number)  | Polarization mode dispersion (PMD). (s)            |
+-------------------------------+-----------+----------------------------------------------------+
| ``restrictions``              | (dict of  | If non-empty, keys ``preamp_variety_list``         |
|                               |  strings) | and ``booster_variety_list`` represent             |
|                               |           | list of ``type_variety`` amplifiers which          |
|                               |           | are allowed for auto-design within ROADM's         |
|                               |           | line degrees.                                      |
|                               |           |                                                    |
|                               |           | If no booster should be placed on a degree,        |
|                               |           | insert a ``Fused`` node on the degree              |
|                               |           | output.                                            |
+-------------------------------+-----------+----------------------------------------------------+

Global parameters
-----------------

The following options are still defined in ``eqpt_config.json`` for legacy reasons, but
they do not correspond to tangible network devices.

Auto-design automatically creates EDFA amplifier network elements when they are missing, after a fiber, or between a ROADM and a fiber.
This auto-design functionality can be manually and locally deactivated by introducing a ``Fused`` network element after a ``Fiber`` or a ``Roadm`` that doesn't need amplification.
The amplifier is chosen in the EDFA list of the equipment library based on gain, power, and NF criteria.
Only the EDFA that are marked ``'allowed_for_design': true`` are considered.

For amplifiers defined in the topology JSON input but whose ``gain = 0`` (placeholder), auto-design will set its gain automatically: see ``power_mode`` in the ``Spans`` library to find out how the gain is calculated.

The file ``sim_params.json`` contains the tuning parameters used within both the ``gnpy.science_utils.RamanSolver`` and
the ``gnpy.science_utils.NliSolver`` for the evaluation of the Raman profile and the NLI generation, respectively.

If amplifiers don't have settings, auto-design also sets amplifiers gain, output VOA and target powers according to [J. -L. Auge, V. Curri and E. Le Rouzic, Open Design for Multi-Vendor Optical Networks, OFC 2019](https://ieeexplore.ieee.org/document/8696699), equation 4.
See ``delta_power_range_db`` for more explaination.

+---------------------------------------------+-----------+---------------------------------------------+
| field                                       |   type    | description                                 |
+=============================================+===========+=============================================+
| ``raman_params.flag``                       | (boolean) | Enable/Disable the Raman effect that        |
|                                             |           | produces a power transfer from higher to    |
|                                             |           | lower frequencies.                          |
|                                             |           | In general, considering the Raman effect    |
|                                             |           | provides more accurate results. It is       |
|                                             |           | mandatory when Raman amplification is       |
|                                             |           | included in the simulation                  |
+---------------------------------------------+-----------+---------------------------------------------+
| ``raman_params.result_spatial_resolution``  | (number)  | Spatial resolution of the output            |
|                                             |           | Raman profile along the entire fiber span.  |
|                                             |           | This affects the accuracy and the           |
|                                             |           | computational time of the NLI               |
|                                             |           | calculation when the GGN method is used:    |
|                                             |           | smaller the spatial resolution higher both  |
|                                             |           | the accuracy and the computational time.    |
|                                             |           | In C-band simulations, with input power per |
|                                             |           | channel around 0 dBm, a suggested value of  |
|                                             |           | spatial resolution is 10e3 m                |
+---------------------------------------------+-----------+---------------------------------------------+
| ``raman_params.solver_spatial_resolution``  | (number)  | Spatial step for the iterative solution     |
|                                             |           | of the first order differential equation    |
|                                             |           | used to calculate the Raman profile         |
|                                             |           | along the entire fiber span.                |
|                                             |           | This affects the accuracy and the           |
|                                             |           | computational time of the evaluated         |
|                                             |           | Raman profile:                              |
|                                             |           | smaller the spatial resolution higher both  |
|                                             |           | the accuracy and the computational time.    |
|                                             |           | In C-band simulations, with input power per |
|                                             |           | channel around 0 dBm, a suggested value of  |
|                                             |           | spatial resolution is 100 m                 |
+---------------------------------------------+-----------+---------------------------------------------+
| ``nli_params.method``                       | (string)  | Model used for the NLI evaluation. Valid    |
|                                             |           | choices are ``gn_model_analytic`` (see      |
|                                             |           | eq. 120 from `arXiv:1209.0394               |
|                                             |           | <https://arxiv.org/abs/1209.0394>`_) and    |
|                                             |           | ``ggn_spectrally_separated`` (see eq. 21    |
|                                             |           | from `arXiv:1710.02225                      |
|                                             |           | <https://arxiv.org/abs/1710.02225>`_).      |
+---------------------------------------------+-----------+---------------------------------------------+
| ``nli_params.computed_channels``            | (number)  | The channels on which the NLI is            |
|                                             |           | explicitly evaluated.                       |
|                                             |           | The NLI of the other channels is            |
|                                             |           | interpolated using ``numpy.interp``.        |
|                                             |           | In a C-band simulation with 96 channels in  |
|                                             |           | a 50 GHz spacing fix-grid we recommend at   |
|                                             |           | one computed channel every 20 channels.     |
+---------------------------------------------+-----------+---------------------------------------------+

Span
~~~~

Span configuration is not a list (which may change in later releases) and the user can only modify the value of existing parameters:

+-------------------------------------+-----------+---------------------------------------------+
| field                               | type      | description                                 |
+=====================================+===========+=============================================+
| ``power_mode``                      | (boolean) | If false, **gain mode**. In the gain mode,  |
|                                     |           | only gain settings are used for             |
|                                     |           | propagation, and ``delta_p`` is ignored.    |
|                                     |           | If no ``gain_target`` is set in an          |
|                                     |           | amplifier, auto-design computes one         |
|                                     |           | according to the ``delta_power_range``      |
|                                     |           | optimisation range.                         |
|                                     |           | The gain mode                               |
|                                     |           | is recommended if all the amplifiers        |
|                                     |           | have already consistent gain settings in    |
|                                     |           | the topology input file.                    |
|                                     |           |                                             |
|                                     |           | If true, **power mode**. In the power mode, |
|                                     |           | only the ``delta_p`` is used for            |
|                                     |           | propagation, and ``gain_target`` is         |
|                                     |           | ignored.                                    |
|                                     |           | The power mode is recommended for           |
|                                     |           | auto-design and power sweep.                |
|                                     |           | If no ``delta_p``  is set,                  |
|                                     |           | auto-design sets an amplifier power target  |
|                                     |           | according to delta_power_range_db.          |
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

GNPy requires a description of all channels that are propagated through the network.
Flexgrid channel partitioning is available since the 2.7 release via the extra ``--spectrum`` option.
In the simplest case, homogeneous channel allocation can be defined via the ``SpectralInformation`` construct which defines a spectrum of N identical carriers:

+----------------------+-----------+-------------------------------------------+
| field                |   type    | description                               |
+======================+===========+===========================================+
| ``f_min``,           | (number)  | In Hz. Define spectrum boundaries. Note   |
| ``f_max``            |           | that due to backward compatibility, the   |
|                      |           | first channel central frequency is placed |
|                      |           | at :math:`f_{min} + spacing` and the last |
|                      |           | one at :math:`f_{max}`.                   |
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
| ``power_dbm``        | (number)  | Reference channel power, in dBm.          |
|                      |           | In gain mode                              |
|                      |           | (see spans/power_mode = false), if no     |
|                      |           | gain is set in an amplifier, auto-design  |
|                      |           | sets gain to meet this reference          |
|                      |           | power. If amplifiers gain is set,         |
|                      |           | ``power_dbm`` is                          |
|                      |           | ignored.                                  |
|                      |           |                                           |
|                      |           | In power mode, the ``power_dbm``          |
|                      |           | is the reference power for                |
|                      |           | the ``delta_p`` settings in amplifiers.   |
|                      |           | It is also the reference power for        |
|                      |           | auto-design power optimisation range      |
|                      |           | Spans/delta_power_range_db. For example,  |
|                      |           | if delta_power_range_db = `[0,0,0]`, the  |
|                      |           | same power=power_dbm is launched in every |
|                      |           | spans. The network design is performed    |
|                      |           | with the power_dbm value: even if a       |
|                      |           | power sweep is defined (see after) the    |
|                      |           | design is not repeated.                   |
|                      |           |                                           |
|                      |           | If the ``--power`` CLI option is used,    |
|                      |           | its value replaces this parameter.        |
+----------------------+-----------+-------------------------------------------+
| ``power_range_db``   | (number)  | Power sweep excursion around              |
|                      |           | ``power_dbm``.                            |
|                      |           | This defines a list of reference powers   |
|                      |           | to run the propagation, in the range      |
|                      |           | power_range_db + power_dbm.               |
|                      |           | Power sweep uses the ``delta_p`` targets  |
|                      |           | or, if they have not been set, the ones   |
|                      |           | computed by auto-design, regardless of    |
|                      |           | of preceding amplifiers' power            |
|                      |           | saturation.                               |
|                      |           |                                           |
|                      |           | Power sweep is an easy way to find the    |
|                      |           | optimal reference power.                  |
|                      |           |                                           |
|                      |           | Power sweep excursion is ignored in case  |
|                      |           | of gain mode.                             |
+----------------------+-----------+-------------------------------------------+
| ``sys_margins``      | (number)  | In dB. Added margin on min required       |
|                      |           | transceiver OSNR.                         |
+----------------------+-----------+-------------------------------------------+

.. _mixed-rate:

Arbitrary channel definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-uniform channels are defined via a list of spectrum "partitions" which are defined in an extra JSON file via the ``--spectrum`` option.
In this approach, each partition is internally homogeneous, but different partitions might use different channel widths, power targets, modulation rates, etc.

+----------------------+-----------+-------------------------------------------+
| field                |   type    | description                               |
+======================+===========+===========================================+
| ``f_min``,           | (number)  | In Hz. Mandatory.                         |
| ``f_max``            |           | Define partition :math:`f_{min}` is       |
|                      |           | the first carrier central frequency       |
|                      |           | :math:`f_{max}` is the last one.          |
|                      |           | :math:`f_{min}` -:math:`f_{max}`          |
|                      |           | partitions must not overlap.              |
|                      |           |                                           |
|                      |           | Note that the meaning of ``f_min`` and    |
|                      |           | ``f_max`` is different than the one in    |
|                      |           | ``SpectralInformation``.                  |
+----------------------+-----------+-------------------------------------------+
| ``baud_rate``        | (number)  | In Hz. Mandatory. Simulated baud rate.    |
+----------------------+-----------+-------------------------------------------+
| ``slot_width``       | (number)  | In Hz. Carrier spectrum occupation.       |
|                      |           | Carriers of this partition are spaced at  |
|                      |           | ``slot_width`` offsets.                   |
+----------------------+-----------+-------------------------------------------+
| ``roll_off``         | (number)  | Pure number between 0 and 1. Mandatory    |
|                      |           | TX signal roll-off shape. Used by         |
|                      |           | Raman-aware simulation code.              |
+----------------------+-----------+-------------------------------------------+
| ``tx_osnr``          | (number)  | In dB. Optional. OSNR out from            |
|                      |           | transponder. Default value is 40 dB.      |
+----------------------+-----------+-------------------------------------------+
| ``delta_pdb``        | (number)  | In dB. Optional. Power offset compared to |
|                      |           | the reference power used for design       |
|                      |           | (SI block in equipment library) to be     |
|                      |           | applied by ROADM to equalize the carriers |
|                      |           | in this partition. Default value is 0 dB. |
+----------------------+-----------+-------------------------------------------+

For example this example:

.. code-block:: json

 {
   "SI":[
     {
       "f_min": 191.4e12,
       "f_max":193.1e12,
       "baud_rate": 32e9,
       "slot_width": 50e9,
       "roll_off": 0.15,
       "tx_osnr": 40
     },
     {
       "f_min": 193.1625e12,
       "f_max":195e12,
       "baud_rate": 64e9,
       "delta_pdb": 3,
       "slot_width": 75e9,
       "roll_off": 0.15,
       "tx_osnr": 40
     }
   ]
 }

...defines a spectrum split into two parts.
Carriers with central frequencies ranging from 191.4 THz to 193.1 THz will have 32 GBaud rate and will be spaced by 50 Ghz.
Carriers with central frequencies ranging from 193.1625 THz to 195 THz will have 64 GBaud rate and will be spaced by 75 GHz with 3 dB power offset.

If the SI reference carrier is set to ``power_dbm`` = 0dBm, and the ROADM has ``target_pch_out_db`` set to -20 dBm, then all channels ranging from 191.4 THz to 193.1 THz will have their power equalized to -20 + 0 dBm (due to the 0 dB power offset).
All channels ranging from 193.1625 THz to 195 THz will have their power equalized to -20 + 3 = -17 dBm (total power signal + noise).

Note that first carrier of the second partition has center frequency 193.1625 THz (its spectrum occupation ranges from 193.125 THz to 193.2 THz).
The last carrier of the second partition has center frequency 193.1 THz and spectrum occupation ranges from 193.075 THz to 193.125 THz.
There is no overlap of the occupation and both share the same boundary.

.. _equalization:

Equalization choices
~~~~~~~~~~~~~~~~~~~~

ROADMs typically equalize the optical power across multiple channels using one of the available equalization strategies — either targeting a specific output power, or a specific power spectral density (PSD), or a spectfic power spectral density using slot_width as spectrum width reference (PSW).
All of these strategies can be adjusted by a per-channel power offset.
The equalization strategy can be defined globally per a ROADM model, or per each ROADM instance in the topology, and within a ROADM also on a per-degree basis.

Let's consider some example for the equalization. Suppose that the types of signal to be propagated are the following:

.. code-block:: json

   {
        "baud_rate": 32e9,
        "f_min":191.3e12,
        "f_max":192.3e12,
        "spacing": 50e9,
        "label": 1
    },
    {
        "baud_rate": 64e9,
        "f_min":193.3e12,
        "f_max":194.3e12,
        "spacing": 75e9,
        "label": 2
    }


with the PSD equalization in a ROADM:

.. code-block:: json

    {
      "uid": "roadm A",
      "type": "Roadm",
      "params": {
        "target_psd_out_mWperGHz": 3.125e-4,
      }
    },


This means that power out of the ROADM will be computed as 3.125e-4 * 32 = 0.01 mW ie -20 dBm for label 1 types of carriers
and 3.125e4 * 64 = 0.02 mW ie -16.99 dBm for label2 channels. So a ratio of ~ 3 dB between target powers for these carriers.

With the PSW equalization:

.. code-block:: json

    {
      "uid": "roadm A",
      "type": "Roadm",
      "params": {
        "target_out_mWperSlotWidth": 2.0e-4,
      }
    },

the power out of the ROADM will be computed as 2.0e-4 * 50 = 0.01 mW ie -20 dBm for label 1 types of carriers
and 2.0e4 * 75 = 0.015 mW ie -18.24 dBm for label2 channels. So a ratio of ~ 1.76 dB between target powers for these carriers.

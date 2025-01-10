.. _legacy-json:

JSON Input Files
================

GNPy uses a set of JSON files for modeling the network.
Some data (such as network topology or the service requests) can be also passed via :ref:`XLS files<excel-service-sheet>`.

Equipment Library
-----------------

Design and transmission parameters are defined in a dedicated json file.
By default, this information is read from `gnpy/example-data/eqpt_config.json <https://github.com/Telecominfraproject/oopt-gnpy/blob/master/gnpy/example-data/eqpt_config.json>`_.
This file defines the equipment libraries that can be customized (Amplifiers, ROADMs, fibers, and transceivers).

It also defines the simulation parameters (spans and the spectral information to transmit.)

Examples of instances are commented here :ref:`json instances examples<json-instance-examples>`.

EDFA
~~~~

The EDFA equipment library is a list of supported amplifiers. New amplifiers
can be added and existing ones removed. Various noise models are available.

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
6. ``'type_def': 'multi_band'`` defines an amplifier type corresponding to an amplification site composed of multiple amplifier elements, where each amplifies a different band of the spectrum.
   The ``amplifiers`` list contains the list of single-band amplifier type varieties that can compose such multiband
   amplifiers. Several options can be listed for the same spectrum band. Only one can be selected
   for the actual :ref:`Multiband_amplifier<multiband_amps>` element.

For all single band amplifier models:

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
| ``f_min``              | (number)  | Optional. In :math:`Hz`. Minimum and    |
| and ``f_max``          |           | maximum frequency range for the         |
|                        |           | amplifier. Signal must fit entirely     |
|                        |           | within this range (center frequency and |
|                        |           | spectrum width).                        |
|                        |           | Default is 191.275e-12 Hz and           |
|                        |           | 196.125e-12.                            |
+------------------------+-----------+-----------------------------------------+

Default values ​​are defined for the frequency range for:
- noise figure ripple
- gain ripple
- dynamic gain tilt

Users can introduce custom values ​​using ``default_config_from_json`` which should be populated with a file name containing the desired parameters.


For multi_band amplifier models:

+------------------------+-----------+-----------------------------------------+
| field                  |   type    | description                             |
+========================+===========+=========================================+
| ``type_variety``       | (string)  | A unique name to ID the amplifier in the|
|                        |           | JSON template topology input file.      |
+------------------------+-----------+-----------------------------------------+
| ``allowed_for_design`` | (boolean) | If false, the amplifier will not be     |
|                        |           | picked by auto-design but it can still  |
|                        |           | be used as a manual input (from JSON or |
|                        |           | Excel template topology files.)         |
+------------------------+-----------+-----------------------------------------+

Fiber
~~~~~

The fiber library currently describes SSMF and NZDF but additional fiber types can be entered by the user following the same model:

+------------------------------+-----------------+------------------------------------------------+
| field                        | type            | description                                    |
+==============================+=================+================================================+
| ``type_variety``             | (string)        | a unique name to ID the fiber in the           |
|                              |                 | JSON or Excel template topology input          |
|                              |                 | file                                           |
+------------------------------+-----------------+------------------------------------------------+
| ``dispersion``               | (number)        | In :math:`s \times m^{-1} \times m^{-1}`.      |
+------------------------------+-----------------+------------------------------------------------+
| ``dispersion_slope``         | (number)        | In :math:`s \times m^{-1} \times m^{-1}        |
|                              |                 | \times m^{-1}`                                 |
+------------------------------+-----------------+------------------------------------------------+
| ``dispersion_per_frequency`` | (dict)          | Dictionary of dispersion values evaluated at   |
|                              |                 | various frequencies, as follows:               |
|                              |                 | ``{"value": [], "frequency": []}``.            |
|                              |                 | ``value`` in                                   |
|                              |                 | :math:`s \times m^{-1} \times m^{-1}` and      |
|                              |                 | ``frequency`` in Hz.                           |
+------------------------------+-----------------+------------------------------------------------+
| ``effective_area``           | (number)        | Effective area of the fiber (not just          |
|                              |                 | the MFD circle). This is the                   |
|                              |                 | :math:`A_{eff}`, see e.g., the                 |
|                              |                 | `Corning whitepaper on MFD/EA`_.               |
|                              |                 | Specified in :math:`m^{2}`.                    |
+------------------------------+-----------------+------------------------------------------------+
| ``gamma``                    | (number)        | Coefficient :math:`\gamma = 2\pi\times         |
|                              |                 | n^2/(\lambda*A_{eff})`.                        |
|                              |                 | If not provided, this will be derived          |
|                              |                 | from the ``effective_area``                    |
|                              |                 | :math:`A_{eff}`.                               |
|                              |                 | In :math:`w^{-1} \times m^{-1}`.               |
|                              |                 | This quantity is evaluated at the              |
|                              |                 | reference frequency and it is scaled           |
|                              |                 | along frequency accordingly to the             |
|                              |                 | effective area scaling.                        |
+------------------------------+-----------------+------------------------------------------------+
| ``pmd_coef``                 | (number)        | Polarization mode dispersion (PMD)             |
|                              |                 | coefficient. In                                |
|                              |                 | :math:`s\times\sqrt{m}^{-1}`.                  |
+------------------------------+-----------------+------------------------------------------------+
| ``lumped_losses``            | (array)         | Places along the fiber length with extra       |
|                              |                 | losses. Specified as a loss in dB at           |
|                              |                 | each relevant position (in km):                |
|                              |                 | ``{"position": 10, "loss": 1.5}``)             |
+------------------------------+-----------------+------------------------------------------------+
| ``raman_coefficient``        | (dict)          | The fundamental parameter that describes       |
|                              |                 | the regulation of the power transfer           |
|                              |                 | between channels during fiber propagation      |
|                              |                 | is the Raman gain coefficient (see             |
|                              |                 | :cite:`DAmicoJLT2022` for further              |
|                              |                 | details); :math:`f_{ref}` represents the       |
|                              |                 | pump reference frequency used for the          |
|                              |                 | Raman gain coefficient profile                 |
|                              |                 | measurement ("reference_frequency"),           |
|                              |                 | :math:`\Delta f` is the frequency shift        |
|                              |                 | between the pump and the specific Stokes       |
|                              |                 | wave, the Raman gain coefficient               |
|                              |                 | in terms of optical power                      |
|                              |                 | :math:`g_0`, expressed in                      |
|                              |                 | :math:`1/(m\;W)`.                              |
|                              |                 | Default values measured for a SSMF are         |
|                              |                 | considered when not specified.                 |
+------------------------------+-----------------+------------------------------------------------+

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

.. _transceiver:

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

+----------------------------+-----------+-----------------------------------------+
| field                      | type      | description                             |
+============================+===========+=========================================+
| ``format``                 | (string)  | a unique name to ID the mode            |
+----------------------------+-----------+-----------------------------------------+
| ``baud_rate``              | (number)  | in Hz                                   |
+----------------------------+-----------+-----------------------------------------+
| ``OSNR``                   | (number)  | min required OSNR in 0.1nm (dB)         |
+----------------------------+-----------+-----------------------------------------+
| ``bit_rate``               | (number)  | in bit/s                                |
+----------------------------+-----------+-----------------------------------------+
| ``roll_off``               | (number)  | Pure number between 0 and 1. TX signal  |
|                            |           | roll-off shape. Used by Raman-aware     |
|                            |           | simulation code.                        |
+----------------------------+-----------+-----------------------------------------+
| ``tx_osnr``                | (number)  | In dB. OSNR out from transponder.       |
+----------------------------+-----------+-----------------------------------------+
| ``equalization_offset_db`` | (number)  | In dB. Deviation from the per channel   |
|                            |           | equalization target in ROADM for this   |
|                            |           | type of transceiver.                    |
+----------------------------+-----------+-----------------------------------------+
| ``penalties``              | (list)    | list of impairments as described in     |
|                            |           | impairment table.                       |
+----------------------------+-----------+-----------------------------------------+
| ``cost``                   | (number)  | Arbitrary unit                          |
+----------------------------+-----------+-----------------------------------------+

Penalties are linearly interpolated between given points and set to 'inf' outside interval.
The accumulated penalties are substracted to the path GSNR before comparing with the min required OSNR.
The penalties per impairment type are defined as a list of dict (impairment type - penalty values) as follows:

+-----------------------------+-----------+-----------------------------------------------+
| field                       | type      | description                                   |
+=============================+===========+===============================================+
| ``chromatic_dispersion`` or | (number)  | In ps/nm/. Value of chromatic dispersion.     |
| ``pdl`` or                  |           | In dB. Value of polarization dependant loss.  |
| ``pmd``                     | (string)  | In ps. Value of polarization mode dispersion. |
+-----------------------------+-----------+-----------------------------------------------+
| ``penalty_value``           | (number)  | in dB. Penalty on the transceiver min OSNR    |
|                             |           | corresponding to the impairment level         |
+-----------------------------+-----------+-----------------------------------------------+

for example:

.. code-block:: json

    "penalties": [{
            "chromatic_dispersion": 360000,
            "penalty_value": 0.5
        }, {
            "pmd": 110,
            "penalty_value": 0.5
        }
    ]

.. _roadm:

ROADM
~~~~~

The user can only modify the value of existing parameters:

+-------------------------------+-----------+----------------------------------------------------+
| field                         |   type    | description                                        |
+===============================+===========+====================================================+
| ``type_variety``              | (string)  | Optional. Default: ``default``                     |
|                               |           | A unique name to ID the ROADM variety in the JSON  |
|                               |           | template topology input file.                      |
+-------------------------------+-----------+----------------------------------------------------+
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
| ``roadm-path-impairments``    | (list of  | Optional. List of ROADM path category impairments. |
|                               | dict)     |                                                    |
+-------------------------------+-----------+----------------------------------------------------+

In addition to these general impairment, the user may define detailed set of impairments for add,
drop and express path within the the ROADM. The impairment description is inspired from the `IETF
CCAMP optical impairment topology <https://github.com/ietf-ccamp-wg/draft-ietf-ccamp-optical-impairment-topology-yang>`_
(details here: `ROADM attributes IETF <https://github.com/ietf-ccamp-wg/draft-ietf-ccamp-optical-impairment-topology-yang/files/4262135/ROADM.attributes_IETF_v8draft.pptx>`_).

The ``roadm-path-impairments`` list allows the definition of the list of impairments by internal path category (add, drop or express). Several additional paths can be defined -- add-path, drop-path or express-path. They are indexed and the related impairments are defined per band.

Each item should contain:

+--------------------------------+-----------+----------------------------------------------------+
| field                          |   type    | description                                        |
+================================+===========+====================================================+
| ``roadm-path-impairments-id``  | (number)  | A unique number to ID the impairments.             |
+--------------------------------+-----------+----------------------------------------------------+
| ``roadm-express-path``         | (list)    | List of the impairments defined per frequency      |
| or                             |           | range. The impairments are detailed in the         |
| ``roadm-add-path``             |           | following table.                                   |
| or                             |           |                                                    |
| ``roadm-drop-path``            |           |                                                    |
| (mutually exclusive)           |           |                                                    |
+--------------------------------+-----------+----------------------------------------------------+

Here are the parameters for each path category and the implementation status:

+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| field                      | Type      | Description                                               | Drop path   | Add path    | Express (thru) path |
+============================+===========+===========================================================+=============+=============+=====================+
| ``frequency-range``        | (list)    | List containing ``lower-frequency`` and                   |             |             |                     |
|                            |           | ``upper-frequency`` in Hz.                                |             |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-maxloss``          | (number)  | In dB. Default: 0 dB. Maximum expected path loss on this  | Implemented | Implemented | Implemented         |
|                            |           | roadm-path assuming no additional path loss is added =    |             |             |                     |
|                            |           | minimum loss applied to channels when crossing the ROADM  |             |             |                     |
|                            |           | (worst case expected loss due to the ROADM).              |             |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-minloss``          |           | The net loss from the ROADM input, to the  output of the  | Not yet     | N.A.        | N.A.                |
|                            |           | drop block (best case expected loss).                     | implemented |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-typloss``          |           | The net loss from the ROADM input, to the output of the   | Not yet     | N.A.        | N.A.                |
|                            |           | drop block (typical).                                     | implemented |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-pmin``             |           | Minimum power levels per carrier expected at the output   | Not yet     | N.A.        | N.A.                |
|                            |           | of the drop block.                                        | implemented |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-pmax``             |           | (Add) Maximum (per carrier) power level permitted at the  | Not yet     | Not yet     | N.A.                |
|                            |           | add block input ports.                                    | implemented | implemented |                     |
|                            |           |                                                           |             |             |                     |
|                            |           | (Drop) Best case per carrier power levels expected at     |             |             |                     |
|                            |           | the output of the drop block.                             |             |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-ptyp``             |           | Typical case per carrier power levels expected at the     | Not yet     | N.A.        | N.A.                |
|                            |           | output of the drop block.                                 | implemented |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-noise-figure``     |           | If the add (drop) path contains an amplifier, this is     | Not yet     | Not yet     | N.A.                |
|                            |           | the noise figure of that amplifier inferred to the        | Implemented | Implemented |                     |
|                            |           | add (drop) port.                                          |             |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-osnr``             | (number)  | (Add) Optical Signal-to-Noise Ratio (OSNR).               | implemented | Implemented | N.A.                |
|                            |           | If the add path contains the ability to adjust the        |             |             |                     |
|                            |           | carrier power levels into an add path amplifier           |             |             |                     |
|                            |           | (if present) to a target value,                           |             |             |                     |
|                            |           | this reflects the OSNR contribution of the                |             |             |                     |
|                            |           | add amplifier assuming this target value is obtained.     |             |             |                     |
|                            |           |                                                           |             |             |                     |
|                            |           | (Drop) Expected OSNR contribution of the drop path        |             |             |                     |
|                            |           | amplifier(if present)                                     |             |             |                     |
|                            |           | for the case of additional drop path loss                 |             |             |                     |
|                            |           | (before this amplifier)                                   |             |             |                     |
|                            |           | in order to hit a target power level (per carrier).       |             |             |                     |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-pmd``              | (number)  | PMD contribution of the specific roadm path.              | Implemented | Implemented | Implemented         |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-cd``               |           |                                                           | Not yet     | Not yet     | Not yet             |
|                            |           |                                                           | Implemented | Implemented | Implemented         |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-pdl``              | (number)  | PDL contribution of the specific roadm path.              | Implemented | Implemented | Implemented         |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+
| ``roadm-inband-crosstalk`` |           |                                                           | Not yet     | Not yet     | Not yet             |
|                            |           |                                                           | Implemented | Implemented | Implemented         |
+----------------------------+-----------+-----------------------------------------------------------+-------------+-------------+---------------------+

Here is a ROADM example with two add-path possible impairments:

.. code-block:: json

    "roadm-path-impairments": [
      {
          "roadm-path-impairments-id": 0,
          "roadm-express-path": [{
              "frequency-range": {
                  "lower-frequency": 191.3e12,
                  "upper-frequency": 196.1e12
                  },
              "roadm-maxloss": 16.5
              }]
      }, {
          "roadm-path-impairments-id": 1,
          "roadm-add-path": [{
              "frequency-range": {
                  "lower-frequency": 191.3e12,
                  "upper-frequency": 196.1e12
              },
              "roadm-maxloss": 11.5,
              "roadm-osnr": 41
          }]
      }, {
          "roadm-path-impairments-id": 2,
          "roadm-drop-path": [{
              "frequency-range": {
                  "lower-frequency": 191.3e12,
                  "upper-frequency": 196.1e12
                  },
              "roadm-pmd": 0,
              "roadm-cd": 0,
              "roadm-pdl": 0,
              "roadm-maxloss": 11.5,
              "roadm-osnr": 41
          }]
      }, {
          "roadm-path-impairments-id": 3,
          "roadm-add-path": [{
              "frequency-range": {
                  "lower-frequency": 191.3e12,
                  "upper-frequency": 196.1e12
              },
              "roadm-pmd": 0,
              "roadm-cd": 0,
              "roadm-pdl": 0,
              "roadm-maxloss": 11.5,
              "roadm-osnr": 20
          }]
      }]

On this example, the express channel has at least 16.5 dB loss when crossing the ROADM express path with the corresponding impairment id.

roadm-path-impairments is optional. If present, its values are considered instead of the ROADM general parameters.
For example, if add-path specifies 0.5 dB PDL and the general PDL parameter states 1.0 dB, then 0.5 dB is applied for this roadm-path only.
If present in add and/or drop path, roadm-osnr replaces the portion of add-drop-osnr defined for the whole ROADM,
assuming that add and drop contribution aggregated in add-drop-osnr are identical:

.. math::

  add\_drop\_osnr = - 10log10(1/add_{osnr} + 1/drop_{osnr})

when:

.. math::

  add_{osnr} = drop_{osnr}

.. math::

  add_{osnr} = drop_{osnr} = add\_drop\_osnr + 10log10(2)


The user can specify the roadm type_variety in the json topology ROADM instance. If no variety is defined, ``default`` ID is used.
The user can define the impairment type for each roadm-path using the degrees ingress/egress immediate neighbor elements and the roadm-path-impairment-id defined in the library for the corresponding type-variety.
Here is an example:

.. code-block:: json

    {
      "uid": "roadm SITE1",
      "type": "Roadm",
      "type_variety": "detailed_impairments",
      "params": {
        "per_degree_impairments": [
        {
          "from_degree": "trx SITE1",
          "to_degree": "east edfa in SITE1 to ILA1",
          "impairment_id": 1
        }]
      }
    }

It is not permitted to use a roadm-path-impairment-id for the wrong roadm path type (add impairment only for add path).
If nothing is stated for impairments on roadm-paths, the program identifies the paths implicitly and assigns the first impairment_id that matches the type: if a transceiver is present on one degree, then it is an add/drop degree.

On the previous example, all «implicit» express roadm-path are assigned roadm-path-impairment-id = 0

.. _sim-params:

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
| ``raman_params.method``                     | (string)  | Model used for Raman evaluation. Valid      |
|                                             |           | choices are ``perturbative`` (see           |
|                                             |           | `arXiv:2304.11756                           |
|                                             |           | <https://arxiv.org/abs/2304.11756>`_) and   |
|                                             |           | ``numerical``, the GNPy legacy first order  |
|                                             |           | derivative numerical solution.              |
+---------------------------------------------+-----------+---------------------------------------------+
|``raman_params.order``                       |           | Order of the perturbative expansion.        |
|                                             |           | For C- and C+L-band transmission scenarios  |
|                                             |           | the second order provides high accuracy     |
|                                             |           | considering common values of fiber input    |
|                                             |           | power. (Default is 2)                       |
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
| ``raman_params.solver_spatial_resolution``  | (number)  | When using the ``perturbative`` method,     |
|                                             |           | the step for the spatial integration does   |
|                                             |           | not affect the first order. Therefore, a    |
|                                             |           | large step can be used when no              |
|                                             |           | counter-propagating Raman amplification is  |
|                                             |           | present; a suggested value is 10e3 m.       |
|                                             |           | In presence of counter-propagating Raman    |
|                                             |           | amplification or when using the             |
|                                             |           | ``numerical`` method the following remains  |
|                                             |           | valid.                                      |
|                                             |           | The spatial step for the iterative solution |
|                                             |           | affects the accuracy and the                |
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
|                                             |           | ``ggn_approx`` (see eq. 24-25               |
|                                             |           | from `jlt:9741324                           |
|                                             |           | <https://eeexplore.ieee.org/document/       |
|                                             |           | 9741324>`_).                                |
+---------------------------------------------+-----------+---------------------------------------------+
| ``dispersion_tolerance``                    | (number)  | Optional. Pure number. Tuning parameter for |
|                                             |           | ggn model solution. Default value is 1.     |
+---------------------------------------------+-----------+---------------------------------------------+
| ``phase_shift_tolerance``                   | (number)  | Optional. Pure number. Tuning parameter for |
|                                             |           | ggn model solution. Defaut value is 0.1.    |
+---------------------------------------------+-----------+---------------------------------------------+
| ``nli_params.computed_channels``            | (list     | Optional. The exact channel indices         |
|                                             | of        | (starting from 1) on which the NLI is       |
|                                             | numbers)  | explicitly evaluated.                       |
|                                             |           | The NLI of the other channels is            |
|                                             |           | interpolated using ``numpy.interp``.        |
|                                             |           | In a C-band simulation with 96 channels in  |
|                                             |           | a 50 GHz spacing fix-grid we recommend at   |
|                                             |           | least one computed channel every 20         |
|                                             |           | channels. If this option is present, the    |
|                                             |           | next option "computed_number_of_channels"   |
|                                             |           | is ignored. If none of the options are      |
|                                             |           | present, the NLI is computed for all        |
|                                             |           | channels (no interpolation)                 |
+---------------------------------------------+-----------+---------------------------------------------+
| ``nli_params.computed_number_of_channels``  | (number)  | Optional. The number of channels on which   |
|                                             |           | the NLI is explicitly evaluated.            |
|                                             |           | The channels are                            |
|                                             |           | evenly selected between the first and the   |
|                                             |           | last carrier of the current propagated      |
|                                             |           | spectrum.                                   |
|                                             |           | The NLI of the other channels is            |
|                                             |           | interpolated using ``numpy.interp``.        |
|                                             |           | In a C-band simulation with 96 channels in  |
|                                             |           | a 50 GHz spacing fix-grid we recommend at   |
|                                             |           | least 6 channels.                           |
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

Power sweep functionality is triggered when setting "power_range_db" in SI in the library. This defines a
list of reference powers on which a new design is performed and propagation is triggered
(only gnpy-transmission-example script).

for example, with the following settings:

  - ``power_dbm`` = 0 dBm
  - max power of the amplifier = 20 dBm,
  - user defined ``delta_p`` set by user = 3 dB
  - 80 channels, so :math:`pch_{max}` = 20 - 10log10(80) = 0.96 dBm
  - ``delta_power_range_db`` = [-3, 0, 3]
  - power_sweep -> power range [-3, 0] dBm

then the computation of delta_p during design for each power of this power sweep is:

  - with :math:`p_{ref}` = 0 dBm, computed_delta_p = min(:math:`pch_{max}`, :math:`p_{ref}` + ``delta_p``) - :math:`p_{ref}` = 0.96 ;
    - user defined ``delta_p`` = 3 dB **can not** be applied because of saturation,
  - with :math:`p_{ref}` = -3 dBm (power sweep) computed_delta_p = min(:math:`pch_{max}`, :math:`p_{ref}` + ``delta_p``) - :math:`p_{ref}` =
    min(0.96, -3.0 + 3.0) - (-3.0) = 3.0 ;
    - user defined ``delta_p`` = 3 dB **can** be applied.

so the user defined delta_p is applied as much as possible.

.. _spectral_info:

SpectralInformation
~~~~~~~~~~~~~~~~~~~

GNPy requires a description of all channels that are propagated through the network.

This block defines a reference channel (target input power in spans, nb of channels) which is used to design the network or correct the settings.
It may be updated with different options --power.
It also defines the channels to be propagated for the gnpy-transmission-example script unless a different definition is provided with ``--spectrum`` option.

Flexgrid channel partitioning is available since the 2.7 release via the extra ``--spectrum`` option.
In the simplest case, homogeneous channel allocation can be defined via the ``SpectralInformation`` construct which defines a spectrum of N identical carriers:

+----------------------+-----------+-------------------------------------------+
| field                |   type    | description                               |
+======================+===========+===========================================+
| ``type_variety``     | (string)  | Optional. Default: ``default``            |
|                      |           | A unique name to ID the band for          |
|                      |           | propagation or design.                    |
+----------------------+-----------+-------------------------------------------+
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
| ``power_dbm``        | (number)  | In dBm. Target input power in spans to    |
|                      |           | be considered for the design              |
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
| ``tx_power_dbm``     | (number)  | In dBm. Optional. Power out from          |
|                      |           | transceiver. Default = power_dbm          |
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

It is possible to define a set of bands in the SI block. In this case, type_variety must be used.
Each set defines a reference channel used for design functions and autodesign processes.

If no spectrum is defined (--spectrum or --services), then the same type of reference channel is
also used for simulation.


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
| ``tx_power_dbm``     | (number)  | In dBm. Optional. Power out from          |
|                      |           | transceiver. Default value is 0 dBm       |
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
   "spectrum":[
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
       "f_max": 195e12,
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


.. _topology:

Topology
--------

Topology file contains a list of elements and a list of connections between the elements to form a graph.

Elements can be:

- Fiber
- RamanFiber
- Edfa
- Fused
- Roadm
- Transceiver


Common attributes
~~~~~~~~~~~~~~~~~

All elements contain the followind attributes:

- **"uid"**: mandatory, element unique identifier.
- **"type"**: mandatory, element type among possible types (Fiber, RamanFiber, Edfa, Fused, Roadm, Transceiver).
- **"metadata"**: optional data including goelocation.


Fiber attributes/ RamanFiber attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+-----------+--------------------------------------------------+
| field                |   type    | description                                      |
+======================+===========+==================================================+
| ``type_variety``     | (string)  | optional, value must be listed in the            |
|                      |           | library to be a valid type. Default type         |
|                      |           | is SSMF.                                         |
+----------------------+-----------+--------------------------------------------------+
| ``params``           | (dict of  | see table below.                                 |
|                      | numbers)  |                                                  |
+----------------------+-----------+--------------------------------------------------+


+----------------------+-----------+--------------------------------------------------+
| params fields        |   type    | description                                      |
+======================+===========+==================================================+
| ``length``           | (number)  | optional, length in ``length_units``, default    |
|                      |           | length is 80 km.                                 |
+----------------------+-----------+--------------------------------------------------+
| ``length_units``     | (string)  | Length unit of measurement. Default is "km".     |
+----------------------+-----------+--------------------------------------------------+
| ``loss_coef``        | (number   | In dB/km. Optional, loss coefficient. Default    |
|                      | or dict)  | is 0.2 dB/km. Slope of the loss can be defined   |
|                      |           | using a dict of frequency values such as         |
|                      |           | ``{"value": [0.18, 0.18, 0.20, 0.20],            |
|                      |           | "frequency": [191e12, 196e12, 200e12, 210e12]}`` |
+----------------------+-----------+--------------------------------------------------+
| ``att_in``           | (number)  | In dB. Optional, attenuation at fiber input, for |
|                      |           | padding purpose. Default is 0 dB.                |
+----------------------+-----------+--------------------------------------------------+
| ``con_in``           | (number)  | In dB. Optional, input connector loss. Default   |
|                      |           | is using value defined in library ``Span``       |
|                      |           | section.                                         |
+----------------------+-----------+--------------------------------------------------+
| ``con_out``          | (number)  | In dB. Optional, output connector loss. Default  |
|                      |           | is using value defined in library ``Span``       |
|                      |           | section.                                         |
+----------------------+-----------+--------------------------------------------------+

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

The RamanFiber can be used to simulate Raman amplification through dedicated Raman pumps. The Raman pumps must be listed
in the key ``raman_pumps`` within the RamanFiber ``operational`` dictionary. The description of each Raman pump must
contain the following:

+---------------------------+-----------+------------------------------------------------------------+
| operational fields        | type      | description                                                |
+===========================+===========+============================================================+
| ``power``                 | (number)  | Total pump power in :math:`W`                              |
|                           |           | considering a depolarized pump                             |
+---------------------------+-----------+------------------------------------------------------------+
| ``frequency``             | (number)  | Pump central frequency in :math:`Hz`                       |
+---------------------------+-----------+------------------------------------------------------------+
| ``propagation_direction`` | (string)  | The pumps can propagate in the same or opposite direction  |
|                           |           | with respect the signal. Valid choices are ``coprop`` and  |
|                           |           | ``counterprop``, respectively                              |
+---------------------------+-----------+------------------------------------------------------------+

Beside the list of Raman pumps, the RamanFiber ``operational`` dictionary must include the ``temperature`` that affects
the amplified spontaneous emission noise generated by the Raman amplification.
As the loss coefficient significantly varies outside the C-band, where the Raman pumps are usually placed,
it is suggested to include an estimation of the loss coefficient for the Raman pump central frequencies within
a dictionary-like definition of the ``RamanFiber.params.loss_coef``
(e.g. ``loss_coef = {"value": [0.18, 0.18, 0.20, 0.20], "frequency": [191e12, 196e12, 200e12, 210e12]}``).

.. code-block:: json

    {
      "uid": "Span1",
      "type": "RamanFiber",
      "type_variety": "SSMF",
      "operational": {
        "temperature": 283,
        "raman_pumps": [
          {
            "power": 224.403e-3,
            "frequency": 205e12,
            "propagation_direction": "counterprop"
          },
          {
            "power": 231.135e-3,
            "frequency": 201e12,
            "propagation_direction": "counterprop"
          }
        ]
      },
      "params": {
        "type_variety": "SSMF",
        "length": 80.0,
        "loss_coef": {
          "value": [0.18, 0.18, 0.20, 0.20],
          "frequency": [191e12, 196e12, 200e12, 210e12]
        },
        "length_units": "km",
        "att_in": 0,
        "con_in": 0.5,
        "con_out": 0.5
      },
      "metadata": {
        "location": {
          "latitude": 1,
          "longitude": 0,
          "city": null,
          "region": ""
        }
      }
    }

Edfa attributes
~~~~~~~~~~~~~~~

The user can specify the amplifier configurations, which are applied depending on general simulation setup:
- if the user has specified ``power_mode`` as True in Span section, delta_p is applied and gain_target is ignored and recomputed.
- if the user has specified ``power_mode`` as False in Span section, gain_target is applied and delta_p is ignored.
If the user has specified unfeasible targets with respect to the type_variety, targets might be changed accordingly.
For example, if gain_target leads to a power value above the maximum output power of the amplifier, the gain is saturated to
the maximum achievable total power.

The exact layout used by simulation can be retrieved thanks to --save-network option.

.. _operational_field:

+----------------------+-----------+--------------------------------------------------+
| field                |   type    | description                                      |
+======================+===========+==================================================+
| ``type_variety``     | (string)  | Optional, value must be listed in the library    |
|                      |           | to be a valid type. If not defined, autodesign   |
|                      |           | will pick one in the library among the           |
|                      |           | ``allowed_for_design``. Autodesign selection is  |
|                      |           | based J. -L. Auge, V. Curri and E. Le Rouzic,    |
|                      |           | Open Design for Multi-Vendor Optical Networks    |
|                      |           | , OFC 2019. equation 4                           |
+----------------------+-----------+--------------------------------------------------+
| ``operational``      | (dict of  | Optional, configuration settings of the          |
|                      | numbers)  | amplifier. See table below                       |
+----------------------+-----------+--------------------------------------------------+

+----------------------+-----------+-------------------------------------------------------------+
| operational field    |   type    | description                                                 |
+======================+===========+=============================================================+
| ``gain_target``      | (number)  | In dB. Optional Gain target between in_voa and out_voa.     |
|                      |           |                                                             |
+----------------------+-----------+-------------------------------------------------------------+
| ``delta_p``          | (number)  | In dB. Optional Power offset at the outpout of the          |
|                      |           | amplifier and before out_voa compared to reference channel  |
|                      |           | power defined in SI block of library.                       |
+----------------------+-----------+-------------------------------------------------------------+
| ``out_voa``          | (number)  | In dB. Optional, output variable optical attenuator loss.   |
+----------------------+-----------+-------------------------------------------------------------+
| ``in_voa``           | (number)  | In dB. Optional, input variable optical attenuator loss.    |
+----------------------+-----------+-------------------------------------------------------------+
| ``tilt_target``      | (number)  | In dB. Optional, tilt target on the whole wavelength range  |
|                      |           | of the amplifier.                                           |
+----------------------+-----------+-------------------------------------------------------------+

.. code-block:: json

    {
      "uid": "Edfa1",
      "type": "Edfa",
      "type_variety": "std_low_gain",
      "operational": {
        "gain_target": 15.0,
        "delta_p": -2,
        "tilt_target": -1,
        "out_voa": 0
      },
      "metadata": {
        "location": {
          "latitude": 2,
          "longitude": 0,
          "city": null,
          "region": ""
        }
      }
    }

.. _multiband_amps:

Multiband_amplifier attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+-----------+--------------------------------------------------+
| field                |   type    | description                                      |
+======================+===========+==================================================+
| ``type``             | (string)  | Mandatory: ``Multiband_amplifier``               |
+----------------------+-----------+--------------------------------------------------+
| ``type_variety``     | (string)  | Optional, value must be listed in the library    |
|                      |           | to be a valid type. If not defined, autodesign   |
|                      |           | will pick one in the library among the           |
|                      |           | ``allowed_for_design``.                          |
+----------------------+-----------+--------------------------------------------------+
| ``amplifiers``       | (list of  | Optional, configuration settings of the          |
|                      |  dict)    | amplifiers composing the multiband amplifier.    |
|                      |           | Single band amplifier can be set with the        |
|                      |           | parameters of tables:                            |
|                      |           | :ref:`operational_field<operational_field>`:     |
+----------------------+-----------+--------------------------------------------------+

Example of Multiband_amplifier element setting:

    .. code-block:: json

      {
          "uid": "east edfa in Site_A to Site_B",
          "type": "Multiband_amplifier",
          "type_variety": "std_medium_gain_multiband",
          "amplifiers": [{
                  "type_variety": "std_medium_gain_C",
                  "operational": {
                      "gain_target": 22.55,
                      "delta_p": 0.9,
                      "out_voa": 3.0,
                      "tilt_target": 0.0
                  }
              }, {
                  "type_variety": "std_medium_gain_L",
                  "operational": {
                      "gain_target": 21,
                      "delta_p": 3.0,
                      "out_voa": 3.0,
                      "tilt_target": 0.0
                  }
              }
          ]
      }

The frequency band of the element is the concatenation of the bands of each individual amplifier contained in
the Multiband_amplifier element. Only carriers within these bands are propagated through the Multiband_amplifier
element. If the user defines a spectrum larger than these bands, carriers that do not match the bands will be
filtered out. The user can define the bandwidth of the amplifiers in the library. f_min and f_max represent the
bandwidth of the amplifier (the entire channel must fit). The individual amplifier type_variety must be part of the
allowed ``amplifiers`` list defined in the library.

Roadm
~~~~~

.. _roadm_json_instance:

+----------------------------------------+-----------+----------------------------------------------------+
| field                                  |   type    | description                                        |
+========================================+===========+====================================================+
| ``type_variety``                       | (string)  | Optional. If no variety is defined, ``default``    |
|                                        |           | ID is used.                                        |
|                                        |           | A unique name must be used to ID the ROADM         |
|                                        |           | variety in the JSON library file.                  |
+----------------------------------------+-----------+----------------------------------------------------+
| ``target_pch_out_db``                  | (number)  | :ref:`Equalization strategy<equalization>`         |
| or                                     |           | for this ROADM. Optional: if not defined, the      |
| ``target_psd_out_mWperGHz``            |           | one defined in library for this type_variety is    |
| or                                     |           | used.                                              |
| ``target_out_mWperSlotWidth``          |           |                                                    |
| (mutually exclusive)                   |           |                                                    |
+----------------------------------------+-----------+----------------------------------------------------+
| ``restrictions``                       | (dict of  | Optional. If defined, it overrides restriction     |
|                                        |  strings) | defined in library for this roadm type_variety.    |
+----------------------------------------+-----------+----------------------------------------------------+
| ``per_degree_pch_out_db``              | (dict of  | Optional. If defined, it overrides ROADM's general |
| or                                     |  string,  | target power/psd for this degree. Dictionary with  |
| ``per_degree_psd_out_mWperGHz``        |  number)  | key = degree name (uid of the immediate adjacent   |
| or                                     |           | element) and value = target power/psd value.       |
| ``per_degree_psd_out_mWperSlotWidth``  |           |                                                    |
+----------------------------------------+-----------+----------------------------------------------------+
| ``per_degree_impairments``             | (list of  | Optional. Impairments id for roadm-path. If        |
|                                        |  dict)    | defined, it overrides the general values defined   |
|                                        |           | by type_variety.                                   |
+----------------------------------------+-----------+----------------------------------------------------+
| ``design_bands``                       | (list of  | Optional. List of bands expressed as dictionnary,  |
|                                        |  dict)    | e.g. {"f_min": 191.3e12, "f_max": 195.1e12}        |
|                                        |           | To be considered for autodesign on all degrees of  |
|                                        |           | the ROADM, if nothing is defined on the degrees.   |
+----------------------------------------+-----------+----------------------------------------------------+
| ``per_degree_design_bands``            | (dict of  | Optional. If defined, it overrides ROADM's general |
|                                        |  string,  | design_bands, on the degree identified with the    |
|                                        |  list of  | key string. Value is a list of bands defined by    |
|                                        |  dict)    | their frequency bounds ``f_min`` and ``f_max``     |
|                                        |           | expressed in THz.                                  |
+----------------------------------------+-----------+----------------------------------------------------+


Definition example:

  .. code-block:: json

    {
      "uid": "roadm SITE1",
      "type": "Roadm",
      "type_variety": "detailed_impairments",
      "params": {
        "per_degree_impairments": [
          {
            "from_degree": "trx SITE1",
            "to_degree": "east edfa in SITE1 to ILA1",
            "impairment_id": 1
          }],
        "per_degree_pch_out_db": {
            "east edfa in SITE1 to ILA1": -13.5
        }
      }
    }

In this example, all «implicit» express roadm-path are assigned as roadm-path-impairment-id = 0, and the target power is
set according to the value defined in the library except for the direction heading to "east edfa in SITE1 to ILA1", where
constant power equalization is used to reach -13.5 dBm target power.

  .. code-block:: json

    {
      "uid": "roadm SITE1",
      "type": "Roadm",
      "params": {
        "per_degree_design_bands": {
          "east edfa in SITE1 to ILA1": [
            {"f_min": 191.3e12, "f_max": 196.0e12},
            {"f_min": 187.0e12, "f_max": 190.0e12}
          ]
        }
      }
    }

In this example the OMS starting from east edfa in SITE1 to ILA1 is defined as a multiband OMS. This means that
if there is no setting in all or some of the amplifiers in the OMS, the autodesign function will select amplifiers
from those that have ``multi_band`` ``type_def`` amplifiers.

The default ``design_bands`` is inferred from the :ref:`SI<spectral_info>` block.

Note that ``design_bands`` and ``type_variety`` amplifiers must be consistent:
- you cannot mix single band and multiband amplifiers on the same OMS;
- the frequency range of the amplifiers must include ``design_bands``.

Fused
~~~~~

The user can define concentrated losses thanks to Fused element. This can be useful for example to materialize connector with its loss between two fiber spans.
``params`` and ``loss`` are optional, loss of the concentrated loss is in dB. Default value is 0 dB.
A fused element connected to the egress of a ROADM will disable the automatic booster/preamp selection.

Fused ``params`` only contains a ``loss`` value in dB.

  .. code-block:: json

      "params": {
        "loss": 2
      }


Transceiver
~~~~~~~~~~~

Transceiver elements represent the logical function that generates a spectrum. This must be specified to start and stop propagation. However, the characteristics of the spectrum are defined elsewhere, so Transceiver elements do not contain any attribute.
Information on transceivers' type, modes and frequency must be listed in :ref:`service file<service>` or :ref:`spectrum file<mixed-rate>`. Without any definition, default :ref:`SI<spectral_info>` values of the library are propagated.

.. _service:

Service JSON file
-----------------

Service file lists all requests and their possible constraints. This is derived from draft-ietf-teas-yang-path-computation-01.txt:
gnpy-path-request computes performance of each request independantly from each other, considering full load (based on the request settings),
but computes spectrum occupation based on the list of request, so that the requests should not define overlapping spectrum.
Lack of spectrum leads to blocking, but performance estimation is still returned for information.


+-----------------------+-------------------+----------------------------------------------------------------+
| field                 |   type            | description                                                    |
+=======================+===================+================================================================+
| ``path-request``      | (list of          | list of requests.                                              |
|                       |  request)         |                                                                |
+-----------------------+-------------------+----------------------------------------------------------------+
| ``synchronization``   | (list of          | Optional. List of synchronization vector. One synchronization  |
|                       | synchronization)  | vector contains the disjunction constraints.                   |
+-----------------------+-------------------+----------------------------------------------------------------+

- **"path-request"** list of requests made of:

+-----------------------+------------+----------------------------------------------------------------+
| field                 |   type     | description                                                    |
+=======================+============+================================================================+
| ``request-id``        | (number)   | Mandatory. Unique id of request. The same id is referenced in  |
|                       |            | response with ``response-id``.                                 |
+-----------------------+------------+----------------------------------------------------------------+
| ``source``            | (string)   | Mandatory. Source of traffic. It must be one of the UID of     |
|                       |            | transceivers listed in the topology.                           |
+-----------------------+------------+----------------------------------------------------------------+
| ``src-tp-id``         | (string)   | Mandatory. It must be equal to ``source``.                     |
+-----------------------+------------+----------------------------------------------------------------+
| ``destination``       | (string)   | Mandatory. Destination of traffic. It must be one of the UID   |
|                       |            | of transceivers listed in the topology.                        |
+-----------------------+------------+----------------------------------------------------------------+
| ``dst-tp-id``         | (string)   | Mandatory. It must be equal to ``destination``.                |
+-----------------------+------------+----------------------------------------------------------------+
| ``bidirectional``     | (boolean)  | Mandatory. Boolean indicating if the propagation should be     |
|                       |            | checked on source-destination only (false) or on               |
|                       |            | destination-source (true).                                     |
+-----------------------+------------+----------------------------------------------------------------+
| ``path-constraints``  | (dict)     | Mandatory. It contains the list of constraints including type  |
|                       |            | of transceiver, mode and nodes to be included in the path.     |
+-----------------------+------------+----------------------------------------------------------------+

``path-constraints`` contains ``te-bandwidth`` with the following attributes:

+-----------------------------------+------------+----------------------------------------------------------------+
| field                             |   type     | description                                                    |
+===================================+============+================================================================+
| ``technology``                    | (string)   | Mandatory. Only one possible value ``flex-grid``.              |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``trx_type``                      | (string)   | Mandatory. Type of the transceiver selected for this request.  |
|                                   |            | It must be listed in the library transceivers list.            |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``trx_mode``                      | (string)   | Optional. Mode selected for this path. It must be listed       |
|                                   |            | within the library transceiver's modes. If not defined,        |
|                                   |            | the gnpy-path-request script automatically selects the mode    |
|                                   |            | that has performance above minimum required threshold          |
|                                   |            | including margins and penalties for all channels (full load)   |
|                                   |            | and 1) fit in the spacing, 2) has the largest baudrate,        |
|                                   |            | 3) has the largest bitrate.                                    |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``spacing``                       | (number)   | Mandatory. In :math:`Hz`. Spacing is used for full spectral    |
|                                   |            | load feasibility evaluation.                                   |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``path_bandwidth``                | (number)   | Mandatory. In :math:`bit/s`. Required capacity on this         |
|                                   |            | service. It is used to determine the needed number of channels |
|                                   |            | and spectrum occupation.                                       |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``max-nb-of-channel``             | (number)   | Optional. Number of channels to take into account for the full |
|                                   |            | load computation. Default value is computed based on f_min     |
|                                   |            | and f_max of transceiver frequency range and min_spacing of    |
|                                   |            | mode (once selected).                                          |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``output-power``                  | (number)   | Optional. In :math:`W`. Target power to be considered at the   |
|                                   |            | fiber span input. Default value uses power defined in  SI in   |
|                                   |            | the library converted in Watt:                                 |
|                                   |            | :math:`10^(power\_dbm/10)`.                                    |
|                                   |            |                                                                |
|                                   |            | Current script gnpy-path-request redesign the network on each  |
|                                   |            | new request, using this power together with                    |
|                                   |            | ``max-nb-of-channel`` to compute target gains or power in      |
|                                   |            | amplifiers. This parameter can therefore be useful to test     |
|                                   |            | different designs with the same script.                        |
|                                   |            |                                                                |
|                                   |            | In order to keep the same design for different requests,       |
|                                   |            | ``max-nb-of-channel` and ``output-power`` of each request      |
|                                   |            | should be kept identical.                                      |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``tx_power``                      | (number)   | Optional. In :math:`W`.  Optical output power emitted by the   |
|                                   |            | transceiver. Default value is output-power.                    |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``effective-freq-slot``           | (list)     | Optional. List of N, M values defining the requested spectral  |
|                                   |            | occupation for this service. N, M use ITU-T G694.1 Flexible    |
|                                   |            | DWDM grid definition.                                          |
|                                   |            | For the flexible DWDM grid, the allowed frequency slots have a |
|                                   |            | nominal central frequency (in :math:`THz`) defined by:         |
|                                   |            | 193.1 + N × 0.00625 where N is a positive or negative integer  |
|                                   |            | including 0                                                    |
|                                   |            | and 0.00625 is the nominal central frequency granularity in    |
|                                   |            | :math:`THz` and a slot width defined by:                       |
|                                   |            | 12.5 × M where M is a positive integer and 12.5 is the slot    |
|                                   |            | width granularity in :math:`GHz`.                              |
|                                   |            | Any combination of frequency slots is allowed as long as       |
|                                   |            | there is no overlap between two frequency slots.               |
|                                   |            | Requested spectrum should be consistent with mode min_spacing  |
|                                   |            | and path_bandwidth: 1) each slot inside the list must be       |
|                                   |            | large enough to fit one carrier with min_spacing width,        |
|                                   |            | 2) total number of channels should be large enough to support  |
|                                   |            | the requested path_bandwidth.                                  |
|                                   |            | Note that gnpy-path-request script uses full spectral load and |
|                                   |            | not this spectrum constraint to compute performance. Thus, the |
|                                   |            | specific mix of channels resulting from the list of requests   |
|                                   |            | is not considered to compute performances.                     |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``route-object-include-exclude``  | (list)     | Optional. Indexed List of routing include/exclude constraints  |
|                                   |            | to compute the path between source and destination.            |
+-----------------------------------+------------+----------------------------------------------------------------+

``route-object-include-exclude`` attributes:

+-----------------------------------+------------+----------------------------------------------------------------+
| field                             |   type     | description                                                    |
+===================================+============+================================================================+
| ``explicit-route-usage``          | (string)   | Mandatory. Only one value is supported: ``route-include-ero``  |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``index``                         | (number)   | Mandatory. Index of the element to be included.                |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``nodes_id``                      | (string)   | Mandatory. UID of the node to include in the path.             |
|                                   |            | It must be listed in the list of elements in topology file.    |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``hop-type``                      | (string)   | Mandatory. One among these two values: ``LOOSE`` or            |
|                                   |            | ``STRICT``.  If LOOSE, constraint may be ignored at            |
|                                   |            | computation time if no solution is found that satisfies the    |
|                                   |            | constraint. If STRICT, constraint MUST be satisfied, else the  |
|                                   |            | computation is stopped and no solution is returned.            |
+-----------------------------------+------------+----------------------------------------------------------------+

- **"synchronization"**:

+-----------------------------------+------------+----------------------------------------------------------------+
| field                             |   type     | description                                                    |
+===================================+============+================================================================+
| ``"relaxable``                    | (boolean)  | Mandatory. Only false is supported.                            |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``disjointness``                  | (string)   | Mandatory. Only ``node link`` is supported.                    |
+-----------------------------------+------------+----------------------------------------------------------------+
| ``request-id-number``             | (list)     | Mandatory. List of ``request-id`` whose path should be         |
|                                   |            | disjointed.                                                    |
+-----------------------------------+------------+----------------------------------------------------------------+

.. code-block:: json

    "synchronization-id": "3",
      "svec": {
        "relaxable": false,
        "disjointness": "node link",
        "request-id-number": [
          "3",
          "1"
        ]

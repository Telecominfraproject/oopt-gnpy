.. _release-notes:

Release change log
==================

Each release introduces some changes and new features.

(prepare text for next release)
**Important Changes:**

The default values for EDFA configuration, including frequency range, gain ripple, noise figure ripple, or dynamic gain tilt
are now hardcoded in parameters.py and are no longer read from the default_edfa_config.json file (the file has been removed).
However, users can define their own custom parameters using the default_config_from_json variable, which should be populated with a file name containing the desired parameter description. This applies to both variable_gain and fixed_gain amplifier types.

This change streamlines the configuration process but requires users to explicitly set parameters through the new
model if the default values do not suit their needs.

v2.11
-----

**New feature**

A new type_def for amplifiers has been introduced: multi_band. This allows the definition of a
multiband amplifier site composed of several amplifiers per band (a typical application is C+L transmission). The
release also includes autodesign for links (Optical Multiplex Section, OMS) composed of multi_band amplifiers.
Multi_band autodesign includes basic tilt and tilt_target calculation when the Raman flag is enabled with the
--sim-params option. The spectrum is demultiplexed before propagation in the amplifier and multiplexed in the output
fiber at the amplifier output.


In the library:

    .. code-block:: json

        {
            "type_variety": "std_medium_gain_C",
            "f_min": 191.225e12,
            "f_max": 196.125e12,
            "type_def": "variable_gain",
            "gain_flatmax": 26,
            "gain_min": 15,
            "p_max": 21,
            "nf_min": 6,
            "nf_max": 10,
            "out_voa_auto": false,
            "allowed_for_design": false
        },
        {
            "type_variety": "std_medium_gain_L",
            "f_min": 186.5e12,
            "f_max": 190.1e12,
            "type_def": "variable_gain",
            "gain_flatmax": 26,
            "gain_min": 15,
            "p_max": 21,
            "nf_min": 6,
            "nf_max": 10,
            "out_voa_auto": false,
            "allowed_for_design": true
        },
        {
            "type_variety": "std_medium_gain_multiband",
            "type_def": "multi_band",
            "amplifiers": [
                "std_medium_gain_C",
                "std_medium_gain_L"
            ],
            "allowed_for_design": false
        },

In the network topology:

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

**Network design**

Optionally, users can define a design target per OMS (single or multi-band), with specific frequency ranges.
Default design bands are defined in the SI.

    .. code-block:: json

      {
          "uid": "roadm Site_A",
          "type": "Roadm",
          "params": {
              "target_pch_out_db": -20,
              "design_bands": [{"f_min": 191.3e12, "f_max": 195.1e12}]
          }
      }

It is possible to define a set of bands in the SI block instead of a single Spectrum Information.
In this case type_variety must be used.
Each set defines a reference channel used for design functions and autodesign.

The default design settings for the path-request-run script have been modified.
Now, design is performed once for the reference channel defined in the SI block of the eqpt_config,
and requests are propagated based on this design.
The --redesign-per-request option can be used to restore previous behaviour
(design using request channel types).

The autodesign function has been updated to insert multiband booster, preamp or inline amplifiers based on the OMS
nature. If nothing is stated (no amplifier defined in the OMS, no design_bands attribute in the ROADM), then
it uses single band Edfas.

**Propagation**

Only carriers within the amplifier bandwidth are propagated, improving system coherence. This more rigorous checking
of the spectrum to be propagated and the amplifier bandwidth may lead to changes in the total number of channels
compared to previous releases. The range can be adjusted by changing the values ​​of ``f_min`` and ``f_max``
in the amplifier library.


``f_min`` and ``f_max`` represent the boundary frequencies of the amplification bandwidth (the entire channel must fit
within this range).
In the example below, a signal center frequency of 190.05THz with a 50GHz width cannot fit within the amplifier band.
Note that this has a different meaning in the SI or Transceiver blocks, where ``f_min`` and ``f_max`` refers to the
minimum / maximum values of the carrier center frequency.

    .. code-block:: json

      {
          "type_variety": "std_booster_L",
          "f_min": 186.55e12,
          "f_max": 190.05e12,
          "type_def": "fixed_gain",
          "gain_flatmax": 21,
          "gain_min": 20,
          "p_max": 21,
          "nf0": 5,
          "allowed_for_design": false
      }


**Display**

The CLI output for the transmission_main_example now displays the channels used for design and simulation,
as well as the tilt target of amplifiers.

  .. code-block:: text

    Reference used for design: (Input optical power reference in span = 0.00dBm,
                                spacing = 50.00GHz
                                nb_channels = 76)

    Channels propagating: (Input optical power deviation in span = 0.00dB,
                          spacing = 50.00GHz,
                          transceiver output power = 0.00dBm,
                          nb_channels = 76)

The CLI output displays the settings of each amplifier:

  .. code-block:: text

    Multiband_amplifier east edfa in Site_A to Site_B
      type_variety:           std_medium_gain_multiband
      type_variety:           std_medium_gain_C    type_variety:           std_medium_gain_L
      effective gain(dB):     20.90                effective gain(dB):     22.19
      (before att_in and before output VOA)        (before att_in and before output VOA)
      tilt-target(dB)         0.00                 tilt-target(dB)         0.00
      noise figure (dB):      6.38                 noise figure (dB):      6.19
      (including att_in)                           (including att_in)
      pad att_in (dB):        0.00                 pad att_in (dB):        0.00
      Power In (dBm):         -1.08                Power In (dBm):         -1.49
      Power Out (dBm):        19.83                Power Out (dBm):        20.71
      Delta_P (dB):           0.90                 Delta_P (dB):           2.19
      target pch (dBm):       0.90                 target pch (dBm):       3.00
      actual pch out (dBm):   -2.09                actual pch out (dBm):   -0.80
      output VOA (dB):        3.00                 output VOA (dB):        3.00


**New feature**

The preturbative Raman and the approximated GGN models are introduced for a faster evaluation of the Raman and
Kerr effects, respectively.
These implementation are intended to reduce the computational effort required by multiband transmission scenarios.

Both the novel models have been validated with exstensive simulations
(see `arXiv:2304.11756 <https://arxiv.org/abs/2304.11756>`_ for the new Raman model and
`jlt:9741324 <https://eeexplore.ieee.org/document/9741324>`_ for the new NLI model).
Additionally, they have been experimentally validated in a laboratory setup composed of commertial equipment
(see `icton:10648172 <https://eeexplore.ieee.org/document/10648172>`_).


v2.10
-----

ROADM impairments can be defined per degree and roadm-path type (add, drop or express).
Minimum loss when crossing a ROADM is no more 0 dB. It can be set per ROADM degree with roadm-path-impairments.

The transceiver output power, which was previously set using the same parameter as the input span power (power_dbm),
can now be set using a different parameter. It can be set as:

  - for all channels, with tx_power_dbm using SI similarly to tx_osnr (gnpy-transmission-example script)

    .. code-block:: json

      "SI": [{
              "f_min": 191.35e12,
              "baud_rate": 32e9,
              "f_max": 196.1e12,
              "spacing": 50e9,
              "power_dbm": 3,
              "power_range_db": [0, 0, 1],
              "roll_off": 0.15,
              "tx_osnr": 40,
              "tx_power_dbm": -10,
              "sys_margins": 2
          }
      ]

  - for certain channels, using -spectrum option and tx_channel_power_dbm option (gnpy-transmission-example script).

    .. code-block:: json

      {
        "spectrum": [
          {
            "f_min": 191.35e12,
            "f_max":193.1e12,
            "baud_rate": 32e9,
            "slot_width": 50e9,
            "power_dbm": 0,
            "roll_off": 0.15,
            "tx_osnr": 40
          },
          {
            "f_min": 193.15e12,
            "f_max":193.15e12,
            "baud_rate": 32e9,
            "slot_width": 50e9,
            "power_dbm": 0,
            "roll_off": 0.15,
            "tx_osnr": 40,
            "tx_power_dbm": -10
          },
          {
            "f_min": 193.2e12,
            "f_max":195.1e12,
            "baud_rate": 32e9,
            "slot_width": 50e9,
            "power_dbm": 0,
            "roll_off": 0.15,
            "tx_osnr": 40
          }
        ]
      }

  - per service using the additional parameter ``tx_power`` which similarly to ``power`` should be defined in Watt (gnpy-path-request script)

    .. code-block:: json

      {
        "path-request": [
          {
            "request-id": "0",
            "source": "trx SITE1",
            "destination": "trx SITE2",
            "src-tp-id": "trx SITE1",
            "dst-tp-id": "trx SITE2",
            "bidirectional": false,
            "path-constraints": {
              "te-bandwidth": {
                "technology": "flexi-grid",
                "trx_type": "Voyager",
                "trx_mode": "mode 1",
                "spacing": 50000000000.0,
                "path_bandwidth": 100000000000.0
              }
            }
          },
          {
            "request-id": "0 with tx_power",
            "source": "trx SITE1",
            "destination": "trx SITE2",
            "src-tp-id": "trx SITE1",
            "dst-tp-id": "trx SITE2",
            "bidirectional": false,
            "path-constraints": {
              "te-bandwidth": {
                "technology": "flexi-grid",
                "trx_type": "Voyager",
                "trx_mode": "mode 1",
                "tx_power": 0.0001,
                "spacing": 50000000000.0,
                "path_bandwidth": 100000000000.0
              }
            }
          }
        ]
      }

v2.9
----

The revision introduces a major refactor that separates design and propagation. Most of these changes have no impact
on the user experience, except the following ones:

**Network design - amplifiers**: amplifier saturation is checked during design in all cases, even if type_variety is
set; amplifier gain is no more computed on the fly but only at design phase.

Before, the design did not consider amplifier power saturation during design if amplifier type_variety was stated.
With this revision, the saturation is always applied:
If design is made for a per channel power that leads to saturation, the target are properly reduced and the design
is freezed. So that when a new simulation is performed on the same network for lower levels of power per channel
the same gain target is applied. Before these were recomputed, changing the gain targets, so the simulation was
not considering the exact same working points for amplifiers in case of saturation.

Note that this case (working with saturation settings) is not recommended.

The gain of amplifiers was estimated on the fly also in case of RamanFiber preceding elements. The refactor now
requires that an estimation of Raman gain of the RamanFiber is done during design to properly compute a gain target.
The Raman gain is estimated at design for every RamanFiber span and also during propagation instead of being only
estimated at propagation stage for those Raman Fiber spans concerned with the transmission. The auto-design is more
accurate for unpropagated spans, but this results in an increase overall computation time.
This will be improved in the future.

**Network design - ROADMs**: ROADM target power settings are verified during design.

Design checks that expected power coming from every directions ingress from a ROADM are consistent with output power
targets. The checks only considers the adjacent previous hop. If the expected power at the input of this ROADM is
lower than the target power on the out-degree of the ROADM, a warning is displayed, and user is asked to review the
input network to avoid this situation. This does not change the design or propagation behaviour.

**Propagation**: amplifier gain target is no more recomputed during propagation. It is now possible to freeze
the design and propagate without automatic changes.

In previous release, gain was recomputed during propagation based on an hypothetical reference noiseless channel
propagation. It was not possible to «freeze» the autodesign, and propagate without recomputing the gain target
of amplifiers.
With this new release, the design is freezed, so that it is possible to compare performances on same basis.

**Display**: "effective pch (dbm)" is removed. Display contains the target pch which is the target power per channel
in dBm, computed based on reference channel used for design and the amplifier delta_p in dB (and before out VOA
contribution). Note that "actual pch out (dBm)" is the actual propagated total power per channel averaged per spectrum
band definition at the output of the amplifier element, including noises and out VOA contribution.

v2.8
----

**Spectrum assignment**: requests can now support multiple slots.
The definition in service file supports multiple assignments (unchanged syntax):

  .. code-block:: json

          "effective-freq-slot": [
            {
              "N": 0,
              "M": 4
            }, {
              "N": 50,
              "M": 4
            }
          ],

But in results, label-hop is now a list of slots and center frequency index:

  .. code-block:: json

          {
            "path-route-object": {
              "index": 4,
              "label-hop": [
                {
                  "N": 0,
                  "M": 4
                }, {
                  "N": 50,
                  "M": 4
                }
              ]
            }
          },

instead of 

  .. code-block:: json

          {
            "path-route-object": {
              "index": 4,
              "label-hop": {
                "N": 0,
                "M": 4
              }
            }
          },



**change in display**: only warnings are displayed ; information are disabled and needs the -v (verbose)
option to be displayed on standard output.

**frequency scaling**: A more accurate description of fiber parameters is implemented, including frequency scaling of
chromatic dispersion, effective area, Raman gain coefficient, and nonlinear coefficient.

In particular:

1. Chromatic dispersion can be defined with ``'dispersion'`` and ``'dispersion_slope'``, as in previous versions, or
with ``'dispersion_per_frequency'``; the latter must be defined as a dictionary with two keys, ``'value'`` and
``'frequency'`` and it has higher priority than the entries ``'dispersion'`` and ``'dispersion_slope'``.
Essential change: In previous versions, when it was not provided the ``'dispersion_slope'`` was calculated in an
involute manner to get a vanishing beta3 , and this was a mere artifact for NLI evaluation purposes (namely to evaluate
beta2 and beta3, not for total dispersion accumulation). Now, the evaluation of beta2 and beta3 is performed explicitly
in the element.py module.

2. The effective area is provided as a scalar value evaluated at the Fiber reference frequency and properly scaled
considering the Fiber refractive indices n1 and n2, and the core radius. These quantities are assumed to be fixed and
are hard coded in the parameters.py module. Essential change: The effective area is always scaled along the frequency.

3. The Raman gain coefficient is properly scaled considering the overlapping of fiber effective area values scaled at
the interacting frequencies. Essential change: In previous version the Raman gain coefficient depends only on
the frequency offset.

4. The nonlinear coefficient ``'gamma'`` is properly scaled considering the refractive index n2 and the scaling
effective area.  Essential change: As the effective area, the nonlinear coefficient is always scaled along the
frequency.

**power offset**: Power equalization now enables defining a power offset in transceiver library to represent
the deviation from the general equalisation strategy defined in ROADMs.

  .. code-block:: json

            "mode": [{
                    "format": "100G",
                    "baud_rate": 32.0e9,
                    "tx_osnr": 35.0,
                    "min_spacing": 50.0e9,
                    "cost": 1,
                    "OSNR": 10.0,
                    "bit_rate": 100.0e9,
                    "roll_off": 0.2,
                    "equalization_offset_db": 0.0
                }, {
                    "format": "200G",
                    "baud_rate": 64.0e9,
                    "tx_osnr": 35.0,
                    "min_spacing": 75.0e9,
                    "cost": 1,
                    "OSNR": 13.0,
                    "bit_rate": 200.0e9,
                    "roll_off": 0.2,
                    "equalization_offset_db": 1.76
                }
            ]

v2.7
----

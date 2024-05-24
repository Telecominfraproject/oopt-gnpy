.. _release-notes:

Release change log
==================

Each release introduces some changes and new features.

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
requires that an estimation of Raman gain of the RamanFiber is done during design to properly computes a gain target.
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

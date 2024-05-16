.. _release-notes:

Release change log
==================

Each release introduces some changes and new features.

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

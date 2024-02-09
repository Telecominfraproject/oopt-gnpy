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

**frequency scaling**: Chromatic dispersion, effective area, Raman Gain coefficient,
and nonlinear coefficient can now be defined with a scaling along frequency.

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

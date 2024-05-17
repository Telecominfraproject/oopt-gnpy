.. _release-notes:

Release change log
==================

Each release introduces some changes and new features.

(prepare text for next release)
ROADM impairments can be defined per degree and roadm-path type (add, drop or express).
Minimum loss when crossing a ROADM is no more 0 dB. It can be set per ROADM degree with roadm-path-impairments.

Transceiver output power can be set independently from span input power. It can be set:

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

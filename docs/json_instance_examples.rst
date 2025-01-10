.. _json-instance-examples:

*********************************************
Equipment and Network description definitions
*********************************************

1. Equipment description
########################

Equipment description defines equipment types and those parameters.
Description is made in JSON file with predefined structure. By default
the scripts **gnpy-transmission-example** and **path-request-run** uses
**eqpt_config.json** file present in `example-data`` folder. It
can be changed with **-e** or **--equipment** command line parameter.
Parsing of JSON file is made with
**gnpy.tools.json_io.load_equipment(equipment_file_path)** and return
value is a dictionary of format **dict[`equipment type`][`subtype`]=object**

1.1. Structure definition
*************************

1.1.1. Equipment types
**********************

Every equipment type is defined in JSON equipment library root with according name and
array of parameters as value.

possible types:

  - Edfa,
  - Fiber,
  - RamanFiber,
  - Roadm,
  - Transceiver

.. code-block:: none

    {"Edfa": [...],
    "Fiber": [...]
    }


1.1.2. Equipment parameters and subtypes
*****************************************


Array of parameters is a list of objects with unordered parameter name
and its value definition. In case of multiple equipment subtypes each
object contains **"type_variety": ”type name”** name:value combination,
if only one subtype exists **"type_variety"** name is not mandatory and
it will be marked with **”default”** value.

.. code-block:: json

    {
      "Edfa": [{
          "type_variety": "std_medium_gain",
          "type_def": "variable_gain",
          "gain_flatmax": 26,
          "gain_min": 15,
          "p_max": 23,
          "nf_min": 6,
          "nf_max": 10,
          "out_voa_auto": false,
          "allowed_for_design": true
        },
        {
          "type_variety": "std_low_gain",
          "type_def": "variable_gain",
          "gain_flatmax": 16,
          "gain_min": 8,
          "p_max": 23,
          "nf_min": 6.5,
          "nf_max": 11,
          "out_voa_auto": false,
          "allowed_for_design": true
          }
        ],
      "Fiber": [{
          "type_variety": "SSMF",
          "dispersion": 1.67e-05,
          "effective_area": 83e-12,
          "pmd_coef": 1.265e-15
        }
      ]
    }



1.2. Equipment parameters by type
*********************************

1.2.1. Amplifier types
**********************

Several types of amplfiers definition are possible:

-  `advanced_model` – with JSON file describing gain/noise figure tilt and
   gain/noise figure ripple. **"advanced_config_from_json"** value
   contains filename.
   The corresponding file must be loaded with `--extra-config` option
   in the **gnpy-transmission-example** or **path-request-run** scripts.


.. code-block:: json-object

    "Edfa":[{
        "type_variety": "high_detail_model_example",
        "gain_flatmax": 25,
        "gain_min": 15,
        "p_max": 21,
        "advanced_config_from_json": "std_medium_gain_advanced_config.json",
        "out_voa_auto": false,
        "allowed_for_design": false
      }
    ]

-  `variable_gain` – with JSON file describing gain figure tilt and gain/noise
   figure ripple. Default config from GNPy is used for DGT, noise ripple and
   gain ripple. User can input its own config with the
   **"default_config_from_json"**. The corresponding file must be loaded with
   `-extra-config option` in the **gnpy-transmission-example** or
   **path-request-run** scripts.
   Note that the extra_config must contain the frequency bandwidth of the
   amplifier. ``f_min`` and ``f_max`` represent the boundary frequencies of
   the amplification bandwidth (the entire channel must fit within this range).
   if present in the amplifier equiment library definition, ``f_min`` and ``f_max``
   are used instead.

.. code-block:: json-object

    "Edfa":[{
        "type_variety": "std_medium_gain",
        "type_def": "variable_gain",
        "gain_flatmax": 26,
        "gain_min": 15,
        "p_max": 23,
        "nf_min": 6,
        "nf_max": 10,
        "out_voa_auto": false,
        "allowed_for_design": true
      },
      {
        "type_variety": "user_defined",
        "type_def": "variable_gain",
        "f_min": 192.0e12,
        "f_max": 195.9e12,
        "gain_flatmax": 25,
        "gain_min": 15,
        "p_max": 21,
        "nf_min": 6,
        "nf_max": 10,
        "default_config_from_json": "user_edfa_config.json",
        "out_voa_auto": false,
        "allowed_for_design": true
      }
    ]

-  `fixed_gain` – with default config from GNPy describing gain figure tilt and gain/noise
   figure ripple. User can input its own config with the
   **"default_config_from_json"**.

.. code-block:: json-object

    "Edfa":[{
        "type_variety": "std_fixed_gain",
        "type_def": "fixed_gain",
        "gain_flatmax": 21,
        "gain_min": 20,
        "p_max": 21,
        "nf0": 5.5,
        "allowed_for_design": false
      }
    ]

- `openroadm` – with default config from GNPy describing gain figure tilt and gain/noise
   figure ripple.

.. code-block:: json-object

    "Edfa":[{
        "type_variety": "openroadm_ila_low_noise",
        "type_def": "openroadm",
        "gain_flatmax": 27,
        "gain_min": 12,
        "p_max": 22,
        "nf_coef": [-8.104e-4,-6.221e-2,-5.889e-1,37.62],
        "allowed_for_design": false
      }
    ]

- `openroadm_preamp` and `openroadm_booster` - with default config from GNPy
   describing gain figure tilt and gain/noise figure ripple.
   The model approximates the noise mask defined by OpenRoadm in :ref:`preamp and booster within an OpenROADM network<ext-nf-model-noise-mask-OpenROADM>`.
   No extra parameters specific to the NF model are accepted.

.. code-block:: json-object

    "Edfa":[{
        "type_variety": "openroadm_mw_mw_preamp",
        "type_def": "openroadm_preamp",
        "gain_flatmax": 27,
        "gain_min": 0,
        "p_max": 22,
        "pmd": 0,
        "pdl": 0,
        "allowed_for_design": false
      },
      {
        "type_variety": "openroadm_mw_mw_booster",
        "type_def": "openroadm_booster",
        "gain_flatmax": 32,
        "gain_min": 0,
        "p_max": 22,
        "pmd": 0,
        "pdl": 0,
        "allowed_for_design": false
      }
    ]

Composed amplifier types
------------------------

- `multiband`
   This type enables the definition of multiband amplifiers that consist of multiple
   single-band amplifier elements, with each amplifier responsible for amplifying a
   different portion of the spectrum. The types of single-band amplifiers that can be
   included in these multiband amplifiers are specified, allowing for multiple options
   to be available for the same spectrum band (for instance, providing several permitted
   type varieties for both the C-band and the L-band). The actual element utilizing the
   type_variety must implement only one option for each band.

.. code-block:: json-object

    "Edfa":[{
          "type_variety": "std_low_gain",
          "f_min": 191.25e12,
          "f_max": 196.15e12,
          "type_def": "variable_gain",
          "gain_flatmax": 16,
          "gain_min": 8,
          "p_max": 21,
          "nf_min": 7,
          "nf_max": 11,
          "out_voa_auto": false,
          "allowed_for_design": true
        }, {
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
              "std_low_gain",
              "std_medium_gain_C",
              "std_medium_gain_L"
          ],
          "allowed_for_design": false
      }
    ]

- `dual_stage`
   This model allows for the combination of pre-defined amplifiers (`advanced_model`,
   `variable_gain`, `fixed_gain`) into a cascade configuration, which consists of
   any two amplifiers already described in the library.
    
    - The preamp_variety specifies the type variety for the first stage.
    - The booster_variety defines the type variety for the second stage.

   One potential application is the creation of a simplified Raman-EDFA hybrid
   amplifier, where a fixed-gain amplifier with a very low to negative noise figure
   serves as the Raman amplifier preamp, and an EDFA serves as the booster.
   Please note that there is currently no connection between the dual-stage
   amplifier definition and the RamanFiber definition for modeling Raman
   amplification. Users should avoid using dual-stage amplifiers modelling Raman
   with a fixed_gain amplifier and RamanFiber simultaneously when defining a
   Raman-amplified link.

.. code-block:: json-object

    "Edfa":[ {
      "type_variety": "std_low_gain",
      "type_def": "variable_gain",
      "gain_flatmax": 16,
      "gain_min": 8,
      "p_max": 23,
      "nf_min": 6.5,
      "nf_max": 11,
      "out_voa_auto": false,
      "allowed_for_design": true
    },
    {
      "type_variety": "4pumps_raman",
      "type_def": "fixed_gain",
      "gain_flatmax": 12,
      "gain_min": 12,
      "p_max": 21,
      "nf0": -1,
      "allowed_for_design": false
    },
    {
      "type_variety": "hybrid_4pumps_lowgain",
      "type_def": "dual_stage",
      "raman": true,
      "gain_min": 25,
      "preamp_variety": "4pumps_raman",
      "booster_variety": "std_low_gain",
      "allowed_for_design": true
    }

1.2.2. Fiber and RamanFiber types
*********************************

Fiber type with its parameters:

.. code-block:: json-object

  {
    "Fiber":[{
        "type_variety": "SSMF",
        "dispersion": 1.67e-05,
        "effective_area": 83e-12,
        "pmd_coef": 1.265e-15
      }
    ],
    "RamanFiber": [
      {
        "type_variety": "SSMF",
        "dispersion": 1.67e-05,
        "effective_area": 83e-12,
        "pmd_coef": 1.265e-15
      }
    ]
  }

The parameters Gamma and Raman efficiency are calculated using the effective area.
In releases prior to version 2.5, Gamma and Raman efficiency were defined instead
of effective area. Both parameters are managed as optional for
backward compatibility.

The RamanFiber is a specialized variant of the regular Fiber, where the simulation
engine incorporates Raman amplification. The actual pump definitions must be
specified within the RamanFiber instance in the topology
(refer to the gnpy.core.elements.RamanFiber class).

More details can be found in :cite:curri_merit_2016.

Raman efficiency is scaled against the effective area using the default Raman
coefficient profile (g0 * A_ff_overlap), where g0 is a Raman coefficient profile
defined for a reference effective area (gnpy.core.parameters.DEFAULT_RAMAN_COEFFICIENT).

If a RamanFiber is defined in the library, a corresponding Fiber must also be defined
with the same type_variety.

It is important to note that since version 2.5, Raman effects (Stimulated Raman Scattering)
are modeled in both Fiber and RamanFiber types if the raman_flag is set to True in the
global simulation definition (using the --sim-params option). The default value for this
flag is False. However, Raman amplification using co-propagation and counter-propagation
is only available in RamanFiber, which requires the flag to be set to True, making the
use of the --sim-params option mandatory.

1.2.3 Roadm types
*****************

Roadm element with its parameters:

Since v2.10, it is possible to define several types of ROADM and to
describe their contribution to optical impairments. This follows a model
created at the IETF: `IETF
CCAMP optical impairment topology <https://github.com/ietf-ccamp-wg/draft-ietf-ccamp-optical-impairment-topology-yang>`_

.. code-block:: json-object

  "Roadm": [
    {
      "target_pch_out_db": -20,
      "add_drop_osnr": 38,
      "pmd": 0,
      "pdl": 0,
      "restrictions": {
        "preamp_variety_list": [],
        "booster_variety_list": []
      }
    },
    {
      "type_variety": "roadm_type_1",
      "target_pch_out_db": -18,
      "add_drop_osnr": 35,
      "pmd": 0,
      "pdl": 0,
      "restrictions": {
        "preamp_variety_list": [],
        "booster_variety_list": []
      },
      "roadm-path-impairments": []
    },
    {
      "type_variety": "detailed_impairments",
      "target_pch_out_db": -20,
      "add_drop_osnr": 38,
      "pmd": 0,
      "pdl": 0,
      "restrictions": {
        "preamp_variety_list": [],
        "booster_variety_list": []
      },
      "roadm-path-impairments": [
        {
          "roadm-path-impairments-id": 0,
          "roadm-express-path": [
            {
              "frequency-range": {
                "lower-frequency": 191.3e12,
                "upper-frequency": 196.1e12
              },
              "roadm-pmd": 0,
              "roadm-cd": 0,
              "roadm-pdl": 0,
              "roadm-inband-crosstalk": 0,
              "roadm-maxloss": 16.5
            }
          ]
        },
        {
          "roadm-path-impairments-id": 1,
          "roadm-add-path": [
            {
              "frequency-range": {
                "lower-frequency": 191.3e12,
                "upper-frequency": 196.1e12
              },
              "roadm-pmd": 0,
              "roadm-cd": 0,
              "roadm-pdl": 0,
              "roadm-inband-crosstalk": 0,
              "roadm-maxloss": 11.5,
              "roadm-pmax": 2.5,
              "roadm-osnr": 41,
              "roadm-noise-figure": 23
            }
          ]
        },
        {
          "roadm-path-impairments-id": 2,
          "roadm-drop-path": [
            {
              "frequency-range": {
                "lower-frequency": 191.3e12,
                "upper-frequency": 196.1e12
              },
              "roadm-pmd": 0,
              "roadm-cd": 0,
              "roadm-pdl": 0,
              "roadm-inband-crosstalk": 0,
              "roadm-maxloss": 11.5,
              "roadm-minloss": 7.5,
              "roadm-typloss": 10,
              "roadm-pmin": -13.5,
              "roadm-pmax": -9.5,
              "roadm-ptyp": -12,
              "roadm-osnr": 41,
              "roadm-noise-figure": 15
            }
          ]
        }
      ]
    }
  ]

1.2.5. Transceiver type
**************************

Transceiver element with its parameters. **”mode”** can contain multiple
Transceiver operation formats.

Note that ``OSNR`` parameter refers to the receiver's minimal OSNR threshold for a given mode.

.. code-block:: json-object

    "Transceiver":[
      {
          "frequency":{
                      "min": 191.35e12,
                      "max": 196.1e12
                      },
          "mode":[
                  {
                      "format": "mode 1",
                      "baud_rate": 32e9,
                      "OSNR": 11,
                      "bit_rate": 100e9,
                      "roll_off": 0.15,
                      "tx_osnr": 40,
                      "min_spacing": 37.5e9,
                      "cost":1
                  },
                  {
                    "format": "mode 2",
                      "baud_rate": 66e9,
                      "OSNR": 15,
                      "bit_rate": 200e9,
                      "roll_off": 0.15,
                      "tx_osnr": 40,
                      "min_spacing": 75e9,
                      "cost":1
                  }
          ]
      }
    ]

1.2.3. Spans element
********************

Spans element with its parameters:

.. code-block:: json-object

    "Spans":[{
            "power_mode":true,
            "delta_power_range_db": [0,0,0.5],
            "max_length": 150,
            "length_units": "km",
            "max_loss": 28,
            "padding": 10,
            "EOL": 0,
            "con_in": 0,
            "con_out": 0
            }
        ]


1.2.4. Spectral Information
***************************

Spectral information with its parameters:

.. code-block:: json-object

    "SI":[{
            "f_min": 191.3e12,
            "baud_rate": 32e9,
            "f_max":195.1e12,
            "spacing": 50e9,
            "power_dbm": 0,
            "tx_power_dbm": 0,
            "power_range_db": [0,0,0.5],
            "roll_off": 0.15,
            "tx_osnr": 40,
            "sys_margins": 0
            }
        ]



2. Network description
######################

Network description defines network elements with additional to
equipment description parameters, metadata and elements interconnection.
Description is made in JSON file with predefined structure. By default
**gnpy-transmission-example** uses **gnpy/example-data/edfa_example_network.json** file
and can be changed from command line. By default
**gnpy-path-request** uses **gnpy/example-data/meshTopologyExampleV2.xls**.
Parsing of JSON file is made with
**gnpy.core.network.load_network(network_description,
equipment_description)** and return value is **DiGraph** object which
mimics network description.

2.1. Structure definition
##########################

2.1.1. File root structure
***************************

Network description JSON file root consist of three unordered parts:

-  network_name – name of described network or service, is not used as
   of now,

-  elements - contains a list of network element objects with their
   respective parameters,

-  connections – contains a list of unidirectional connection objects.

.. code-block:: none

    {"network_name": "Example Network",
    "elements": [{...},
                {...}
                ],
    "connections": [{...},
                    {...}
                    ]
    }


2.1.2. Elements parameters and subtypes
****************************************

The list of network element objects consist of unordered parameter names
and those values. In case of **"type_variety"** absence, the
**"type_variety": ”default”** name:value combination is used.
**"type_variety"** must be defined in equipment library.

2.2. Element parameters by type
*********************************

2.2.1. Transceiver element
***************************

Transceiver element with its parameters.

.. code-block:: json

    {"uid": "trx Site_A",
    "metadata": {
                "location": {
                            "city": "Site_A",
                            "region": "",
                            "latitude": 0,
                            "longitude": 0
                            }
                },
    "type": "Transceiver"
    }

``Transceiver`` elements represent the logical function that generates a spectrum.
This must be specified to start and stop propagation. However, the characteristics of the spectrum
are defined elsewhere, so ``Transceiver`` elements do not contain any attribute.

Information on transceivers' type, modes and frequency must be listed in
:ref:`service file<service>` or :ref:`spectrum file<mixed-rate>`.
Without any definition, default :ref:`SI<spectral_info>` values of the library are propagated.


2.2.2. ROADM element
*********************

ROADM element with its parameters. **“params”** is optional, if nothing is defined,
it uses default values from equipment library.

.. code-block:: json

    {
      "uid": "roadm Lorient_KMA",
      "metadata": {
        "location": {
          "city": "Lorient_KMA",
          "region": "RLD",
          "latitude": 2.0,
          "longitude": 3.0
        }
      },
      "type": "Roadm"
    }

.. code-block:: json

    {
      "uid": "roadm Lannion_CAS",
      "type": "Roadm",
      "type_variety": "default",
      "params": {
        "target_pch_out_db": -20,
        "restrictions": {
          "preamp_variety_list": [],
          "booster_variety_list": []
        },
        "per_degree_pch_out_db": {
          "east edfa in Lannion_CAS to Corlay": -20,
          "east edfa in Lannion_CAS to Stbrieuc": -20,
          "east edfa in Lannion_CAS to Morlaix": -20
        }
      },
      "metadata": {
        "location": {
          "latitude": 2.0,
          "longitude": 0.0,
          "city": "Lannion_CAS",
          "region": "RLD"
        }
      }
    }


2.2.3. Fused element
*********************

Fused element with its parameters. **“params”** is optional, if not used
default loss value of 1dB is used.

.. code-block:: json

    {
      "uid": "ingress fused spans in Site_B",
      "metadata": {
        "location": {
          "city": "Site_B",
          "region": "",
          "latitude": 0,
          "longitude": 0
          }
        },
      "type": "Fused",
      "params": {
        "loss": 0.5
      }
    }


2.2.4. Fiber element
*********************

Fiber element with its parameters.

.. code-block:: json

    {
      "uid": "fiber (Site_A \\u2192 Site_B)",
      "metadata": {
        "location": {
          "city": "",
          "region": "",
          "latitude": 0.0,
          "longitude": 0.0
        }
      },
      "type": "Fiber",
      "type_variety": "SSMF",
      "params": {
        "length": 77.3,
        "length_units": "km",
        "loss_coef": 0.2542
      }
    }

.. code-block:: json

    {
      "uid": "Span2",
      "type": "Fiber",
      "type_variety": "SSMF_freq",
      "params": {
        "length": 80,
        "loss_coef": {
          "value": [0.2121641791044776, 0.20703358208955223, 0.21636194029850745],
          "frequency": [186.3e12, 194e12, 197e12]
        },
        "length_units": "km"
      },
      "metadata": {
        "location": {
          "region": "",
          "latitude": 1,
          "longitude": 0
        }
      }
    },


2.2.5. RamanFiber element
*************************

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
        "loss_coef": 0.2,
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


2.2.6. EDFA element
********************

EDFA element with its parameters.

.. code-block:: json

    {
      "uid": "Edfa1",
      "type": "Edfa",
      "type_variety": "std_low_gain",
      "operational": {
        "gain_target": 16,
        "tilt_target": 0
      },
      "metadata": {
        "location": {
          "city": "Site_A",
          "region": "",
          "latitude": 2,
          "longitude": 0
        }
      }
    }

2.3. Connections objects
*************************

Each unidirectional connection object in connections array consist of
two unordered **”from_node”** and **”to_node”** name pair with values
corresponding to element **”uid”**

.. code-block:: json

    {
      "from_node": "roadm Site_C",
      "to_node": "trx Site_C"
    }


3. Simulation Parameters
########################

Additional details of the simulation are controlled via ``sim_params.json``:

.. code-block:: json

  {
    "raman_params": {
      "flag": true,
      "result_spatial_resolution": 10e3,
      "solver_spatial_resolution": 50
    },
    "nli_params": {
      "method": "ggn_spectrally_separated",
      "dispersion_tolerance": 1,
      "phase_shift_tolerance": 0.1,
      "computed_channels": [1, 18, 37, 56, 75]
    }
  }


4. Services file
################

**gnpy-path-request** requires a second positional file that contains a list of services to be computed.


4.1. Service Excel format
*************************

Services can be defined either via a :ref:`XLS files<excel-service-sheet>`.


4.2. Service JSON format
************************

The JSON format is derived from draft-ietf-teas-yang-path-computation-01.txt.

It contains a list of requests and a list of constraints between the requests, named `synchronization`,
to define disjunctions among services.

.. code-block:: none

    {
      "path-request": [...],
      "synchronization": [...]
    }

4.2.1. requests
***************

**path-request** contains the path and transceiver details.
See :ref:`services<service>` for a detailed description of each parameter.

.. code-block:: json

    {
      "request-id": "1",
      "source": "trx Brest_KLA",
      "destination": "trx Vannes_KBE",
      "src-tp-id": "trx Brest_KLA",
      "dst-tp-id": "trx Vannes_KBE",
      "bidirectional": false,
      "path-constraints": {
        "te-bandwidth": {
          "technology": "flexi-grid",
          "trx_type": "Voyager",
          "trx_mode": "mode 1",
          "effective-freq-slot": [{"N": 0, "M": 4}],
          "spacing": 50000000000.0,
          "tx_power": 0.0005,
          "max-nb-of-channel": null,
          "output-power": 0.0012589254117941673,
          "path_bandwidth": 200000000000.0
        }
      },
      "explicit-route-objects": {
        "route-object-include-exclude": [
          {
            "explicit-route-usage": "route-include-ero",
            "index": 0,
            "num-unnum-hop": {
              "node-id": "roadm Brest_KLA",
              "link-tp-id": "link-tp-id is not used",
              "hop-type": "LOOSE"
            }
          },
          {
            "explicit-route-usage": "route-include-ero",
            "index": 1,
            "num-unnum-hop": {
              "node-id": "roadm Lannion_CAS",
              "link-tp-id": "link-tp-id is not used",
              "hop-type": "LOOSE"
            }
          },
          {
            "explicit-route-usage": "route-include-ero",
            "index": 2,
            "num-unnum-hop": {
              "node-id": "roadm Lorient_KMA",
              "link-tp-id": "link-tp-id is not used",
              "hop-type": "LOOSE"
            }
          },
          {
            "explicit-route-usage": "route-include-ero",
            "index": 3,
            "num-unnum-hop": {
              "node-id": "roadm Vannes_KBE",
              "link-tp-id": "link-tp-id is not used",
              "hop-type": "LOOSE"
            }
          }
        ]
      }
    }

4.2.2. synchronization
**********************

.. code-block:: json

    {
      "synchronization-id": "3",
      "svec": {
        "relaxable": "false",
        "disjointness": "node link",
        "request-id-number": [
          "3",
          "1"
        ]
      }
    }


****************
1. Spectrum file
****************

**gnpy-transmission-example** supports a `--spectrum` option to specify non identical type
of channels derailed in a JSON file (details :ref:`here<mixed-rate>`). Note that **gnpy-path-request**
script does not support this option.

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

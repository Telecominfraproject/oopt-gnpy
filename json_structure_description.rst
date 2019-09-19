*********************************************
Equipment and Network description definitions
*********************************************

1. Equipment description
########################

Equipment description defines equipment types and those parameters.
Description is made in JSON file with predefined structure. By default
**transmission_main_example.py** uses **eqpt_config.json** file and that
can be changed with **-e** or **--equipment** command line parameter.
Parsing of JSON file is made with
**gnpy.core.equipment.load_equipment(equipment_description)** and return
value is a dictionary of format **dict[‘equipment
type’][‘subtype’]=object**

1.1. Structure definition
*************************

1.1.1. Equipment types
*************************

Every equipment type is defined in JSON root with according name and
array of parameters as value.

.. code-block:: none

    {"Edfa": [...],
    "Fiber": [...]
    }


1.1.2. Equipment parameters and subtypes
*****************************************


Array of parameters is a list of objects with unordered parameter name
and its value definition. In case of multiple equipment subtypes each
object contains **"type_variety":”type name”** name:value combination,
if only one subtype exists **"type_variety"** name is not mandatory and
it will be marked with **”default”** value.

.. code-block:: json

    {"Edfa": [{
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
                "gamma": 0.00127
                }
        ]
    }



1.2. Equipment parameters by type
*********************************

1.2.1. EDFA element
*******************

Four types of EDFA definition are possible. Description JSON file
location is in **transmission_main_example.py** folder:

-  Advanced – with JSON file describing gain/noise figure tilt and
   gain/noise figure ripple. **"advanced_config_from_json"** value
   contains filename.

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

-  Variable gain – with JSON file describing gain figure tilt and gain/noise
   figure ripple. **”default_edfa_config.json”** as source file.

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
            }
        ]

-  Fixed gain – with JSON file describing gain figure tilt and gain/noise
   figure ripple. **”default_edfa_config.json”** as source file.

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

- openroadm – with JSON file describing gain figure tilt and gain/noise
   figure ripple. **”default_edfa_config.json”** as source file. 

.. code-block:: json-object

    "Edfa":[{
            "type_variety": "low_noise",
            "type_def": "openroadm",
            "gain_flatmax": 27,
            "gain_min": 12,
            "p_max": 22,
            "nf_coef": [-8.104e-4,-6.221e-2,-5.889e-1,37.62],
            "allowed_for_design": false
            }
        ]

1.2.2. Fiber element
********************

Fiber element with its parameters:

.. code-block:: json-object

    "Fiber":[{
            "type_variety": "SSMF",
            "dispersion": 1.67e-05,
            "gamma": 0.00127
            }
        ]

RamanFiber element
******************

A special variant of the regular ``Fiber`` where the simulation engine accounts for the Raman effect.
The newly added parameters are nested in the ``raman_efficiency`` dictionary.
Its shape corresponds to typical properties of silica.
More details are available from :cite:`curri_merit_2016`.

The ``cr`` property is the normailzed Raman efficiency, so it is is (almost) independent of the fiber type, while the coefficient actually giving Raman gain is g_R=C_R/Aeff.

The ``frequency_offset`` represents the spectral difference between the pumping photon and the one receiving energy.

.. code-block:: json-object

    "RamanFiber":[{
      "type_variety": "SSMF",
      "dispersion": 1.67e-05,
      "gamma": 0.00127,
      "raman_efficiency": {
        "cr":[
            0, 9.4E-06, 2.92E-05, 4.88E-05, 6.82E-05, 8.31E-05, 9.4E-05, 0.0001014, 0.0001069, 0.0001119,
            0.0001217, 0.0001268, 0.0001365, 0.000149, 0.000165, 0.000181, 0.0001977, 0.0002192, 0.0002469,
            0.0002749, 0.0002999, 0.0003206, 0.0003405, 0.0003592, 0.000374, 0.0003826, 0.0003841, 0.0003826,
            0.0003802, 0.0003756, 0.0003549, 0.0003795, 0.000344, 0.0002933, 0.0002024, 0.0001158, 8.46E-05,
            7.14E-05, 6.86E-05, 8.5E-05, 8.93E-05, 9.01E-05, 8.15E-05, 6.67E-05, 4.37E-05, 3.28E-05, 2.96E-05,
            2.65E-05, 2.57E-05, 2.81E-05, 3.08E-05, 3.67E-05, 5.85E-05, 6.63E-05, 6.36E-05, 5.5E-05, 4.06E-05,
            2.77E-05, 2.42E-05, 1.87E-05, 1.6E-05, 1.4E-05, 1.13E-05, 1.05E-05, 9.8E-06, 9.8E-06, 1.13E-05,
            1.64E-05, 1.95E-05, 2.38E-05, 2.26E-05, 2.03E-05, 1.48E-05, 1.09E-05, 9.8E-06, 1.05E-05, 1.17E-05,
            1.25E-05, 1.21E-05, 1.09E-05, 9.8E-06, 8.2E-06, 6.6E-06, 4.7E-06, 2.7E-06, 1.9E-06, 1.2E-06, 4E-07,
            2E-07, 1E-07
        ],
        "frequency_offset":[
          0, 0.5e12, 1e12, 1.5e12, 2e12, 2.5e12, 3e12, 3.5e12, 4e12, 4.5e12, 5e12, 5.5e12, 6e12, 6.5e12, 7e12,
          7.5e12, 8e12, 8.5e12, 9e12, 9.5e12, 10e12, 10.5e12, 11e12, 11.5e12, 12e12, 12.5e12, 12.75e12,
          13e12, 13.25e12, 13.5e12, 14e12, 14.5e12, 14.75e12, 15e12, 15.5e12, 16e12, 16.5e12, 17e12,
          17.5e12, 18e12, 18.25e12, 18.5e12, 18.75e12, 19e12, 19.5e12, 20e12, 20.5e12, 21e12, 21.5e12,
          22e12, 22.5e12, 23e12, 23.5e12, 24e12, 24.5e12, 25e12, 25.5e12, 26e12, 26.5e12, 27e12, 27.5e12, 28e12,
          28.5e12, 29e12, 29.5e12, 30e12, 30.5e12, 31e12, 31.5e12, 32e12, 32.5e12, 33e12, 33.5e12, 34e12, 34.5e12,
          35e12, 35.5e12, 36e12, 36.5e12, 37e12, 37.5e12, 38e12, 38.5e12, 39e12, 39.5e12, 40e12, 40.5e12, 41e12,
          41.5e12, 42e12
        ]
        }
      }
    ]


1.2.3 Roadm element
*******************

Roadm element with its parameters:

.. code-block:: json-object

      "Roadms":[{
            "gain_mode_default_loss": 20,
            "power_mode_pout_target": -20,
            "add_drop_osnr": 38
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
            "power_range_db": [0,0,0.5],
            "roll_off": 0.15,
            "tx_osnr": 40,
            "sys_margins": 0
            }
        ]


1.2.5. Transceiver element
**************************

Transceiver element with its parameters. **”mode”** can contain multiple
Transceiver operation formats.

Note that ``OSNR`` parameter refers to the receiver's minimal OSNR threshold for a given mode.

.. code-block:: json-object

    "Transceiver":[{
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

***********************
2. Network description
***********************

Network description defines network elements with additional to
equipment description parameters, metadata and elements interconnection.
Description is made in JSON file with predefined structure. By default
**transmission_main_example.py** uses **edfa_example_network.json** file
and can be changed from command line. Parsing of JSON file is made with
**gnpy.core.network.load_network(network_description,
equipment_description)** and return value is **DiGraph** object which
mimics network description.

2.1. Structure definition
##########################

2.1.1. File root structure
***************************

Network description JSON file root consist of three unordered parts:

-  network_name – name of described network or service, is not used as
   of now

-  elements - contains array of network element objects with their
   respective parameters

-  connections – contains array of unidirectional connection objects

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

Array of network element objects consist of unordered parameter names
and those values. In case of **"type_variety"** absence
**"type_variety":”default”** name:value combination is used. As of the
moment, existence of used **"type_variety"** in equipment description is
obligatory.

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



2.2.2. ROADM element
*********************

ROADM element with its parameters. **“params”** is optional, if not used
default loss value of 20dB is used.

.. code-block:: json

    {"uid": "roadm Site_A",
    "metadata": {
                "location": {
                            "city": "Site_A",
                            "region": "",
                            "latitude": 0,
                            "longitude": 0
                            }
                },
    "type": "Roadm",
    "params": {
                "loss": 17
            }
    }


2.2.3. Fused element
*********************

Fused element with its parameters. **“params”** is optional, if not used
default loss value of 1dB is used.

.. code-block:: json

    {"uid": "ingress fused spans in Site_B",
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

    {"uid": "fiber (Site_A \\u2192 Site_B)",
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
                "length": 40.0,
                "length_units": "km",
                "loss_coef": 0.2
                }
    }

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
            "power": 200e-3,
            "frequency": 205e12,
            "propagation_direction": "counterprop"
          },
          {
            "power": 206e-3,
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

    {"uid": "Edfa1",
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

    {"from_node": "roadm Site_C",
    "to_node": "trx Site_C"
    }

************************
3. Simulation Parameters
************************

Additional details of the simulation are controlled via ``sim_params.json``:

.. code-block:: json

  {
    "raman_computed_channels": [1, 18, 37, 56, 75],
    "raman_parameters": {
      "flag_raman": true,
      "space_resolution": 10e3,
      "tolerance": 1e-8
    },
    "nli_parameters": {
      "nli_method_name": "ggn_spectrally_separated",
      "wdm_grid_size": 50e9,
      "dispersion_tolerance": 1,
      "phase_shift_tollerance": 0.1
    }
  }

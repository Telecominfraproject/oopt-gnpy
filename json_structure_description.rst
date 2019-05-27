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


2.2.5. EDFA element
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

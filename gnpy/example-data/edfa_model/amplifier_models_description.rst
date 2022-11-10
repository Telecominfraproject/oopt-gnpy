*********************************************
Amplifier models and configuration
*********************************************


1. Equipment configuration description
#######################################

Equipment description defines equipment types and parameters.
It takes place in the default **eqpt_config.json** file. 
By default **gnpy-transmission-example** uses **eqpt_config.json** file and that
can be changed with **-e** or **--equipment** command line parameter.

2. Amplifier parameters and subtypes
#######################################

Several amplifiers can be used by GNpy, so they are defined as an array of equipment parameters in **eqpt_config.json** file.

- *"type_variety"*:
    Each amplifier is identified by its unique *"type_variety"*, which is used in the topology files input to reference a specific amplifier. It is a user free defined id.
    
    For each amplifier *type_variety*, specific parameters are describing its attributes and performance:

- *"type_def"*:
    Sets the amplifier model that the simulation will use to calculate the ase noise contribution. 5 models are defined with reserved words:

    - *"advanced_model"*
    - *"variable_gain"*
    - *"fixed_gain"*
    - *"dual_stage"*
    - *"openroadm"*
        *see next section for a full description of these models*

- *"advanced_config_from_json"*:
    **This parameter is only applicable to the _"advanced_model"_ model**
    
    json file name describing:

    - nf_fit_coeff
    - f_min/max
    - gain_ripple
    - nf_ripple 
    - dgt
    
    *see next section for a full description*

- *"gain_flatmax"*: 
    amplifier maximum gain in dB before its extended gain range: flat or nominal tilt output. 
    
    If gain > gain_flatmax, the amplifier will tilt, based on its dgt function

    If gain > gain_flatmax + target_extended_gain, the amplifier output power is reduced to  not exceed the extended gain range.

- *"gain_min"*: 
    amplifier minimum gain in dB.

    If gain < gain_min, the amplifier input is automatically padded, which results in

    NF += gain_min - gain 

- *"p_max"*: 
    amplifier max output power, full load

    Total signal output power will not be allowed beyond this value

- *"nf_min/max"*:
    **These parameters are only applicable to the _"variable_gain"_ model**

    min & max NF values in dB

    NF_min is the amplifier NF @ gain_max  

    NF_max is the amplifier NF @ gain_min  

- *"nf_coef"*: 
    **This parameter is only applicable to the *"openroadm"* model**

    [a, b, c, d] 3rd order polynomial coefficients list to define the incremental OSNR vs Pin
    
    Incremental OSNR is the amplifier OSNR contribution
    
    Pin is the amplifier channel input power defined in a 50GHz bandwidth
    
    Incremental OSNR = a*Pin³ + b*Pin² + c*Pin + d

- *"preamp_variety"*: 
    **This parameter is only applicable to the _"dual_stage"_ model**

    1st stage type_variety

- *"booster_variety"*: 
    **This parameter is only applicable to the *"dual_stage"* model**

    2nd stage type_variety

- *"out_voa_auto"*: true/false
    **power_mode only**

    **This parameter is only applicable to the *"advanced_model"* and *"variable_gain"* models**

    If "out_voa_auto": true, auto_design will chose the output_VOA value that maximizes the amplifier gain within its power capability and therefore minimizes its NF.

- *"allowed_for_design"*: true/false
    **auto_design only**

    Tells auto_design if this amplifier can be picked for the design (deactivates unwanted amplifiers)

    It does not prevent the use of an amplifier if it is placed in the topology input.

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
            ]}


3. Amplifier models
#######################################

In an opensource and multi-vendor environnement, it is needed to support different use cases and context. Therefore several models are supported for amplifiers.

5 types of EDFA definition are possible and referenced by the *"type_def"* parameter with the following reserved words:

-  *"advanced_model"* 
    This model is refered as a whitebox model because of the detailed level of knowledge that is required. The amplifier NF model and ripple definition are described by a json file referenced with *"advanced_config_from_json"*: json filename. This json file contains:

    - nf_fit_coeff: [a,b,c,d]
         
        3rd order polynomial NF = f(-dg) coeficients list

        dg = gain - gain_max

    - f_min/max: amplifier frequency range in Hz
    - gain_ripple : [...]

        amplifier gain ripple excursion comb list in dB across the frequency range.
    - nf_ripple : [...]
        
        amplifier nf ripple excursion comb list in dB across the frequency range. 
    - dgt : [...]
        amplifier dynamic gain tilt comb list across the frequency range.
            
        *See next section for the generation of this json file*

    .. code-block:: json-object

        "Edfa":[{
                "type_variety": "high_detail_model_example",
                "type_def": "advanced_model",
                "gain_flatmax": 25,
                "gain_min": 15,
                "p_max": 21,
                "advanced_config_from_json": "std_medium_gain_advanced_config.json",
                "out_voa_auto": false,
                "allowed_for_design": false
                }
            ]

- *"variable_gain"* 
    This model is refered as an operator model because a lower level of knowledge is required. A full polynomial description of the NF cross the gain range is not required. Instead, NF_min and NF_max values are required and used by the code to model a dual stage amplifier with an internal mid stage VOA. NF_min and NF_max values are typically available from equipment suppliers data-sheet.

    There is a default JSON file ”default_edfa_config.json”* to enforce 0 tilt and ripple values because GNpy core algorithm is a multi-carrier propogation.
    - gain_ripple =[0,...,0]
    - nf_ripple = [0,...,0]
    - dgt = [...] generic dgt comb

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

-  *"fixed_gain"* 
    This model is also an operator model with a single NF value that emulates basic single coil amplifiers without internal VOA.

    if gain_min < gain < gain_max, NF == nf0
    
    if gain < gain_min, the amplifier input is automatically padded, which results in 

    NF += gain_min - gain

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

- *"openroadm"* 
    This model is a black box model replicating OpenRoadm MSA spec for ILA.

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

- *"dual_stage"* 
    This model allows the cascade (pre-defined combination) of any 2 amplifiers already described in the eqpt_config.json library.
    
    - preamp_variety defines the 1st stge type variety
    
    - booster variety defines the 2nd stage type variety
    
    Both preamp and booster variety must exist in the eqpt libray
    The resulting NF is the sum of the 2 amplifiers 
    The preamp is operated to its maximum gain
    
    - gain_min indicates to auto_design when this dual_stage should be used
    
    But unlike other models the 1st stage input will not be padded: it is always operated to its maximu gain and min NF. Therefore if gain adaptation and padding is needed it will be performed by the 2nd stage.

    .. code-block:: json

                {
                "type_variety": "medium+low_gain",
                "type_def": "dual_stage",
                "gain_min": 25,
                "preamp_variety": "std_medium_gain",
                "booster_variety": "std_low_gain",
                "allowed_for_design": true
                }

4. advanced_config_from_json 
#######################################

The build_oa_json.py library in ``gnpy/example-data/edfa_model/`` can be used to build the json file required for the amplifier advanced_model type_def:

Update an existing json file with all the 96ch txt files for a given amplifier type
amplifier type 'OA_type1' is hard coded but can be modified and other types added
returns an updated amplifier json file: output_json_file_name = 'edfa_config.json'
amplifier file names

Convert a set of amplifier files + input json definiton file into a valid edfa_json_file:

nf_fit_coeff: NF 3rd order polynomial coefficients txt file

nf = f(dg) with dg = gain_operational - gain_max

nf_ripple: NF ripple excursion txt file

gain_ripple: gain ripple txt file

dgt: dynamic gain txt file

input json file in argument (defult = 'OA.json')

the json input file should have the following fields:

    .. code-block:: json

        {
            "nf_fit_coeff": "nf_filename.txt",
            "nf_ripple": "nf_ripple_filename.txt", 
            "gain_ripple": "DFG_filename.txt",
            "dgt": "DGT_filename.txt"
        }


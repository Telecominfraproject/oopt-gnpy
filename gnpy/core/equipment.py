#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
nf model parameters calculation
calculate nf1, nf2 and Delta_P of a 2 coils edfa with internal VOA
from nf_min and nf_max inputs 
'''
import numpy as np
from pathlib import Path
from gnpy.core.utils import lin2db, db2lin, load_json

GAIN_MIN_FIELD = "gain_min"
GAIN_MAX_FIELD = "gain_flatmax"
NF_MIN_FIELD  ="nf_min"
NF_MAX_FIELD = "nf_max"

DEFAULT_CONFIG_JSON_FILENAME = 'default_edfa_config.json'
ADVANCED_CONFIG_JSON_FILENAME = 'advanced_config_from_json'


def nf_model(amp_dict):
    gain_min = amp_dict[GAIN_MIN_FIELD]
    gain_max = amp_dict[GAIN_MAX_FIELD]
    nf_min = amp_dict.get(NF_MIN_FIELD,-100)
    nf_max = amp_dict.get(NF_MAX_FIELD,-100)
    if nf_min<-10 or nf_max<-10:
        raise ValueError(f'invalid or missing nf_min or nf_max values in eqpt_config.json for {amp_dict["type_variety"]}')
    nf_min = amp_dict.get(NF_MIN_FIELD,-100)
    nf_max = amp_dict.get(NF_MAX_FIELD,-100)
    #use NF estimation model based on NFmin and NFmax in json OA file
    delta_p = 5 #max power dB difference between 1st and 2nd stage coils
    #dB g1a = (1st stage gain) - (internal voa attenuation)
    g1a_min = gain_min - (gain_max-gain_min) - delta_p
    g1a_max = gain_max - delta_p
    #nf1 and nf2 are the nf of the 1st and 2nd stage coils
    #calculate nf1 and nf2 values that solve nf_[min/max] = nf1 + nf2 / g1a[min/max]
    nf2 = lin2db((db2lin(nf_min) - db2lin(nf_max)) / (1/db2lin(g1a_max)-1/db2lin(g1a_min)))
    nf1 = lin2db(db2lin(nf_min)- db2lin(nf2)/db2lin(g1a_max)) #expression (1)

    #now checking and recalculating the results:
    #recalculate delta_p to check it is within [1-6] boundaries
    #This is to check that the nf_min and nf_max values from the json file
    #make sense. If not a warning is printed 
    if nf1 < 4:
        print('1st coil nf calculated value {} is too low: revise inputs'.format(nf1))
    if nf2 < nf1 + 0.3 or nf2 > nf1 + 2: 
        #nf2 should be with [nf1+0.5 - nf1 +2] boundaries
        #there shouldn't be very high nf differences between 2 coils
        #=> recalculate delta_p            
        nf2 = max(nf2, nf1+0.3)
        nf2 = min(nf2, nf1+2)
        g1a_max = lin2db(db2lin(nf2) / (db2lin(nf_min) - db2lin(nf1))) #use expression (1)
        delta_p = gain_max - g1a_max
        g1a_min = gain_min - (gain_max-gain_min) - delta_p
        if delta_p < 1 or delta_p > 6:
            #delta_p should be > 1dB and < 6dB => consider user warning if not
            print('!WARNING!: 1st coil vs 2nd coil calculated DeltaP {} is not valid: revise inputs'
                        .format(delta_p))
    #check the calculated values for nf1 & nf2:
    nf_min_calc = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_max))
    nf_max_calc = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_min))
    if (abs(nf_min_calc-nf_min) > 0.01) or (abs(nf_max_calc-nf_max) > 0.01):
        raise ValueError(f'nf model calculation failed with nf_min {nf_min_calc} and nf_max {nf_max_calc} calculated')
    return nf1, nf2, delta_p


def read_eqpt_library(filename):
    """build global dictionnary eqpt_library that stores all eqpt characteristics:
    edfa type type_variety, fiber type_variety
    from the eqpt_config.json (filename parameter)
    also read advanced_config_from_json file parameters for edfa if they are available:
    typically nf_ripple, dfg gain ripple, dgt and nf polynomial nf_fit_coeff
    if advanced_config_from_json file parameter is not present: use nf_model:
    requires nf_min and nf_max values boundaries of the edfa gain range
    """
    global eqpt_library
    eqpt_library = load_json(filename)
    for i, el in enumerate(eqpt_library['Edfa']): 
        dict_nf_model = {}
        if 'advanced_config_from_json' in el:
            #use advanced amplifier model with full ripple characterization
            config_json_filename = el.pop(ADVANCED_CONFIG_JSON_FILENAME)
            dict_nf_model['nf_model'] = {'enabled': False}
        else:
            #use a default ripple model (only default dgt is defined)
            config_json_filename = DEFAULT_CONFIG_JSON_FILENAME
            (nf1, nf2, delta_p) = nf_model(el)
            #remove nf_min and nf_max field and replace by nf1, nf2 & delta_p
            nf_min = el.pop('nf_min','')
            nf_max = el.pop('nf_max','')
            dict_nf_model['nf_model'] = {"enabled": True, "nf1": nf1, "nf2": nf2, "delta_p": delta_p}
        json_data = load_json(Path(filename).parent / config_json_filename)
        eqpt_library['Edfa'][i] = {**el, **json_data, **dict_nf_model}


def get_eqpt_params(eqpt_name):
    """returns the params attributs of an Edfa or Fiber 
    by finding it in the eqpt_library
    input parameter eqpt_name = type_variety of the eqpt
    """
    eqpt_config={}
    try:
        #go through the eqpt library to find the demanded type_variety
        eqpt = next(eqpt for eqpt_type in eqpt_library
                    for eqpt in eqpt_library[eqpt_type] 
                    if eqpt.get('type_variety','') == eqpt_name)
        eqpt_config = dict(eqpt)
        del eqpt_config['type_variety']
    except StopIteration as e:
        print(f'cannot find eqpt {eqpt_name} in eqpt library')
        sys.exit(1)
        #TODO log it
    return eqpt_config

def generate_edfa_config(eqpt_name, gain, tilt):
    """returns the full config
    """
    config = dict(zip(
            ['params','operational'],
            [get_eqpt_config(eqpt_name),
            {
                "gain_target": gain,
                "tilt_target": tilt
            }]))

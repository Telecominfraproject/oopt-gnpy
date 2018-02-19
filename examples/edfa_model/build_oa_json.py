#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:32:00 2018

@author: jeanluc-auge
@comments about amplifier input files from Brian Taylor & Dave Boertjes

update an existing json file with all the 96ch txt files for a given amplifier type
amplifier type 'OA_type1' is hard coded but can be modified and other types added
returns an updated amplifier json file: output_json_file_name = 'edfa_config.json'
"""
import re
import sys
import json
import numpy as np
from gnpy.core.utils import lin2db, db2lin

"""amplifier file names
convert a set of amplifier files + input json definiton file into a valid edfa_json_file:
nf_fit_coeff: NF polynomial coefficients txt file (optional)
nf_ripple: NF ripple excursion txt file
dfg: gain txt file
dgt: dynamic gain txt file
input json file in argument (defult = 'OA.json')
the json input file should have the following fields:
{
    "gain_flatmax": 25,
    "gain_min": 15,
    "p_max": 21,
    "nf_fit_coeff": "pNFfit3.txt",
    "nf_ripple": "NFR_96.txt", 
    "dfg": "DFG_96.txt",
    "dgt": "DGT_96.txt",
    "nf_model": 
        {
        "enabled": true,
        "nf_min": 5.8,
        "nf_max": 10
        }
}
gain_flat = max flat gain (dB)
gain_min = min gain (dB) : will consider an input VOA if below (TBD vs throwing an exception)
p_max = max power (dBm)
nf_fit = boolean (True, False) : 
        if False nf_fit_coeff are ignored and nf_model fields are used
"""

input_json_file_name = "OA.json" #default path
output_json_file_name = "edfa_config.json"
param_field  ="params"
gain_min_field = "gain_min"
gain_max_field = "gain_flatmax"
gain_ripple_field = "dfg"
nf_ripple_field = "nf_ripple"
nf_fit_coeff = "nf_fit_coeff"
nf_model_field = "nf_model"
nf_model_enabled_field = "enabled"
nf_min_field  ="nf_min"
nf_max_field = "nf_max"

def read_file(field, file_name):
    """read and format the 96 channels txt files describing the amplifier NF and ripple
        convert dfg into gain ripple by removing the mean component
    """

    #with open(path + file_name,'r') as this_file:
    #   data = this_file.read()
    #data.strip()
    #data = re.sub(r"([0-9])([ ]{1,3})([0-9-+])",r"\1,\3",data)
    #data = list(data.split(","))
    #data = [float(x) for x in data]
    data = np.loadtxt(file_name)
    if field == gain_ripple_field or field == nf_ripple_field:
        #consider ripple excursion only to avoid redundant information
        #because the max flat_gain is already given by the 'gain_flat' field in json
        #remove the mean component
        data = data - data.mean()
    data = data.tolist()
    return data

def nf_model(amp_dict):
    if amp_dict[nf_model_field][nf_model_enabled_field] == True:
        gain_min = amp_dict[gain_min_field]
        gain_max = amp_dict[gain_max_field]
        nf_min = amp_dict[nf_model_field][nf_min_field]
        nf_max = amp_dict[nf_model_field][nf_max_field]
        #use NF estimation model based on NFmin and NFmax in json OA file
        delta_p = 5 #max power dB difference between 1st and 2nd stage coils
        #dB g1a = (1st stage gain) - (internal voa attenuation)
        g1a_min = gain_min - (gain_max-gain_min) - delta_p
        g1a_max = gain_max - delta_p
        #nf1 and nf2 are the nf of the 1st and 2nd stage coils
        #calculate nf1 and nf2 values that solve nf_[min/max] = nf1 + nf2 / g1a[min/max]
        nf2 = lin2db((db2lin(nf_min) - db2lin(nf_max)) / (1/db2lin(g1a_max)-1/db2lin(g1a_min)))
        nf1 = lin2db(db2lin(nf_min)- db2lin(nf2)/db2lin(g1a_max)) #expression (1)

        """ now checking and recalculating the results:
        recalculate delta_p to check it is within [1-6] boundaries
        This is to check that the nf_min and nf_max values from the json file
        make sense. If not a warning is printed """
        if nf1 < 4:
            print('1st coil nf calculated value {} is too low: revise inputs'.format(nf1))
        if nf2 < nf1 + 0.3 or nf2 > nf1 + 2: 
            """nf2 should be with [nf1+0.5 - nf1 +2] boundaries
            there shouldn't be very high nf differences between 2 coils
            => recalculate delta_p 
            """            
            nf2 = max(nf2, nf1+0.3)
            nf2 = min(nf2, nf1+2)
            g1a_max = lin2db(db2lin(nf2) / (db2lin(nf_min) - db2lin(nf1))) #use expression (1)
            delta_p = gain_max - g1a_max
            g1a_min = gain_min - (gain_max-gain_min) - delta_p
            if delta_p < 1 or delta_p > 6:
                #delta_p should be > 1dB and < 6dB => consider user warning if not
                print('1st coil vs 2nd coil calculated DeltaP {} is not valid: revise inputs'
                            .format(delta_p))
        #check the calculated values for nf1 & nf2:
        nf_min_calc = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_max))
        nf_max_calc = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_min))
        if (abs(nf_min_calc-nf_min) > 0.01) or (abs(nf_max_calc-nf_max) > 0.01):
            print('nf model calculation failed with nf_min {} and nf_max {} calculated'
                    .format(nf_min_calc, nf_max_calc))
            print('do not use the generated edfa_config.json file')
    else :
        (nf1, nf2, delta_p) = (0, 0, 0)

    return (nf1, nf2, delta_p)

def input_json(path):
    """read the json input file and add all the 96 channels txt files
    create the output json file with output_json_file_name"""
    with open(path,'r') as edfa_json_file:
        amp_text = edfa_json_file.read()
    amp_dict = json.loads(amp_text)

    for k, v in amp_dict.items():
        if re.search(r'.txt$',str(v)) :
            amp_dict[k] = read_file(k, v)

    #calculate nf of 1st and 2nd coil for the nf_model if 'enabled'==true
    (nf1, nf2, delta_p) = nf_model(amp_dict)
    #rename nf_min and nf_max in nf1 and nf2 after the nf model calculation:
    del amp_dict[nf_model_field][nf_min_field]
    del amp_dict[nf_model_field][nf_max_field]
    amp_dict[nf_model_field]['nf1'] = nf1
    amp_dict[nf_model_field]['nf2'] = nf2
    amp_dict[nf_model_field]['delta_p'] = delta_p
    #rename dfg into gain_ripple after removing the average part:
    amp_dict['gain_ripple'] = amp_dict.pop(gain_ripple_field)

    new_amp_dict = {}
    new_amp_dict[param_field] = amp_dict
    amp_text = json.dumps(new_amp_dict, indent=4)
    #print(amp_text)
    with  open(output_json_file_name,'w') as edfa_json_file:
        edfa_json_file.write(amp_text)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = input_json_file_name
    input_json(path)
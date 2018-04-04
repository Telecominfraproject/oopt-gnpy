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
output_json_file_name = "default_edfa_config.json"
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
    print(len(data), file_name)
    if field == gain_ripple_field or field == nf_ripple_field:
        #consider ripple excursion only to avoid redundant information
        #because the max flat_gain is already given by the 'gain_flat' field in json
        #remove the mean component
        data = data - data.mean()
    data = data.tolist()
    return data

def input_json(path):
    """read the json input file and add all the 96 channels txt files
    create the output json file with output_json_file_name"""
    with open(path,'r') as edfa_json_file:
        amp_text = edfa_json_file.read()
    amp_dict = json.loads(amp_text)

    for k, v in amp_dict.items():
        if re.search(r'.txt$',str(v)) :
            amp_dict[k] = read_file(k, v)

    amp_text = json.dumps(amp_dict, indent=4)
    #print(amp_text)
    with  open(output_json_file_name,'w') as edfa_json_file:
        edfa_json_file.write(amp_text)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = input_json_file_name
    input_json(path)
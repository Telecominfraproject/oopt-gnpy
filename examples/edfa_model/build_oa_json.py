#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:32:00 2018

@author: jeanluc-auge
@comments about amplifier input files from Brian Taylor & Dave Boertjes

update an existing json file with all the 96ch txt files for a given amplifier type
amplifier type 'OA_type1' is hard coded but can be modified and other types added
returns an updated amplifier json file: output_json_file_name = 'newOA.json'
"""
import re
import json
import numpy as np

"""amplifier file names
there is one set of files / amplifier type:
NF polynomial coefficients txt file (optional)
NF ripple excursion txt file (optional)
gain ripple excursion txt file
dgt function txt file
+ requires an input and output json file name
the json input file should have prefilled fields for each amplifier type:
{
"OA_type1": {
        "type":"EDFA",
        "gain_flat":22,
        "gain_min":15,
        "p_max":21,
        "use_nf_fit":"False",
        "nf_model": 
            {
        	"nf_min":5.5,
        	"nf_max":8,
        	"delta_p":5
            }
        }
}
type = EDFA, RAMAN(not yet supported)
gain_flat = max flat gain (dB)
gain_min = min gain (dB) : will consider an input VOA if below (TBD vs throwing an exception)
p_max = max power (dBm)
nf_fit = boolean (True, False) : if False the nf_fit_file_name is ignored and nf_model field is used
"""
input_json_file_name = 'OA.json'
output_json_file_name = 'newOA.json'

"""
pNFfit3:  Cubic polynomial fit coefficients to noise figure in dB 
averaged across wavelength as a function of gain change from design flat:  

    NFavg = pNFfit3(1)*dG^3 + pNFfit3(2)*dG^2 pNFfit3(3)*dG + pNFfit3(4)
where 
    dG = GainTarget - average(DFG_96)
note that dG will normally be a negative value.
=> json field 'nf_fit3'
"""
nf_fit_file_name = 'pNFfit3.txt'
nf_fit_field = 'nf_fit3'

"""NFR_96:  Noise figure ripple in dB away from the average noise figure
across the band.  This captures the wavelength dependence of the NF.  To
calculate the NF across channels, one uses the cubic fit coefficients
with the external gain target to get the average nosie figure, NFavg and 
then adds this to NFR_96:
NF_96 = NFR_96 + NFavg
=> json field 'nf_ripple'
""" 
nf_ripple_file_name = 'NFR_96.txt'
nf_ripple_field = 'nf_ripple'

"""
DFG_96:  Design flat gain at each wavelength in the 96 channel 50GHz ITU
grid in dB.  This can be experimentally determined by measuring the gain 
at each wavelength using a full, flat channel (or ASE) load at the input.
The amplifier should be set to its maximum flat gain (tilt = 0dB).  This 
measurement captures the ripple of the amplifier.  If the amplifier was 
designed to be mimimum ripple at some other tilt value, then the ripple
reflected in this measurement will not be that minimum.  However, when
the DGT gets applied through the provisioning of tilt, the model should
accurately reproduce the expected ripple at that tilt value.  One could
also do the measurement at some expected tilt value and back-calculate
this vector using the DGT method.  Alternatively, one could re-write the
algorithm to accept a nominal tilt and a tiled version of this vector.
=> json field 'gain_ripple'
"""
gain_ripple_file_name = 'DFG_96.txt'
gain_ripple_field = 'gain_ripple'

"""
DGT_96:  This is the so-called Dynamic Gain Tilt of the EDFA in dB/dB. It
is the change in gain at each wavelength corresponding to a 1dB change at
the longest wavelength supported.  The value can be obtained
experimentally or through analysis of the cross sections or Giles
parameters of the Er fibre.  This is experimentally measured by changing 
the gain of the amplifier above the maximum flat gain while not changing 
the internal VOA (i.e. the mid-stage VOA is set to minimum and does not 
change during the measurement). Note that the measurement can change the 
gain by an arbitrary amount and divide by the gain change (in dB) which
is measured at the reference wavelength (the red end of the band).
=> json field 'dgt'
"""
dgt_file_name = 'DGT_96.txt'
dgt_field = 'dgt'

def read_file(path, file_name, field):
	"""read and format the 96 channels txt files describing the amplifier NF and ripple"""
	#with open(path + file_name,'r') as this_file:
	#	data = this_file.read()
	#data.strip()
	#data = re.sub(r"([0-9])([ ]{1,3})([0-9-+])",r"\1,\3",data)
	#data = list(data.split(","))
	#data = [float(x) for x in data]
	data = np.loadtxt(path + file_name)
	if field == gain_ripple_field or field == nf_ripple_field:
		#consider ripple excursion only to avoid redundant information
		#because the max flat_gain is already given by the 'gain_flat' field in json
		data = data - data.mean()
	data = data.tolist()
	return data

def input_json(path, ampli_name):
	"""read the json input file and add all the 96 channels txt files
	create the output json file with output_json_file_name"""
	with open(path + input_json_file_name,'r') as edfa_json_file:
		amp_dict = edfa_json_file.read()
	amp_dict = json.loads(amp_dict)

	amp_dict[ampli_name][nf_fit_field] = read_file(path,nf_fit_file_name,nf_fit_field)
	amp_dict[ampli_name][nf_ripple_field] = read_file(path,nf_ripple_file_name,nf_ripple_field)
	amp_dict[ampli_name][gain_ripple_field] = read_file(path,gain_ripple_file_name,gain_ripple_field)
	amp_dict[ampli_name][dgt_field] = read_file(path,dgt_file_name,dgt_field)

	amp_dict = json.dumps(amp_dict, indent=4)
	with  open(path + 'newOA.json','w') as edfa_json_file:
		edfa_json_file.write(amp_dict)
	print(amp_dict)

if __name__ == '__main__':
	input_json('', 'OA_type1')
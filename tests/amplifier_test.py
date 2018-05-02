#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

from gnpy.core.elements import Edfa
import numpy as np
from json import load
import pytest
from gnpy.core import network_from_json
from gnpy.core.elements import Transceiver, Fiber, Edfa
from gnpy.core.utils import lin2db, db2lin , load_json
from gnpy.core.info import SpectralInformation, Channel, Power
from gnpy.core.equipment import read_eqpt_library
from examples.convert import convert_file
from pathlib import Path
import filecmp 

#network_file_name = 'tests/test_network.json'
network_file_name = 'tests/test_network.json'
eqpt_library_name = 'tests/eqpt_config_test.json'

@pytest.fixture(params=[(96, 0.05e12), (60, 0.075e12), (45, 0.1e12), (2, 0.1e12)], 
    ids=['50GHz spacing', '75GHz spacing', '100GHz spacing', '2 channels'])
# TODO in elements.py code: pytests doesn't pass with 1 channel: interpolate fail
def nch_and_spacing(request):
    """parametrize channel count vs channel spacing (Hz)"""
    yield request.param

@pytest.fixture()
def bw():
    """parametrize signal bandwidth (Hz)"""
    return 45e9

@pytest.fixture()
def setup_edfa():
    """init edfa class by reading test_network.json file
    remove all gain and nf ripple"""
    # eqpt_library = pytest_eqpt_library() 
    read_eqpt_library(eqpt_library_name)
    with open(network_file_name) as network_file:
        network_json = load(network_file)
    network = network_from_json(network_json)
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][0]

    #edfa.params.dgt = np.zeros(96)
    edfa.params.gain_ripple = np.zeros(96)
    edfa.params.nf_ripple = np.zeros(96)
    yield edfa

@pytest.fixture()
def setup_trx():
    """init transceiver class to access snr and osnr calculations"""
    with open(network_file_name) as network_file:
        network_json = load(network_file)
    network = network_from_json(network_json)
    trx = [n for n in network.nodes() if isinstance(n, Transceiver)][0]
    return trx

@pytest.fixture()
def si(nch_and_spacing, bw):
    """parametrize a channel comb with nch, spacing and signal bw"""
    nch, spacing = nch_and_spacing
    si = SpectralInformation()
    si = si.update(carriers=tuple(Channel(f, 191.3e12+spacing*f, 
            bw, 0.15, Power(1e-6, 0, 0)) for f in range(1,nch+1)))
    return si

@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("gain, nf_expected", [(10, 15), (15, 10), (25, 5.8)])              
def test_nf_calc(gain, nf_expected, enabled, setup_edfa, si):
    """ compare the 2 amplifier models (polynomial and estimated from nf_min and max)
     => nf_model vs nf_poly_fit for boundary gain values: gain_min (and below) & gain_flatmax
    same values are expected between the 2 models
    => unitary test for Edfa._calc_nf() (and Edfa.interpol_params)"""
    # eqpt_lib()
    edfa = setup_edfa
    frequencies = np.array([c.frequency for c in si.carriers])
    pin = np.array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    baud_rates = np.array([c.baud_rate for c in si.carriers])
    edfa.operational.gain_target = gain
    edfa.params.nf_model_enabled = enabled
    edfa.interpol_params(frequencies, pin, baud_rates)

    nf = edfa.nf
    print(nf)
    dif = abs(nf[0] - nf_expected)
    assert dif < 0.01

@pytest.mark.parametrize("gain", [17, 19, 21, 23])
def test_compare_nf_models(gain, setup_edfa, si):
    """ compare the 2 amplifier models (polynomial and estimated from nf_min and max)
     => nf_model vs nf_poly_fit for intermediate gain values:
     between gain_min and gain_flatmax some discrepancy is expected but target < 0.5dB
     => unitary test for Edfa._calc_nf (and Edfa.interpol_params)"""
    edfa = setup_edfa
    frequencies = np.array([c.frequency for c in si.carriers])
    pin = np.array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    baud_rates = np.array([c.baud_rate for c in si.carriers])
    edfa.operational.gain_target = gain
    edfa.params.nf_model_enabled = True
    edfa.interpol_params(frequencies, pin, baud_rates)
    nf_model = edfa.nf[0]

    edfa.params.nf_model_enabled = False
    edfa.interpol_params(frequencies, pin, baud_rates)
    nf_poly = edfa.nf[0]
    dif = abs(nf_model - nf_poly)
    assert dif < 0.5

def test_si(si, nch_and_spacing):
    """basic total power check of the channel comb generation"""
    nch = nch_and_spacing[0]
    pin = np.array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    p_tot = np.sum(pin)
    expected_p_tot = si.carriers[0].power.signal * nch
    dif = abs(lin2db(p_tot/expected_p_tot))
    assert dif < 0.01

@pytest.mark.parametrize("gain", [13, 15, 17, 19, 21, 23, 25, 27])
def test_ase_noise(gain, si, setup_edfa, setup_trx, bw):
    """testing 3 different ways of calculating osnr:
    1-pin-edfa.nf+58 vs 
    2-pout/pase afet propagate
    3-Transceiver osnr_ase_01nm
    => unitary test for Edfa.noise_profile (Edfa.interpol_params, Edfa.propagate)"""
    edfa = setup_edfa
    frequencies = np.array([c.frequency for c in si.carriers])
    pin = np.array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    baud_rates = np.array([c.baud_rate for c in si.carriers])
    edfa.operational.gain_target = gain
    edfa.params.nf_model_enabled = False
    edfa.interpol_params(frequencies, pin, baud_rates)
    nf = edfa.nf
    pin = lin2db(pin[0]*1e3)
    osnr_expected = pin - nf[0] + 58

    si = edfa(si)
    pout = np.array([c.power.signal for c in si.carriers])
    pase = np.array([c.power.ase for c in si.carriers])
    osnr = lin2db(pout[0] / pase[0]) - lin2db(12.5e9/bw)
    dif = abs(osnr - osnr_expected)

    trx = setup_trx
    si = trx(si)
    osnr = trx.osnr_ase_01nm[0]
    dif = dif + abs(osnr - osnr_expected)

    assert dif < 0.01


# adding tests to check the parser non regression
# convention of naming of test files:
#    - excelTest... .xls for the xls undertest
#    - test...  .json for the reference output
excel_filename = ['tests/excelTestFileFusedandEqt.xls',
 'tests/excelTestFileFused.xls',
 'tests/excelTestFilenoILA.xls',
 'tests/excelTestFileallILA.xls',
 'tests/excelTestFileallILAandEqt.xls',
 'tests/excelTestFileCORONET_Global_Topology.xls']
@pytest.mark.parametrize("inputfile",excel_filename)
def test_excel_with_fuse(inputfile) :
     convert_file(Path(inputfile)) 
     json_filename = f'{inputfile[:-3]}json'
     test_filename = f'tests/t{json_filename[12:]}'
     print(json_filename)
     print(test_filename)
     
     assert filecmp.cmp(json_filename,test_filename) is True

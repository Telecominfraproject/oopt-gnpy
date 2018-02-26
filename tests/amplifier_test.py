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
from gnpy.core.utils import lin2db, db2lin
from gnpy.core.info import SpectralInformation, Channel, Power

network_file_name = 'test_network.json'

@pytest.fixture()
def nch():
    return 96

@pytest.fixture()
def bw():
    return 45e9

@pytest.fixture()
def setup_edfa():
    with open(network_file_name) as network_file:
        network_json = load(network_file)
    network = network_from_json(network_json)
    for n in network.nodes():
        if isinstance(n, Edfa):
            edfa = n

    #edfa.params.dgt = np.zeros(96)
    edfa.params.gain_ripple = np.zeros(96)
    edfa.params.nf_ripple = np.zeros(96)

    yield edfa

@pytest.fixture()
def si_comb(nch, bw):
    spacing = 0.05 #THz
    si = SpectralInformation()
    si = si.update(carriers=tuple(Channel(f+1, (191.3+spacing*(f+1))*1e12, 
            bw, 0.15, Power(1e-6, 0, 0)) for f in range(nch)))
    pin = np.array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    frequencies = np.array([c.frequency for c in si.carriers])
    yield (frequencies, pin, si)

@pytest.mark.parametrize("gain, nf_expected, enabled", 
        [(10,15, True), (15, 10, True), (25, 5.8, True),
                (10,15, False), (15, 10, False), (25, 5.8, False)])
def test_nf_calc(gain, nf_expected, enabled, setup_edfa, si_comb):
    """check edfa._nf_calc boundary values @ min & max gain for both models"""
    edfa = setup_edfa
    frequencies, pin = si_comb[:2]
    edfa.operational.gain_target = gain
    edfa.params.nf_model_enabled = enabled
    edfa.interpol_params(frequencies, pin)

    nf = edfa.nf
    dif = abs(nf[0] - nf_expected)
    assert dif < 0.01

@pytest.mark.parametrize("gain", [17, 19, 21, 23])
def test_compare_nf_models(gain, setup_edfa, si_comb):
    """ compare nf_model vs nf_poly_fit for intermediate gain values"""
    edfa = setup_edfa
    frequencies, pin = si_comb[:2]
    edfa.operational.gain_target = gain
    edfa.params.nf_model_enabled = True
    edfa.interpol_params(frequencies, pin)
    nf_model = edfa.nf[0]

    edfa.params.nf_model_enabled = False
    edfa.interpol_params(frequencies, pin)
    nf_poly = edfa.nf[0]
    dif = abs(nf_model - nf_poly)
    assert dif < 0.5

def test_si(si_comb, nch):
    pin = si_comb[1]
    si = si_comb[2]
    p_tot = np.sum(pin)
    expected_p_tot = si.carriers[0].power.signal * nch
    dif = abs(p_tot - expected_p_tot)
    assert dif < 0.01

@pytest.mark.parametrize("gain", [15])#, 15, 17, 19, 21, 23, 25, 27])
def test_ase_noise(gain, si_comb, setup_edfa, bw):
    edfa = setup_edfa
    frequencies, pin = si_comb[:2]
    si = si_comb[2]
    edfa.operational.gain_target = gain
    edfa.params.nf_model_enabled = False
    edfa.interpol_params(frequencies, pin)
    nf = edfa.nf
    pin = lin2db(pin[0]*1e3)
    osnr_expected = pin - nf[0] + 58

    si = edfa(si)
    pout = np.array([c.power.signal for c in si.carriers])
    pase = np.array([c.power.ase for c in si.carriers])
    osnr = lin2db(pout[0] / pase[0]) - lin2db(12.5e9/bw)
    dif = abs(osnr - osnr_expected)

    assert dif < 0.01
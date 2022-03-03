#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

from numpy import zeros, array
from numpy.testing import assert_allclose
from gnpy.core.elements import Transceiver, Edfa, Fiber
from gnpy.core.utils import automatic_fmax, lin2db, db2lin, merge_amplifier_restrictions, dbm2watt, watt2dbm, lin2db
from gnpy.core.info import create_input_spectral_information, Pref, create_arbitrary_spectral_information
from gnpy.core.network import build_network
from gnpy.tools.json_io import load_network, load_equipment, network_from_json
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
test_network = DATA_DIR / 'test_network.json'
eqpt_library = DATA_DIR / 'eqpt_config.json'

# TODO in elements.py code: pytests doesn't pass with 1 channel: interpolate fail


@pytest.fixture(
    params=[(96, 0.05e12), (60, 0.075e12), (45, 0.1e12), (2, 0.1e12)],
    ids=['50GHz spacing', '75GHz spacing', '100GHz spacing', '2 channels'])
def nch_and_spacing(request):
    """parametrize channel count vs channel spacing (Hz)"""
    yield request.param


@pytest.fixture()
def bw():
    """parametrize signal bandwidth (Hz)"""
    return 45e9


@pytest.fixture()
def setup_edfa_variable_gain():
    """init edfa class by reading test_network.json file
    remove all gain and nf ripple"""
    equipment = load_equipment(eqpt_library)
    network = load_network(test_network, equipment)
    build_network(network, equipment, 0, 20)
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][0]
    edfa.gain_ripple = zeros(96)
    edfa.interpol_nf_ripple = zeros(96)
    yield edfa


@pytest.fixture()
def setup_edfa_fixed_gain():
    """init edfa class by reading the 2nd edfa in test_network.json file"""
    equipment = load_equipment(eqpt_library)
    network = load_network(test_network, equipment)
    build_network(network, equipment, 0, 20)
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][1]
    yield edfa


@pytest.fixture()
def setup_trx():
    """init transceiver class to access snr and osnr calculations"""
    equipment = load_equipment(eqpt_library)
    network = load_network(test_network, equipment)
    build_network(network, equipment, 0, 20)
    trx = [n for n in network.nodes() if isinstance(n, Transceiver)][0]
    return trx


@pytest.fixture()
def si(nch_and_spacing, bw):
    """parametrize a channel comb with nb_channel, spacing and signal bw"""
    nb_channel, spacing = nch_and_spacing
    f_min = 191.3e12
    f_max = automatic_fmax(f_min, spacing, nb_channel)
    return create_input_spectral_information(f_min, f_max, 0.15, bw, 1e-3, spacing)


@pytest.mark.parametrize("gain, nf_expected", [(10, 15), (15, 10), (25, 5.8)])
def test_variable_gain_nf(gain, nf_expected, setup_edfa_variable_gain, si):
    """=> unitary test for variable gain model Edfa._calc_nf() (and Edfa.interpol_params)"""
    edfa = setup_edfa_variable_gain
    si.signal /= db2lin(gain)
    si.nli /= db2lin(gain)
    si.ase /= db2lin(gain)
    edfa.operational.gain_target = gain
    si.pref = si.pref._replace(p_span0=0, p_spani=-gain)
    edfa.interpol_params(si)
    result = edfa.nf
    assert pytest.approx(nf_expected, abs=0.01) == result[0]


@pytest.mark.parametrize("gain, nf_expected", [(15, 10), (20, 5), (25, 5)])
def test_fixed_gain_nf(gain, nf_expected, setup_edfa_fixed_gain, si):
    """=> unitary test for fixed gain model Edfa._calc_nf() (and Edfa.interpol_params)"""
    edfa = setup_edfa_fixed_gain
    si.signal /= db2lin(gain)
    si.nli /= db2lin(gain)
    si.ase /= db2lin(gain)
    edfa.operational.gain_target = gain
    si.pref = si.pref._replace(p_span0=0, p_spani=-gain)
    edfa.interpol_params(si)
    assert pytest.approx(nf_expected, abs=0.01) == edfa.nf[0]


def test_si(si, nch_and_spacing):
    """basic total power check of the channel comb generation"""
    nb_channel = nch_and_spacing[0]
    p_tot = sum(si.signal + si.ase + si.nli)
    expected_p_tot = si.signal[0] * nb_channel
    assert pytest.approx(expected_p_tot, abs=0.01) == p_tot


@pytest.mark.parametrize("gain", [17, 19, 21, 23])
def test_compare_nf_models(gain, setup_edfa_variable_gain, si):
    """ compare the 2 amplifier models (polynomial and estimated from nf_min and max)
     => nf_model vs nf_poly_fit for intermediate gain values:
     between gain_min and gain_flatmax some discrepancy is expected but target < 0.5dB
     => unitary test for Edfa._calc_nf (and Edfa.interpol_params)"""
    edfa = setup_edfa_variable_gain
    si.signal /= db2lin(gain)
    si.nli /= db2lin(gain)
    si.ase /= db2lin(gain)
    edfa.operational.gain_target = gain
    # edfa is variable gain type
    si.pref = si.pref._replace(p_span0=0, p_spani=-gain)
    edfa.interpol_params(si)
    nf_model = edfa.nf[0]

    # change edfa type variety to a polynomial
    el_config = {
        "uid": "Edfa1",
        "operational": {
            "gain_target": gain,
            "tilt_target": 0
        },
        "metadata": {
            "location": {
                "region": "",
                "latitude": 2,
                "longitude": 0
            }
        }
    }
    equipment = load_equipment(eqpt_library)
    extra_params = equipment['Edfa']['CienaDB_medium_gain']
    temp = el_config.setdefault('params', {})
    temp = merge_amplifier_restrictions(temp, extra_params.__dict__)
    el_config['params'] = temp
    edfa = Edfa(**el_config)

    # edfa is variable gain type
    edfa.interpol_params(si)
    nf_poly = edfa.nf[0]
    print(nf_poly, nf_model)
    assert pytest.approx(nf_model, abs=0.5) == nf_poly


@pytest.mark.parametrize("gain", [13, 15, 17, 19, 21, 23, 25, 27])
def test_ase_noise(gain, si, setup_trx, bw):
    """testing 3 different ways of calculating osnr:
    1-pin-edfa.nf+58 vs
    2-pout/pase afet propagate
    3-Transceiver osnr_ase_01nm
    => unitary test for Edfa.noise_profile (Edfa.interpol_params, Edfa.propagate)"""
    equipment = load_equipment(eqpt_library)
    network = load_network(test_network, equipment)
    edfa = next(n for n in network.nodes() if n.uid == 'Edfa1')
    span = next(n for n in network.nodes() if n.uid == 'Span1')
    # update span1 and Edfa1 according to new gain before building network
    # updating span 1  avoids to overload amp
    span.params.length = gain * 1e3 / 0.2
    edfa.operational.gain_target = gain
    build_network(network, equipment, 0, 20)
    edfa.gain_ripple = zeros(96)
    edfa.interpol_nf_ripple = zeros(96)
    # propagate in span1 to have si with the correct power level
    si = span(si)
    print(span)

    si.pref = si.pref._replace(p_span0=0, p_spani=-gain)
    edfa.interpol_params(si)
    nf = edfa.nf
    print('nf', nf)
    pin = lin2db((si.signal[0] + si.ase[0] + si.nli[0]) * 1e3)
    osnr_expected = pin - nf[0] + 58

    si = edfa(si)
    print(edfa)
    osnr = lin2db(si.signal[0] / si.ase[0]) - lin2db(12.5e9 / bw)
    assert pytest.approx(osnr_expected, abs=0.01) == osnr

    trx = setup_trx
    si = trx(si)
    osnr = trx.osnr_ase_01nm[0]
    assert pytest.approx(osnr_expected, abs=0.01) == osnr


@pytest.mark.parametrize('delta_p', [0, None, 2])
@pytest.mark.parametrize('tilt_target',[0, -4])
def test_amp_behaviour(tilt_target, delta_p):
    """Check that amp correctly applies saturation, when there is tilt
    """
    json_data = {
        "elements": [{
            "uid": "Edfa1",
            "type": "Edfa",
            "type_variety": "test",
            "operational": {
                "delta_p": delta_p,
                "gain_target": 20,
                "tilt_target": tilt_target,
                "out_voa":0
            }
        },
        {
            "uid": "Span1",
            "type": "Fiber",
            "type_variety": "SSMF",
            "params": {
                "length": 100,
                "loss_coef": 0.2,
                "length_units": "km"
                }
        }],
        "connections": []
    }
    equipment = load_equipment(eqpt_library)
    network = network_from_json(json_data, equipment)
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][0]
    fiber = [n for n in network.nodes() if isinstance(n, Fiber)][0]
    fiber.params.con_in = 0
    fiber.params.con_out = 0
    si = create_input_spectral_information(f_min=191.3e12, f_max=196.6e12, roll_off=0.15, baud_rate=64e9, power=0.001,
                                           spacing=75e9)
    si = fiber(si)
    total_sig_powerin = sum(si.signal)
    sig_in = lin2db(si.signal)
    si = edfa(si)
    sig_out = lin2db(si.signal)
    total_sig_powerout = sum(si.signal)
    gain = lin2db(total_sig_powerout / total_sig_powerin)
    expected_total_power_out = total_sig_powerin * 100 * db2lin(delta_p) if delta_p else total_sig_powerin * 100
    assert pytest.approx(total_sig_powerout, abs=1e-7) == min(expected_total_power_out, dbm2watt(21))
    assert pytest.approx(edfa.effective_gain, 1e-5) == gain
    assert watt2dbm(sum(si.signal + si.nli + si.ase))<= 21.01
    # If there is no tilt on the amp: the gain is identical for all carriers
    if tilt_target == 0:
        assert_allclose(sig_in + gain, sig_out, rtol=1e-13)
    else:
        if delta_p != 2:
            expected_sig_out = [
                -33.02944203, -32.98844836, -32.9382214 , -32.8735441 , -32.77769277,
                -32.6509986 , -32.51152453, -32.35201942, -32.17248813, -31.98419556,
                -31.77414438, -31.55543604, -31.38524694, -31.26345832, -31.15366673,
                -31.04380006, -30.94329483, -30.85224768, -30.7682578 , -30.70302965,
                -30.65670757, -30.61319786, -30.57653698, -30.54261041, -30.50865683,
                -30.47467993, -30.44085951, -30.40776241, -30.37538948, -30.34299826,
                -30.30259954, -30.23011898, -30.1253906 , -30.03268964, -29.97006308,
                -29.91933991, -29.87541273, -29.83834469, -29.80380293, -29.77236071,
                -29.74401049, -29.71564068, -29.68718095, -29.65844759, -29.62943729,
                -29.58881371, -29.51899658, -29.43741434, -29.36728994, -29.30861435,
                -29.25479539, -29.20893223, -29.17097705, -29.13299552, -29.09033285,
                -29.02902455, -28.94901196, -28.86919003, -28.78997253, -28.71097166,
                -28.63507244, -28.56229669, -28.49026933, -28.41823779, -28.41823779,
                -28.41823779, -28.41823779, -28.41823779, -28.41823779, -28.41823779]
        else:
            expected_sig_out = [
                -31.02944203, -30.98844836, -30.9382214 , -30.8735441 , -30.77769277,
                -30.6509986 , -30.51152453, -30.35201942, -30.17248813, -29.98419556,
                -29.77414438, -29.55543604, -29.38524694, -29.26345832, -29.15366673,
                -29.04380006, -28.94329483, -28.85224768, -28.7682578 , -28.70302965,
                -28.65670757, -28.61319786, -28.57653698, -28.54261041, -28.50865683,
                -28.47467993, -28.44085951, -28.40776241, -28.37538948, -28.34299826,
                -28.30259954, -28.23011898, -28.1253906 , -28.03268964, -27.97006308,
                -27.91933991, -27.87541273, -27.83834469, -27.80380293, -27.77236071,
                -27.74401049, -27.71564068, -27.68718095, -27.65844759, -27.62943729,
                -27.58881371, -27.51899658, -27.43741434, -27.36728994, -27.30861435,
                -27.25479539, -27.20893223, -27.17097705, -27.13299552, -27.09033285,
                -27.02902455, -26.94901196, -26.86919003, -26.78997253, -26.71097166,
                -26.63507244, -26.56229669, -26.49026933, -26.41823779, -26.41823779,
                -26.41823779, -26.41823779, -26.41823779, -26.41823779, -26.41823779]
        assert_allclose(sig_out, expected_sig_out, rtol=1e-9)

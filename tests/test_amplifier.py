#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

from numpy import zeros, array
from numpy.testing import assert_allclose
from gnpy.core.elements import Transceiver, Edfa, Fiber
from gnpy.core.utils import automatic_fmax, lin2db, db2lin, merge_amplifier_restrictions, dbm2watt, watt2dbm
from gnpy.core.info import create_input_spectral_information, create_arbitrary_spectral_information
from gnpy.core.network import build_network, set_amplifier_voa
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
    return create_input_spectral_information(f_min=f_min, f_max=f_max, roll_off=0.15, baud_rate=bw,
                                             spacing=spacing, tx_osnr=40.0, tx_power=1e-3)


@pytest.mark.parametrize("gain, nf_expected", [(10, 15), (15, 10), (25, 5.8)])
def test_variable_gain_nf(gain, nf_expected, setup_edfa_variable_gain, si):
    """=> unitary test for variable gain model Edfa._calc_nf() (and Edfa.interpol_params)"""
    edfa = setup_edfa_variable_gain
    si.signal /= db2lin(gain)
    si.nli /= db2lin(gain)
    si.ase /= db2lin(gain)
    edfa.operational.gain_target = gain
    edfa.effective_gain = gain
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
    edfa.effective_gain = gain
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
    """compare the 2 amplifier models (polynomial and estimated from nf_min and max)
     => nf_model vs nf_poly_fit for intermediate gain values:
     between gain_min and gain_flatmax some discrepancy is expected but target < 0.5dB
     => unitary test for Edfa._calc_nf (and Edfa.interpol_params)"""
    edfa = setup_edfa_variable_gain
    si.signal /= db2lin(gain)
    si.nli /= db2lin(gain)
    si.ase /= db2lin(gain)
    edfa.operational.gain_target = gain
    edfa.effective_gain = gain
    # edfa is variable gain type
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
@pytest.mark.parametrize('tilt_target', [0, -4])
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
                "gain_target": 20 + delta_p if delta_p else 20,
                "tilt_target": tilt_target,
                "out_voa": 0
            }
        }, {
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
    fiber.ref_pch_in_dbm = 0.0
    si = create_input_spectral_information(f_min=191.3e12, f_max=196.05e12, roll_off=0.15, baud_rate=64e9,
                                           spacing=75e9, tx_osnr=None, tx_power=1e-3)
    si = fiber(si)
    total_sig_powerin = sum(si.signal)
    sig_in = lin2db(si.signal)
    si = edfa(si)
    sig_out = lin2db(si.signal)
    total_sig_powerout = sum(si.signal)
    gain = lin2db(total_sig_powerout / total_sig_powerin)
    expected_total_power_out = total_sig_powerin * 100 * db2lin(delta_p) if delta_p else total_sig_powerin * 100
    assert pytest.approx(total_sig_powerout, abs=1e-6) == min(expected_total_power_out, dbm2watt(21))
    assert pytest.approx(edfa.effective_gain, 1e-5) == gain
    assert watt2dbm(sum(si.signal + si.nli + si.ase)) <= 21.01
    # If there is no tilt on the amp: the gain is identical for all carriers
    if tilt_target == 0:
        assert_allclose(sig_in + gain, sig_out, rtol=1e-13)
    else:
        if delta_p != 2:
            expected_sig_out = [
                -32.00529182, -31.93540907, -31.86554231, -31.79417979, -31.71903263,
                -31.6424009, -31.56531159, -31.48775435, -31.41468382, -31.35973323,
                -31.32286555, -31.28602346, -31.2472908, -31.20086569, -31.14671746,
                -31.08702653, -31.01341963, -30.93430243, -30.87791656, -30.84413339,
                -30.81605918, -30.78824936, -30.76071036, -30.73319161, -30.70494101,
                -30.67368479, -30.63941012, -30.60178381, -30.55585766, -30.5066561,
                -30.43426575, -30.33848379, -30.24471112, -30.18220815, -30.15076699,
                -30.11934744, -30.08776718, -30.05548097, -30.02250068, -29.98954302,
                -29.95661362, -29.92370274, -29.8854762, -29.84193785, -29.79238328,
                -29.72452662, -29.6385071, -29.54788144, -29.44581202, -29.33924103,
                -29.23276107, -29.10289365, -28.91425473, -28.70204648, -28.50670713,
                -28.3282514, -28.15895225, -28.009065, -27.87864672, -27.76315964,
                -27.68523133, -27.62260405, -27.58076622]

        else:
            expected_sig_out = [
                -30.00529182, -29.93540907, -29.86554231, -29.79417979, -29.71903263,
                -29.6424009, -29.56531159, -29.48775435, -29.41468382, -29.35973323,
                -29.32286555, -29.28602346, -29.2472908, -29.20086569, -29.14671746,
                -29.08702653, -29.01341963, -28.93430243, -28.87791656, -28.84413339,
                -28.81605918, -28.78824936, -28.76071036, -28.73319161, -28.70494101,
                -28.67368479, -28.63941012, -28.60178381, -28.55585766, -28.5066561,
                -28.43426575, -28.33848379, -28.24471112, -28.18220815, -28.15076699,
                -28.11934744, -28.08776718, -28.05548097, -28.02250068, -27.98954302,
                -27.95661362, -27.92370274, -27.8854762, -27.84193785, -27.79238328,
                -27.72452662, -27.6385071, -27.54788144, -27.44581202, -27.33924103,
                -27.23276107, -27.10289365, -26.91425473, -26.70204648, -26.50670713,
                -26.3282514, -26.15895225, -26.009065, -25.87864672, -25.76315964,
                -25.68523133, -25.62260405, -25.58076622]

        print(sig_out)
        assert_allclose(sig_out, expected_sig_out, rtol=1e-9)


@pytest.mark.parametrize('delta_p', [0, None, 20])
@pytest.mark.parametrize('base_power', [0, 20])
@pytest.mark.parametrize('delta_pdb_per_channel',
                         [[0, 1, 3, 0.5, -2],
                          [0, 0, 0, 0, 0],
                          [-2, -2, -2, -2, -2],
                          [0, 2, -2, -5, 4],
                          [0, 1, 3, 0.5, -2], ])
def test_amp_saturation(delta_pdb_per_channel, base_power, delta_p):
    """Check that amp correctly applies saturation
    """
    json_data = {
        "elements": [{
            "uid": "Edfa1",
            "type": "Edfa",
            "type_variety": "test",
            "operational": {
                "delta_p": delta_p,
                "gain_target": 20,
                "tilt_target": 0,
                "out_voa": 0
            }
        }],
        "connections": []
    }
    equipment = load_equipment(eqpt_library)
    network = network_from_json(json_data, equipment)
    edfa = [n for n in network.nodes()][0]
    frequency = 193e12 + array([0, 50e9, 150e9, 225e9, 275e9])
    slot_width = array([37.5e9, 50e9, 75e9, 50e9, 37.5e9])
    baud_rate = array([32e9, 42e9, 64e9, 42e9, 32e9])
    signal = dbm2watt(array([-20.0, -18.0, -22.0, -25.0, -16.0]) + array(delta_pdb_per_channel) + base_power)
    si = create_arbitrary_spectral_information(frequency=frequency, slot_width=slot_width,
                                               signal=signal, baud_rate=baud_rate, roll_off=0.15,
                                               delta_pdb_per_channel=delta_pdb_per_channel,
                                               tx_osnr=None, tx_power=None)
    total_sig_powerin = sum(si.signal)
    sig_in = lin2db(si.signal)
    si = edfa(si)
    sig_out = lin2db(si.signal)
    total_sig_powerout = sum(si.signal)
    gain = lin2db(total_sig_powerout / total_sig_powerin)
    assert watt2dbm(sum(si.signal + si.nli + si.ase)) <= 21.02
    assert pytest.approx(edfa.effective_gain, 1e-13) == gain
    assert_allclose(sig_in + gain, sig_out, rtol=1e-13)


def test_set_out_voa():
    """Check that out_voa is correctly set if out_voa_auto is true
    gain is maximized to obtain better NF:
    if optimum input power in next span is -3 + pref_ch_db then total power at optimum is 19 -3 = 16dBm.
    since amp has 21 dBm p_max, power out of amp can be set to 21dBm increasing out_voa by 5 to keep
    same input power in the fiber. Since the optimisation contains a hard coded margin of 1 to account for
    possible degradation on max power, the expected voa value is 4, and delta_p and gain are corrected
    accordingly.
    """
    json_data = {
        "elements": [{
            "uid": "Edfa1",
            "type": "Edfa",
            "type_variety": "test",
            "operational": {
                "delta_p": -3,
                "gain_target": 20,
                "tilt_target": 0
            }
        }],
        "connections": []
    }
    equipment = load_equipment(eqpt_library)
    network = network_from_json(json_data, equipment)
    amp = [n for n in network.nodes()][0]
    print(amp.out_voa)
    power_target = 19 + amp.delta_p
    power_mode = True
    amp.params.out_voa_auto = True
    set_amplifier_voa(amp, power_target, power_mode)
    assert amp.out_voa == 4.0
    assert amp.effective_gain == 20.0 + 4.0
    assert amp.delta_p == -3.0 + 4.0

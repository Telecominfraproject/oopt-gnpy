#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_amplifier
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

from pathlib import Path
import pytest

from numpy import zeros, array
from numpy.testing import assert_allclose

from gnpy.core.elements import Transceiver, Edfa, Fiber
from gnpy.core.utils import automatic_fmax, lin2db, db2lin, merge_amplifier_restrictions, dbm2watt, watt2dbm
from gnpy.core.info import create_input_spectral_information, create_arbitrary_spectral_information
from gnpy.core.network import build_network, set_amplifier_voa
from gnpy.tools.json_io import load_network, load_equipment, load_json, _equipment_from_json, network_from_json
from gnpy.topology.request import PathRequest


TEST_DIR =  Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
test_network = DATA_DIR / 'test_network.json'
eqpt_library = DATA_DIR / 'eqpt_config.json'
extra_configs = {"std_medium_gain_advanced_config.json": load_json(DATA_DIR / "std_medium_gain_advanced_config.json")}

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


def pathrequest(pch_dbm, p_tot_dbm):
    """create ref channel for defined power settings
    """
    params = {
        "power": dbm2watt(pch_dbm),
        "tx_power": dbm2watt(pch_dbm),
        "nb_channel": round(dbm2watt(p_tot_dbm) / dbm2watt(pch_dbm), 0),
        'request_id': None,
        'trx_type': None,
        'trx_mode': None,
        'source': None,
        'destination': None,
        'bidir': False,
        'nodes_list': [],
        'loose_list': [],
        'format': '',
        'baud_rate': None,
        'bit_rate': None,
        'roll_off': None,
        'OSNR': None,
        'penalties': None,
        'path_bandwidth': None,
        'effective_freq_slot': None,
        'f_min': None,
        'f_max': None,
        'spacing': None,
        'min_spacing': None,
        'cost': None,
        'equalization_offset_db': None,
        'tx_osnr': None
    }
    return PathRequest(**params)


@pytest.fixture()
def setup_edfa_variable_gain():
    """init edfa class by reading test_network.json file
    remove all gain and nf ripple"""
    equipment = load_equipment(eqpt_library, extra_configs)
    network = load_network(test_network, equipment)
    build_network(network, equipment, pathrequest(0, 20))
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][0]
    edfa.gain_ripple = zeros(96)
    edfa.interpol_nf_ripple = zeros(96)
    yield edfa


@pytest.fixture()
def setup_edfa_fixed_gain():
    """init edfa class by reading the 2nd edfa in test_network.json file"""
    equipment = load_equipment(eqpt_library, extra_configs)
    network = load_network(test_network, equipment)
    build_network(network, equipment, pathrequest(0, 20))
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][1]
    yield edfa


@pytest.fixture()
def setup_trx():
    """init transceiver class to access snr and osnr calculations"""
    equipment = load_equipment(eqpt_library, extra_configs)
    network = load_network(test_network, equipment)
    build_network(network, equipment, pathrequest(0, 20))
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
    si.apply_attenuation_db(gain)
    edfa.operational.gain_target = gain
    edfa.effective_gain = gain
    edfa.interpol_params(si)
    result = edfa.nf
    assert pytest.approx(nf_expected, abs=0.01) == result[0]


@pytest.mark.parametrize("gain, nf_expected", [(15, 10), (20, 5), (25, 5)])
def test_fixed_gain_nf(gain, nf_expected, setup_edfa_fixed_gain, si):
    """=> unitary test for fixed gain model Edfa._calc_nf() (and Edfa.interpol_params)"""
    edfa = setup_edfa_fixed_gain
    si.apply_attenuation_db(gain)
    edfa.operational.gain_target = gain
    edfa.effective_gain = gain
    edfa.interpol_params(si)
    assert pytest.approx(nf_expected, abs=0.01) == edfa.nf[0]


def test_si(si, nch_and_spacing):
    """basic total power check of the channel comb generation"""
    nb_channel = nch_and_spacing[0]
    p_tot = si.ptot
    expected_p_tot = si.signal[0] * nb_channel
    assert pytest.approx(expected_p_tot, abs=0.01) == p_tot


@pytest.mark.parametrize("gain", [17, 19, 21, 23])
def test_compare_nf_models(gain, setup_edfa_variable_gain, si):
    """compare the 2 amplifier models (polynomial and estimated from nf_min and max)
     => nf_model vs nf_poly_fit for intermediate gain values:
     between gain_min and gain_flatmax some discrepancy is expected but target < 0.5dB
     => unitary test for Edfa._calc_nf (and Edfa.interpol_params)"""
    edfa = setup_edfa_variable_gain
    si.apply_attenuation_db(gain)
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
    equipment = load_equipment(eqpt_library, extra_configs)
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
    equipment = load_equipment(eqpt_library, extra_configs)
    network = load_network(test_network, equipment)
    edfa = next(n for n in network.nodes() if n.uid == 'Edfa1')
    span = next(n for n in network.nodes() if n.uid == 'Span1')
    # update span1 and Edfa1 according to new gain before building network
    # updating span 1  avoids to overload amp
    span.params.length = gain * 1e3 / 0.2
    edfa.operational.gain_target = gain
    build_network(network, equipment, pathrequest(0, 20))
    edfa.gain_ripple = zeros(96)
    edfa.interpol_nf_ripple = zeros(96)
    # propagate in span1 to have si with the correct power level
    si = span(si)
    print(span)

    edfa.interpol_params(si)
    nf = edfa.nf
    print('nf', nf)
    pin = watt2dbm(si.pch[0])
    osnr_expected = pin - nf[0] + 58

    si = edfa(si)
    print(edfa)
    osnr = si.opt_snr_lin_dB[0]
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
    equipment = load_equipment(eqpt_library, extra_configs)
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
    assert si.ptot_dBm <= 21.01
    # If there is no tilt on the amp: the gain is identical for all carriers
    if tilt_target == 0:
        assert_allclose(sig_in + gain, sig_out, rtol=1e-13)
    else:
        if delta_p != 2:
            expected_sig_out = [
                -31.951320304847002, -31.882898264775857, -31.813062398281428, -31.739708658234527,
                -31.6645399525307, -31.589001789953212, -31.512965690269446, -31.43902360915434,
                -31.38268517704092, -31.343905996502897, -31.307762398792974, -31.271187175929704,
                -31.22715700033586, -31.175631090302723, -31.11958037599187, -31.052743032434282,
                -30.975110524921607, -30.908123337738402, -30.867706108610072, -30.84009364314535,
                -30.812708588436696, -30.785597929665546, -30.758591062589833, -30.731604362397064,
                -30.70246760040157, -30.6700304989031, -30.635870195751718, -30.59524027792815,
                -30.548189272676865, -30.493409361924293, -30.41566991033016, -30.315957690395308,
                -30.231453503525945, -30.18410898446455, -30.153262814472527, -30.122439955002424,
                -30.091326363268397, -30.059417160325147, -30.02705824696185, -29.994723834602794,
                -29.962413467238807, -29.929611505438615, -29.891646559181293, -29.848514736567374,
                -29.798893094507243, -29.730892838087822, -29.646479185859654, -29.557403528975744,
                -29.45730998358558, -29.352726663466477, -29.24822822139738, -29.123082201246028,
                -28.94403947547321, -28.735799275668164, -28.540875159588545, -28.36387030774887,
                -28.19515467702171, -28.045291145748983, -27.914303447586164, -27.795804541476127,
                -27.70793118016708, -27.64630743528702, -27.59919239792548]
        else:
            expected_sig_out = [
                -29.951320304847002, -29.882898264775854, -29.813062398281428, -29.739708658234527,
                -29.664539952530703, -29.589001789953215, -29.512965690269443, -29.439023609154344,
                -29.38268517704092, -29.343905996502897, -29.307762398792974, -29.2711871759297,
                -29.227157000335858, -29.175631090302726, -29.119580375991873, -29.05274303243428,
                -28.975110524921604, -28.908123337738395, -28.86770610861007, -28.84009364314535,
                -28.8127085884367, -28.78559792966555, -28.75859106258983, -28.73160436239706,
                -28.70246760040157, -28.670030498903103, -28.635870195751714, -28.595240277928152,
                -28.548189272676865, -28.493409361924297, -28.41566991033016, -28.315957690395308,
                -28.231453503525948, -28.184108984464554, -28.153262814472523, -28.122439955002424,
                -28.0913263632684, -28.059417160325147, -28.027058246961847, -27.994723834602794,
                -27.962413467238804, -27.92961150543862, -27.89164655918129, -27.84851473656737,
                -27.79889309450724, -27.73089283808782, -27.64647918585965, -27.557403528975748,
                -27.45730998358558, -27.352726663466473, -27.248228221397383, -27.12308220124603,
                -26.94403947547321, -26.735799275668164, -26.540875159588538, -26.36387030774887,
                -26.195154677021705, -26.045291145748983, -25.914303447586164, -25.795804541476123,
                -25.70793118016708, -25.64630743528702, -25.599192397925478]
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
    equipment = load_equipment(eqpt_library, extra_configs)
    network = network_from_json(json_data, equipment)
    edfa = [n for n in network.nodes()][0]
    frequency = 193e12 + array([0, 50e9, 150e9, 225e9, 275e9])
    slot_width = array([37.5e9, 50e9, 75e9, 50e9, 37.5e9])
    baud_rate = array([32e9, 42e9, 64e9, 42e9, 32e9])
    pch = dbm2watt(array([-20.0, -18.0, -22.0, -25.0, -16.0]) + array(delta_pdb_per_channel) + base_power)
    si = create_arbitrary_spectral_information(frequency=frequency, slot_width=slot_width,
                                               pch=pch, baud_rate=baud_rate, roll_off=0.15,
                                               delta_pdb_per_channel=delta_pdb_per_channel,
                                               tx_osnr=None, tx_power=None)
    total_sig_powerin = sum(si.signal)
    sig_in = lin2db(si.signal)
    si = edfa(si)
    sig_out = lin2db(si.signal)
    total_sig_powerout = sum(si.signal)
    gain = lin2db(total_sig_powerout / total_sig_powerin)
    assert si.ptot_dBm <= 21.02
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
    equipment = load_equipment(eqpt_library, extra_configs)
    network = network_from_json(json_data, equipment)
    amp = [n for n in network.nodes()][0]
    print(amp.out_voa)
    power_target = 19 + amp.delta_p
    power_mode = True
    amp.params.out_voa_auto = True
    set_amplifier_voa(amp, power_target, power_mode,
                      voa_margin=equipment['Span']['default'].voa_margin, voa_step=equipment['Span']['default'].voa_step)
    assert amp.out_voa == 4.0
    assert amp.effective_gain == 20.0 + 4.0
    assert amp.delta_p == -3.0 + 4.0


def test_multiband():

    equipment_json = load_json(eqpt_library)
    # add some multiband amplifiers
    amps = [
        {
            "type_variety": "std_medium_gain_C",
            "f_min": 191.25e12,
            "f_max": 196.15e12,
            "type_def": "variable_gain",
            "gain_flatmax": 26,
            "gain_min": 15,
            "p_max": 21,
            "nf_min": 6,
            "nf_max": 10,
            "out_voa_auto": False,
            "allowed_for_design": True},
        {
            "type_variety": "std_medium_gain_L",
            "f_min": 186.55e12,
            "f_max": 190.05e12,
            "type_def": "variable_gain",
            "gain_flatmax": 26,
            "gain_min": 15,
            "p_max": 21,
            "nf_min": 6,
            "nf_max": 10,
            "out_voa_auto": False,
            "allowed_for_design": True},
        {
            "type_variety": "std_medium_gain_multiband",
            "type_def": "multi_band",
            "amplifiers": [
                "std_medium_gain_C",
                "std_medium_gain_L"
            ],
            "allowed_for_design": False
        }
    ]
    equipment_json['Edfa'].extend(amps)

    equipment = _equipment_from_json(equipment_json, extra_configs)

    el_config = {
        "uid": "Edfa1",
        "type": "Multiband_amplifier",
        "type_variety": "std_medium_gain_multiband",
        "amplifiers": [
            {
                "type_variety": "std_medium_gain_C",
                "operational": {
                    "gain_target": 22.55,
                    "delta_p": 0.9,
                    "out_voa": 3.0,
                    "tilt_target": 0.0,
                }
            },
            {
                "type_variety": "std_medium_gain_L",
                "operational": {
                    "gain_target": 21,
                    "delta_p": 3.0,
                    "out_voa": 3.0,
                    "tilt_target": 0.0,
                }
            }
        ]
    }
    fused_config = {
        "uid": "[83/WR-2-4-SIG=>930/WRT-1-2-SIG]-Tl/9300",
        "type": "Fused",
        "params": {
            "loss": 20
        }
    }
    json_data = {
        "elements": [
            el_config,
            fused_config
        ],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    amp = next(n for n in network.nodes() if n.uid == 'Edfa1')
    fused = next(n for n in network.nodes() if n.uid == '[83/WR-2-4-SIG=>930/WRT-1-2-SIG]-Tl/9300')
    si = create_input_spectral_information(f_min=186e12, f_max=196e12, roll_off=0.15, baud_rate=32e9, tx_power=1e-3,
                                           spacing=50e9, tx_osnr=40.0)
    assert si.number_of_channels == 200
    si = fused(si)
    si = amp(si)
    # assert nb of channel after mux/demux
    assert si.number_of_channels == 164    # computed based on amp bands
    # Check that multiband amp is correctly created with correct __str__
    actual_c_amp = amp.amplifiers["CBAND"].__str__()
    expected_c_amp = '\n'.join([
        'Edfa Edfa1',
        '  type_variety:           std_medium_gain_C',
        '  effective gain(dB):     21.22',
        '  (before att_in and before output VOA)',
        '  tilt-target(dB)         0.00',
        '  noise figure (dB):      6.32',
        '  (including att_in)',
        '  pad att_in (dB):        0.00',
        '  Power In (dBm):         -0.22',
        '  Power Out (dBm):        21.01',
        '  Delta_P (dB):           0.90',
        '  target pch (dBm):       None',
        '  actual pch out (dBm):   -1.77',
        '  output VOA (dB):        3.00'])
    assert actual_c_amp == expected_c_amp
    actual_l_amp = amp.amplifiers["LBAND"].__str__()
    expected_l_amp = '\n'.join([
        'Edfa Edfa1',
        '  type_variety:           std_medium_gain_L',
        '  effective gain(dB):     21.00',
        '  (before att_in and before output VOA)',
        '  tilt-target(dB)         0.00',
        '  noise figure (dB):      6.36',
        '  (including att_in)',
        '  pad att_in (dB):        0.00',
        '  Power In (dBm):         -1.61',
        '  Power Out (dBm):        19.40',
        '  Delta_P (dB):           3.00',
        '  target pch (dBm):       None',
        '  actual pch out (dBm):   -1.99',
        '  output VOA (dB):        3.00'])
    assert actual_l_amp == expected_l_amp

    # check that f_min, f_max of si are within amp band
    assert amp.amplifiers["LBAND"].params.f_min == 186.55e12
    assert si.frequency[0] >= amp.amplifiers["LBAND"].params.f_min
    assert amp.amplifiers["CBAND"].params.f_max == 196.15e12
    assert si.frequency[-1] <= amp.amplifiers["CBAND"].params.f_max
    for freq in si.frequency:
        if freq > 190.05e12:
            assert freq >= 191.25e12
        if freq < 191.25e12:
            assert freq <= 190.25e12


def test_user_defined_config():
    """Checks that a user defined config is correctly used instead of DEFAULT_EDFA_CONFIG
    """
    extra_configs['user_edfa_config.json'] = load_json(DATA_DIR / 'user_edfa_config.json')
    user_edfa = {
        "type_variety": "user_defined",
        "type_def": "variable_gain",
        "gain_flatmax": 25,
        "gain_min": 15,
        "p_max": 21,
        "nf_min": 6,
        "nf_max": 10,
        "default_config_from_json": "user_edfa_config.json",
        "out_voa_auto": False,
        "allowed_for_design": True
    }

    # add the reference to
    json_data = load_json(eqpt_library)
    json_data['Edfa'].append(user_edfa)
    equipment = _equipment_from_json(json_data, extra_configs)
    json_data = {
        "elements": [{
            "uid": "Edfa1",
            "type": "Edfa",
            "type_variety": "user_defined",
            "operational": {
                "delta_p": -3,
                "gain_target": 20,
                "tilt_target": 0,
                "out_voa": 0
            }
        }],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    amp = [n for n in network.nodes()][0]
    assert_allclose(amp.params.f_min, 193.0e12, rtol=1e-13)
    assert_allclose(amp.params.f_max, 195.0e12, rtol=1e-13)
    assert_allclose(amp.params.gain_ripple[15], 0.01027114740367, rtol=1e-13)
    assert_allclose(amp.params.nf_ripple[15], 0.0, rtol=1e-13)
    assert_allclose(amp.params.dgt[15], 1.847275503201129, rtol=1e-13)


def test_default_config():
    """Checks that a config using a file gives the exact same result as the default config if values are identical
    to DEFAULT_EDFA_CONFIG
    """
    extra_configs['copy_default_edfa_config.json'] = load_json(DATA_DIR / 'copy_default_edfa_config.json')
    user_edfa = {
        "type_variety": "user_defined",
        "type_def": "variable_gain",
        "gain_flatmax": 25,
        "gain_min": 15,
        "p_max": 21,
        "nf_min": 6,
        "nf_max": 10,
        "default_config_from_json": "copy_default_edfa_config.json",
        "out_voa_auto": False,
        "allowed_for_design": True
    }

    default_edfa = {
        "type_variety": "default",
        "type_def": "variable_gain",
        "gain_flatmax": 25,
        "gain_min": 15,
        "p_max": 21,
        "nf_min": 6,
        "nf_max": 10,
        "out_voa_auto": False,
        "allowed_for_design": True
    }

    # add the reference to
    json_data = load_json(eqpt_library)
    json_data['Edfa'].append(user_edfa)
    json_data['Edfa'].append(default_edfa)
    equipment = _equipment_from_json(json_data, extra_configs)
    json_data = {
        "elements": [{
            "uid": "Edfa1",
            "type": "Edfa",
            "type_variety": "user_defined",
            "operational": {
                "delta_p": -3,
                "gain_target": 20,
                "tilt_target": 0,
                "out_voa": 0
            }
        }, {
            "uid": "Edfa2",
            "type": "Edfa",
            "type_variety": "default",
            "operational": {
                "delta_p": -3,
                "gain_target": 20,
                "tilt_target": 0,
                "out_voa": 0
            }
        }],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    amp1, amp2 = [n for n in network.nodes()]
    assert_allclose(amp1.params.f_min, amp2.params.f_min, rtol=1e-13)
    assert_allclose(amp1.params.f_max, amp2.params.f_max, rtol=1e-13)
    assert_allclose(amp1.params.gain_ripple, amp2.params.gain_ripple, rtol=1e-13)
    assert_allclose(amp1.params.nf_ripple, amp2.params.nf_ripple, rtol=1e-13)
    assert_allclose(amp1.params.dgt, amp2.params.dgt, rtol=1e-13)


@pytest.mark.parametrize("file", [None, {"name": "copy_default_edfa_config.json",
                                         "path": DATA_DIR / "copy_default_edfa_config.json"}])
def test_frequency_range(file):
    """Checks that a frequency range is correctly read from the library and pre-empts DEFAULT_EDFA_CONFIG
    """
    user_edfa = {
        "type_variety": "user_defined",
        "type_def": "variable_gain",
        "f_min": 192.0e12,
        "f_max": 195.9e12,
        "gain_flatmax": 25,
        "gain_min": 15,
        "p_max": 21,
        "nf_min": 6,
        "nf_max": 10,
        "out_voa_auto": False,
        "allowed_for_design": True
    }
    if file:
        user_edfa["default_config_from_json"] = file['name']
        extra_configs[file['name']] = load_json(file['path'])
    # add the reference to
    json_data = load_json(eqpt_library)
    json_data['Edfa'].append(user_edfa)
    equipment = _equipment_from_json(json_data, extra_configs)
    json_data = {
        "elements": [{
            "uid": "Edfa1",
            "type": "Edfa",
            "type_variety": "user_defined",
            "operational": {
                "delta_p": -3,
                "gain_target": 20,
                "tilt_target": 0,
                "out_voa": 0
            }
        }],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    amp = [n for n in network.nodes()][0]
    si = create_input_spectral_information(f_min=191.3e12, f_max=196.05e12, roll_off=0.15, baud_rate=64e9,
                                           spacing=75e9, tx_osnr=None, tx_power=1e-5)
    si = amp(si)
    assert_allclose(amp.params.f_min, 192.0e12, rtol=1e-13)
    assert_allclose(amp.params.f_max, 195.9e12, rtol=1e-13)
    assert si.frequency[0] >= 192.0e12 + 75e9 / 2
    assert si.frequency[-1] <= 195.9e12 - 75e9 / 2

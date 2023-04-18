#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2019-05-22
"""
@author: esther.lerouzic
checks that new equalization option give the same output as old one:

"""

from pathlib import Path
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_raises
from numpy import array

from gnpy.core.utils import lin2db, automatic_nch, dbm2watt, power_dbm_to_psd_mw_ghz, watt2dbm, psd2powerdbm
from gnpy.core.network import build_network
from gnpy.core.elements import Roadm
from gnpy.core.info import create_input_spectral_information, Pref, create_arbitrary_spectral_information, \
    ReferenceCarrier
from gnpy.core.equipment import trx_mode_params
from gnpy.core.exceptions import ConfigurationError
from gnpy.tools.json_io import network_from_json, load_equipment, load_network, _spectrum_from_json, load_json
from gnpy.topology.request import PathRequest, compute_constrained_path, propagate


TEST_DIR = Path(__file__).parent
EQPT_FILENAME = TEST_DIR / 'data/eqpt_config.json'
NETWORK_FILENAME = TEST_DIR / 'data/testTopology_expected.json'


@pytest.mark.parametrize('degree, equalization_type, target, expected_pch_out_dbm, expected_si',
    [('east edfa in Lannion_CAS to Morlaix', 'target_pch_out_db', -20, -20, [-20, -20, -20, -20, -20]),
     ('east edfa in Lannion_CAS to Morlaix', 'target_psd_out_mWperGHz', 5e-4, -17.9588,
      [-17.9588, -16.7778, -14.9485, -16.7778, -17.9588]),
     ('east edfa in Lannion_CAS to Morlaix', 'target_out_mWperSlotWidth', 3e-4, -18.2390,
      [-19.4885, -18.2390, -16.4781, -18.2390, -19.4885]),
     ('east edfa in Lannion_CAS to Corlay', 'target_pch_out_db', -20, -16, [-16, -16, -16, -16, -16]),
     ('east edfa in Lannion_CAS to Corlay', 'target_psd_out_mWperGHz', 5e-4, -16, [-16, -16, -16, -16, -16]),
     ('east edfa in Lannion_CAS to Corlay', 'target_out_mWperSlotWidth', 5e-4, -16, [-16, -16, -16, -16, -16]),
     ('east edfa in Lannion_CAS to Stbrieuc', 'target_pch_out_db', -20, -17.16699,
      [-17.16698771, -15.98599459, -14.15668776, -15.98599459, -17.16698771]),
     ('east edfa in Lannion_CAS to Stbrieuc', 'target_psd_out_mWperGHz', 5e-4, -17.16699,
      [-17.16698771, -15.98599459, -14.15668776, -15.98599459, -17.16698771]),
     ('east edfa in Lannion_CAS to Stbrieuc', 'target_out_mWperSlotWidth', 5e-4, -17.16699,
      [-17.16698771, -15.98599459, -14.15668776, -15.98599459, -17.16698771])])
@pytest.mark.parametrize('delta_pdb_per_channel', [[0, 0, 0, 0, 0], [1, 3, 0, -5, 0]])
def test_equalization_combination_degree(delta_pdb_per_channel, degree, equalization_type, target,
                                         expected_pch_out_dbm, expected_si):
    """Check that ROADM correctly computes power of thr reference channel based on different
    combination of equalization for ROADM and per degree
    """

    roadm_config = {
        "uid": "roadm Lannion_CAS",
        "params": {
            "per_degree_pch_out_db": {
                "east edfa in Lannion_CAS to Corlay": -16
            },
            "per_degree_psd_out_mWperGHz": {
                "east edfa in Lannion_CAS to Stbrieuc": 6e-4
            },
            equalization_type: target,
            "add_drop_osnr": 38,
            "pmd": 0,
            "pdl": 0,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            }
        }
    }
    roadm = Roadm(**roadm_config)
    frequency = 191e12 + array([0, 50e9, 150e9, 225e9, 275e9])
    slot_width = array([37.5e9, 50e9, 75e9, 50e9, 37.5e9])
    baud_rate = array([32e9, 42e9, 64e9, 42e9, 32e9])
    signal = dbm2watt(array([-20.0, -18.0, -22.0, -25.0, -16.0]))
    ref_carrier = ReferenceCarrier(baud_rate=32e9, slot_width=50e9)
    pref = Pref(p_span0=0, p_spani=0, ref_carrier=ref_carrier)
    si = create_arbitrary_spectral_information(frequency=frequency, slot_width=slot_width,
                                               signal=signal, baud_rate=baud_rate, roll_off=0.15,
                                               delta_pdb_per_channel=delta_pdb_per_channel,
                                               tx_osnr=None, ref_power=pref)
    to_json_before_propagation = {
        'uid': 'roadm Lannion_CAS',
        'type': 'Roadm',
        'params': {
            equalization_type: target,
            'restrictions': {'preamp_variety_list': [], 'booster_variety_list': []},
            'per_degree_pch_out_db': {
                'east edfa in Lannion_CAS to Corlay': -16},
            "per_degree_psd_out_mWperGHz": {
                "east edfa in Lannion_CAS to Stbrieuc": 6e-4
            }
        },
        'metadata': {'location': {'latitude': 0, 'longitude': 0, 'city': None, 'region': None}}
    }
    assert roadm.to_json == to_json_before_propagation
    si = roadm(si, degree)
    assert roadm.ref_pch_out_dbm == pytest.approx(expected_pch_out_dbm, rel=1e-4)
    assert_allclose(expected_si, roadm.get_per_degree_power(degree, spectral_info=si), rtol=1e-3)


@pytest.mark.parametrize('equalization_type', ["target_psd_out_mWperGHz", "target_out_mWperSlotWidth"])
def test_wrong_element_config(equalization_type):
    """Check that 2 equalization correcty raise a config error
    """
    roadm_config = {
        "uid": "roadm Brest_KLA",
        "params": {
            "per_degree_pch_out_db": {},
            "target_pch_out_db": -20,
            equalization_type: 3.125e-4,
            "add_drop_osnr": 38,
            "pmd": 0,
            "pdl": 0,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            }
        },
        "metadata": {
            "location": {
                "city": "Brest_KLA",
                "region": "RLD",
                "latitude": 4.0,
                "longitude": 0.0
            }
        }
    }
    with pytest.raises(ConfigurationError):
        _ = Roadm(**roadm_config)


def test_merge_equalization():
    """Check that if equalization is not defined default one is correctly take and
    else that it is not overwritten
    """
    json_data = {
        "elements": [{
            "uid": "roadm Brest_KLA",
            "type": "Roadm"}],
        "connections": []
    }
    equipment = load_equipment(EQPT_FILENAME)
    network = network_from_json(json_data, equipment)
    roadm = [n for n in network.nodes()][0]
    assert roadm.target_pch_out_dbm == -20
    delattr(equipment['Roadm']['default'], 'target_pch_out_db')
    setattr(equipment['Roadm']['default'], 'target_psd_out_mWperGHz', power_dbm_to_psd_mw_ghz(-20, 32e9))
    # json_data is changed (type is popped from json_data with network_from_json_function). Create a new one:
    json_data = {
        "elements": [{
            "uid": "roadm Brest_KLA",
            "type": "Roadm"}],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    roadm = [n for n in network.nodes()][0]
    assert roadm.target_pch_out_dbm is None
    assert roadm.target_psd_out_mWperGHz == 3.125e-4
    assert roadm.target_out_mWperSlotWidth is None
    json_data = {
        "elements": [{
            "uid": "roadm Brest_KLA",
            "type": "Roadm",
            "params": {"target_pch_out_db": -18}}],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    roadm = [n for n in network.nodes()][0]
    assert roadm.target_pch_out_dbm == -18
    assert roadm.target_psd_out_mWperGHz is None
    assert roadm.target_out_mWperSlotWidth is None
    json_data = {
        "elements": [{
            "uid": "roadm Brest_KLA",
            "type": "Roadm",
            "params": {"target_psd_out_mWperGHz": 5e-4}}],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    roadm = [n for n in network.nodes()][0]
    assert roadm.target_pch_out_dbm is None
    assert roadm.target_psd_out_mWperGHz == 5e-4
    assert roadm.target_out_mWperSlotWidth is None
    json_data = {
        "elements": [{
            "uid": "roadm Brest_KLA",
            "type": "Roadm",
            "params": {"target_out_mWperSlotWidth": 3e-4}}],
        "connections": []
    }
    network = network_from_json(json_data, equipment)
    roadm = [n for n in network.nodes()][0]
    assert roadm.target_pch_out_dbm is None
    assert roadm.target_psd_out_mWperGHz is None
    assert roadm.target_out_mWperSlotWidth == 3e-4


@pytest.mark.parametrize('target_out, delta_pdb_per_channel, correction',
                         [(-20, [0, 1, 3, 0.5, -2], [0, 0, 5, 5.5, 0]),
                          (-20, [0, 0, 0, 0, 0], [0, 0, 2, 5, 0]),
                          (-20, [-2, -2, -2, -2, -2], [0, 0, 0, 3, 0]),
                          (-20, [0, 2, -2, -5, 4], [0, 0, 0, 0, 0]),
                          (-25.5, [0, 1, 3, 0.5, -2], [0, 0, 0, 0, 0]), ])
def test_low_input_power(target_out, delta_pdb_per_channel, correction):
    """check that ROADM correctly equalizes on small examples, assumes p_span_0 = 0
    case of power equalisation
    """
    frequency = 191e12 + array([0, 50e9, 150e9, 225e9, 275e9])
    slot_width = array([37.5e9, 50e9, 75e9, 50e9, 37.5e9])
    baud_rate = array([32e9, 42e9, 64e9, 42e9, 32e9])
    signal = dbm2watt(array([-20.0, -18.0, -22.0, -25.0, -16.0]))
    target = target_out + array(delta_pdb_per_channel)
    ref_carrier = ReferenceCarrier(baud_rate=32e9, slot_width=50e9)
    pref = Pref(p_span0=0, p_spani=-20, ref_carrier=ref_carrier)
    si = create_arbitrary_spectral_information(frequency=frequency, slot_width=slot_width,
                                               signal=signal, baud_rate=baud_rate, roll_off=0.15,
                                               delta_pdb_per_channel=delta_pdb_per_channel,
                                               tx_osnr=None, ref_power=pref)
    roadm_config = {
        "uid": "roadm Brest_KLA",
        "params": {
            "per_degree_pch_out_db": {},
            "target_pch_out_db": target_out,
            "add_drop_osnr": 38,
            "pmd": 0,
            "pdl": 0,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            }
        },
        "metadata": {
            "location": {
                "city": "Brest_KLA",
                "region": "RLD",
                "latitude": 4.0,
                "longitude": 0.0
            }
        }
    }
    roadm = Roadm(**roadm_config)
    si = roadm(si, 'toto')
    assert_allclose(watt2dbm(si.signal), target - correction, rtol=1e-5)
    # in other words check that if target is below input power, target is applied else power is unchanged
    assert_allclose((watt2dbm(signal) >= target) * target + (watt2dbm(signal) < target) * watt2dbm(signal),
                    watt2dbm(si.signal), rtol=1e-5)


@pytest.mark.parametrize('target_out, delta_pdb_per_channel, correction',
                         [(3.125e-4,
                           [0, 0, 0, 0, 0],
                           [0, 0, 2 + lin2db(64 / 32), 5 + lin2db(42 / 32), 0]),
                          (3.125e-4,
                           [1, 3, 0, -5, 0],
                           [1, 1 + lin2db(42 / 32), 2 + lin2db(64 / 32), 0 + lin2db(42 / 32), 0]), ])
def test_2low_input_power(target_out, delta_pdb_per_channel, correction):
    """check that ROADM correctly equalizes on small examples, assumes p_span_0 = 0
    case of PSD equalisation
    """
    frequency = 191e12 + array([0, 50e9, 150e9, 225e9, 275e9])
    slot_width = array([37.5e9, 50e9, 75e9, 50e9, 37.5e9])
    baud_rate = array([32e9, 42e9, 64e9, 42e9, 32e9])
    signal = dbm2watt(array([-20.0, -18.0, -22.0, -25.0, -16.0]))
    target = psd2powerdbm(target_out, baud_rate) + array(delta_pdb_per_channel)
    ref_carrier = ReferenceCarrier(baud_rate=32e9, slot_width=50e9)
    pref = Pref(p_span0=0, p_spani=-20, ref_carrier=ref_carrier)
    si = create_arbitrary_spectral_information(frequency=frequency, slot_width=slot_width,
                                               signal=signal, baud_rate=baud_rate, roll_off=0.15,
                                               delta_pdb_per_channel=delta_pdb_per_channel,
                                               tx_osnr=None, ref_power=pref)
    roadm_config = {
        "uid": "roadm Brest_KLA",
        "params": {
            "per_degree_pch_out_db": {},
            "target_psd_out_mWperGHz": target_out,
            "add_drop_osnr": 38,
            "pmd": 0,
            "pdl": 0,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            }
        },
        "metadata": {
            "location": {
                "city": "Brest_KLA",
                "region": "RLD",
                "latitude": 4.0,
                "longitude": 0.0
            }
        }
    }
    roadm = Roadm(**roadm_config)
    si = roadm(si, 'toto')
    assert_allclose(watt2dbm(si.signal), target - correction, rtol=1e-5)


def net_setup(equipment):
    """common setup for tests: builds network, equipment and oms only once"""
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    return network


def create_voyager_req(equipment, source, dest, bidir, nodes_list, loose_list, mode, spacing, power_dbm):
    """create the usual request list according to parameters"""
    params = {'request_id': 'test_request',
              'source': source,
              'bidir': bidir,
              'destination': dest,
              'trx_type': 'Voyager',
              'trx_mode': mode,
              'format': mode,
              'spacing': spacing,
              'nodes_list': nodes_list,
              'loose_list': loose_list,
              'path_bandwidth': 100.0e9,
              'effective_freq_slot': None}
    trx_params = trx_mode_params(equipment, params['trx_type'], params['trx_mode'], True)
    params.update(trx_params)
    params['power'] = dbm2watt(power_dbm) if power_dbm else dbm2watt(equipment['SI']['default'].power_dbm)
    f_min = params['f_min']
    f_max_from_si = params['f_max']
    params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
    return PathRequest(**params)


@pytest.mark.parametrize('power_dbm', [0, 1, -2, None])
@pytest.mark.parametrize('mode, slot_width', (['mode 1', 50e9], ['mode 2', 75e9]))
def test_initial_spectrum(mode, slot_width, power_dbm):
    """checks that propagation using the user defined spectrum identical to SI, gives same result as SI"""
    # first propagate without any req.initial_spectrum attribute
    equipment = load_equipment(EQPT_FILENAME)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             mode, slot_width, power_dbm)
    network = net_setup(equipment)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)
    # then creates req.initial_spectrum attribute exactly corresponding to -spectrum option files
    temp = [{
        "f_min": 191.35e12 + slot_width,
        "f_max": 196.15e12 - slot_width,
        "baud_rate": req.baud_rate,
        "slot_width": slot_width,
        "roll_off": 0.15,
        "tx_osnr": 40
    }]
    req.initial_spectrum = _spectrum_from_json(temp)
    infos_actual = propagate(path, req, equipment)
    print(infos_actual.frequency[0], infos_actual.frequency[-1])
    print(infos_expected.frequency[0], infos_expected.frequency[-1])

    assert_array_equal(infos_expected.frequency, infos_actual.frequency)
    assert_array_equal(infos_expected.baud_rate, infos_actual.baud_rate)
    assert_array_equal(infos_expected.slot_width, infos_actual.slot_width)
    assert_array_equal(infos_expected.signal, infos_actual.signal)
    assert_array_equal(infos_expected.nli, infos_actual.nli)
    assert_array_equal(infos_expected.ase, infos_actual.ase)
    assert_array_equal(infos_expected.roll_off, infos_actual.roll_off)
    assert_array_equal(infos_expected.chromatic_dispersion, infos_actual.chromatic_dispersion)
    assert_array_equal(infos_expected.pmd, infos_actual.pmd)
    assert_array_equal(infos_expected.channel_number, infos_actual.channel_number)
    assert_array_equal(infos_expected.number_of_channels, infos_actual.number_of_channels)


def test_initial_spectrum_not_identical():
    """checks that user defined spectrum overrides spectrum defined in SI
    """
    # first propagate without any req.initial_spectrum attribute
    equipment = load_equipment(EQPT_FILENAME)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             'mode 1', 50e9, 0)
    network = net_setup(equipment)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)
    # then creates req.initial_spectrum attribute exactly corresponding to -spectrum option files
    temp = [{
        "f_min": 191.4e12,     # align f_min , f_max on Voyager f_min, f_mix and not SI !
        "f_max": 196.1e12,
        "baud_rate": 40e9,
        "slot_width": 62.5e9,
        "roll_off": 0.15,
        "tx_osnr": 40
    }]
    req.initial_spectrum = _spectrum_from_json(temp)
    infos_actual = propagate(path, req, equipment)
    assert_raises(AssertionError, assert_array_equal, infos_expected.frequency, infos_actual.frequency)
    assert_raises(AssertionError, assert_array_equal, infos_expected.baud_rate, infos_actual.baud_rate)
    assert_raises(AssertionError, assert_array_equal, infos_expected.slot_width, infos_actual.slot_width)
    assert_raises(AssertionError, assert_array_equal, infos_expected.signal, infos_actual.signal)
    assert_raises(AssertionError, assert_array_equal, infos_expected.nli, infos_actual.nli)
    assert_raises(AssertionError, assert_array_equal, infos_expected.ase, infos_actual.ase)
    assert_raises(AssertionError, assert_array_equal, infos_expected.channel_number, infos_actual.channel_number)
    assert_raises(AssertionError, assert_array_equal, infos_expected.number_of_channels, infos_actual.number_of_channels)


@pytest.mark.parametrize('equalization, target_value', [
    ('target_out_mWperSlotWidth', power_dbm_to_psd_mw_ghz(-20, 50e9)),
    ('target_psd_out_mWperGHz', power_dbm_to_psd_mw_ghz(-20, 32e9))])
@pytest.mark.parametrize('power_dbm', [0, 2, -0.5])
def test_target_psd_or_psw(power_dbm, equalization, target_value):
    """checks that if target_out_mWperSlotWidth or target_psd_out_mWperGHz is defined, it is used as equalization
    and it gives same result if computed target is the same
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             'mode 1', 50e9, power_dbm)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)
    # change default equalization to power spectral density
    delattr(equipment['Roadm']['default'], 'target_pch_out_db')
    setattr(equipment['Roadm']['default'], equalization, target_value)
    # create a second instance with this roadm settings,
    network2 = net_setup(equipment)
    path2 = compute_constrained_path(network2, req)
    infos_actual = propagate(path2, req, equipment)
    # since baudrate is the same, resulting propagation should be the same as for power equalization
    assert_array_equal(infos_expected.baud_rate, infos_actual.baud_rate)
    assert_array_equal(infos_expected.slot_width, infos_actual.slot_width)
    assert_array_equal(infos_expected.signal, infos_actual.signal)
    assert_array_equal(infos_expected.nli, infos_actual.nli)
    assert_array_equal(infos_expected.ase, infos_actual.ase)
    assert_array_equal(infos_expected.roll_off, infos_actual.roll_off)
    assert_array_equal(infos_expected.chromatic_dispersion, infos_actual.chromatic_dispersion)
    assert_array_equal(infos_expected.pmd, infos_actual.pmd)
    assert_array_equal(infos_expected.channel_number, infos_actual.channel_number)
    assert_array_equal(infos_expected.number_of_channels, infos_actual.number_of_channels)


def ref_network():
    """Create a network instance with a instance of propagated path"""
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req0 = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                              'mode 1', 50e9, 0)
    path0 = compute_constrained_path(network, req0)
    _ = propagate(path0, req0, equipment)
    return network


@pytest.mark.parametrize('deltap', [0, +1.2, -0.5])
def test_target_psd_out_mwperghz_deltap(deltap):
    """checks that if target_psd_out_mWperGHz is defined, delta_p of amps is correctly updated

    Power over 1.2dBm saturate amp with this test: TODO add a test on this saturation
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             'mode 1', 50e9, deltap)
    temp = [{
        "f_min": 191.35e12,     # align f_min , f_max on Voyager f_min, f_mix and not SI !
        "f_max": 196.05e12,
        "baud_rate": req.baud_rate,
        "slot_width": 50e9,
        "roll_off": 0.15,
        "tx_osnr": 40
    }]
    req.initial_spectrum = _spectrum_from_json(temp)
    path = compute_constrained_path(network, req)
    _ = propagate(path, req, equipment)
    # check that gain of booster is changed accordingly whereas gain of preamp and ila is not (no saturation case)
    boosters = ['east edfa in Brest_KLA to Quimper', 'east edfa in Lorient_KMA to Vannes_KBE']
    ila_preamps = ['east edfa in Quimper to Lorient_KMA', 'west edfa in Lorient_KMA to Quimper',
                   'west edfa in Vannes_KBE to Lorient_KMA']
    for amp in boosters + ila_preamps:
        expected_amp = next(n for n in ref_network() if n.uid == amp)
        actual_amp = next(n for n in network.nodes() if n.uid == amp)
        expected_gain = expected_amp.pout_db - expected_amp.pin_db
        actual_gain = actual_amp.pout_db - actual_amp.pin_db
        print(actual_amp)
        if amp in boosters:
            assert expected_gain + deltap == pytest.approx(actual_gain, rel=1e-3)
        if amp in ila_preamps:
            assert expected_gain == pytest.approx(actual_gain, rel=1e-3)


@pytest.mark.parametrize('equalization', ['target_psd_out_mWperGHz', 'target_out_mWperSlotWidth'])
@pytest.mark.parametrize('case', ['SI', 'nodes'])
@pytest.mark.parametrize('deltap', [0, +2, -0.5])
@pytest.mark.parametrize('target', [-20, -21, -18])
@pytest.mark.parametrize('mode, slot_width', (['mode 1', 50e9], ['mode 2', 75e9]))
def test_equalization(case, deltap, target, mode, slot_width, equalization):
    """check that power target on roadm is correct for these cases; check on booster
    - SI : target_pch_out_db / target_psd_out_mWperGHz
    - node : target_pch_out_db / target_psd_out_mWperGHz
    - per degree : target_pch_out_db / target_psd_out_mWperGHz
    for these cases with and without power from user
    """
    equipment = load_equipment(EQPT_FILENAME)
    setattr(equipment['Roadm']['default'], 'target_pch_out_db', target)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Rennes_STA', False,
                             ['east edfa in Brest_KLA to Quimper', 'roadm Lannion_CAS', 'trx Rennes_STA'],
                             ['STRICT', 'STRICT', 'STRICT'],
                             mode, slot_width, deltap)
    roadms = ['roadm Brest_KLA', 'roadm Lorient_KMA', 'roadm Lannion_CAS', 'roadm Rennes_STA']
    # degree = {'roadm Brest_KLA': 'east edfa in Brest_KLA to Quimper',
    #           'roadm Lorient_KMA': 'east edfa in Lorient_KMA to Loudeac'}
    # boosters = ['east edfa in Brest_KLA to Quimper', 'east edfa in Lorient_KMA to Loudeac',
    #             'east edfa in Lannion_CAS to Stbrieuc']
    target_psd = power_dbm_to_psd_mw_ghz(target, 32e9)
    ref = ReferenceCarrier(baud_rate=32e9, slot_width=50e9)
    if case == 'SI':
        delattr(equipment['Roadm']['default'], 'target_pch_out_db')
        setattr(equipment['Roadm']['default'], equalization, target_psd)
        network = net_setup(equipment)
    elif case == 'nodes':
        json_data = load_json(NETWORK_FILENAME)
        for el in json_data['elements']:
            if el['uid'] in roadms:
                el['params'] = {equalization: target_psd}
        network = network_from_json(json_data, equipment)
        spectrum = equipment['SI']['default']
        p_db = spectrum.power_dbm
        p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
        build_network(network, equipment, p_db, p_total_db)
        # check that nodes not in roadms have target_pch_out_db not None
        pw_roadms = [r for r in network.nodes() if r.uid not in roadms and isinstance(r, Roadm)]
        for roadm in pw_roadms:
            assert roadm.target_psd_out_mWperGHz is None
            assert roadm.target_pch_out_dbm == target
        for roadm in [r for r in network.nodes() if r.uid in roadms and isinstance(r, Roadm)]:
            assert roadm.target_pch_out_dbm is None
            assert getattr(roadm, equalization) == target_psd
    path = compute_constrained_path(network, req)
    si = create_input_spectral_information(
        f_min=req.f_min, f_max=req.f_max, roll_off=req.roll_off, baud_rate=req.baud_rate, power=req.power,
        spacing=req.spacing, tx_osnr=req.tx_osnr, ref_carrier=ref)
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            si = el(si, degree=path[i + 1].uid)
            if case in ['SI', 'nodes', 'degrees']:
                if equalization == 'target_psd_out_mWperGHz':
                    assert_allclose(power_dbm_to_psd_mw_ghz(watt2dbm(si.signal + si.ase + si.nli), si.baud_rate),
                                    target_psd, rtol=1e-3)
                if equalization == 'target_out_mWperSlotWidth':
                    assert_allclose(power_dbm_to_psd_mw_ghz(watt2dbm(si.signal + si.ase + si.nli), si.slot_width),
                                    target_psd, rtol=1e-3)
        else:
            si = el(si)
        print(el.uid)


@pytest.mark.parametrize('req_power', [0, 2, -1.5])
def test_power_option(req_power):
    """check that --po option adds correctly power with spectral information
    """
    equipment = load_equipment(EQPT_FILENAME)
    setattr(equipment['Roadm']['default'], 'target_pch_out_db', None)
    setattr(equipment['Roadm']['default'], 'target_psd_out_mWperGHz', power_dbm_to_psd_mw_ghz(-20, 32e9))
    network = net_setup(equipment)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             'mode 1', 50e9, req_power)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)

    temp = [{
        "f_min": 191.4e12,     # align f_min , f_max on Voyager f_min, f_max and not SI !
        "f_max": 196.1e12,
        "baud_rate": req.baud_rate,
        "slot_width": 50e9,
        "roll_off": 0.15,
        "tx_osnr": 40
    }]
    req.initial_spectrum = _spectrum_from_json(temp)
    network2 = net_setup(equipment)
    path2 = compute_constrained_path(network2, req)
    infos_actual = propagate(path2, req, equipment)
    assert_array_equal(infos_expected.baud_rate, infos_actual.baud_rate)
    assert_array_equal(infos_expected.slot_width, infos_actual.slot_width)
    assert_array_equal(infos_expected.signal, infos_actual.signal)
    assert_array_equal(infos_expected.nli, infos_actual.nli)
    assert_array_equal(infos_expected.ase, infos_actual.ase)
    assert_array_equal(infos_expected.roll_off, infos_actual.roll_off)
    assert_array_equal(infos_expected.chromatic_dispersion, infos_actual.chromatic_dispersion)
    assert_array_equal(infos_expected.pmd, infos_actual.pmd)
    assert_array_equal(infos_expected.channel_number, infos_actual.channel_number)
    assert_array_equal(infos_expected.number_of_channels, infos_actual.number_of_channels)

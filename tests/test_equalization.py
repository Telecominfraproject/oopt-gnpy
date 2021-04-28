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

from gnpy.core.utils import lin2db, automatic_nch, dbm2watt, powerdbm2psdmwperghz, watt2dbm
from gnpy.core.network import build_network
from gnpy.core.elements import Roadm
from gnpy.core.info import create_input_spectral_information
from gnpy.core.equipment import trx_mode_params
from gnpy.tools.json_io import network_from_json, load_equipment, load_network, _spectrum_from_json, load_json
from gnpy.topology.request import PathRequest, compute_constrained_path, propagate, update_spectrum_power, ref_carrier


TEST_DIR = Path(__file__).parent
EQPT_FILENAME = TEST_DIR / 'data/eqpt_config.json'
NETWORK_FILENAME = TEST_DIR / 'data/testTopology_expected.json'


def net_setup(equipment):
    """ common setup for tests: builds network, equipment and oms only once
    """
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    return network


def create_voyager_req(equipment, source, dest, bidir, nodes_list, loose_list, mode, spacing, power_dbm):
    """ create the usual request list according to parameters
    """
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
    """ checks that propagation using the user defined spectrum identical to SI, gives same result as SI
    """
    # first propagate without any req.initial_spectrum attribute
    equipment = load_equipment(EQPT_FILENAME)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             mode, slot_width, power_dbm)
    network = net_setup(equipment)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)
    # then creates req.initial_spectrum attribute exactly corresponding to -spectrum option files
    temp = [{
             "f_min": 191.35e12,     # align f_min , f_max on Voyager f_min, f_mix and not SI !
             "f_max": 196.1e12,
             "baud_rate": req.baud_rate,
             "slot_width": slot_width,
             "roll_off": 0.15,
             "tx_osnr": 40
            }]
    if power_dbm:
        temp[0]['power_dbm'] = power_dbm
    req.initial_spectrum = _spectrum_from_json(temp, equipment)
    update_spectrum_power(req)
    infos_actual = propagate(path, req, equipment)
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
    """ checks that user defined spectrum overrides spectrum defined in SI
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
             "f_min": 191.35e12,     # align f_min , f_max on Voyager f_min, f_mix and not SI !
             "f_max": 196.1e12,
             "baud_rate": 40e9,
             "slot_width": 62.5e9,
             "roll_off": 0.15,
             "tx_osnr": 40
            }]
    req.initial_spectrum = _spectrum_from_json(temp, equipment)
    update_spectrum_power(req)
    infos_actual = propagate(path, req, equipment)
    assert_raises(AssertionError, assert_array_equal, infos_expected.baud_rate, infos_actual.baud_rate)
    assert_raises(AssertionError, assert_array_equal, infos_expected.slot_width, infos_actual.slot_width)
    assert_raises(AssertionError, assert_array_equal, infos_expected.signal, infos_actual.signal)
    assert_raises(AssertionError, assert_array_equal, infos_expected.nli, infos_actual.nli)
    assert_raises(AssertionError, assert_array_equal, infos_expected.ase, infos_actual.ase)
    assert_raises(AssertionError, assert_array_equal, infos_expected.channel_number, infos_actual.channel_number)
    assert_raises(AssertionError, assert_array_equal, infos_expected.number_of_channels, infos_actual.number_of_channels)


@pytest.mark.parametrize('with_initial_spectrum', [None, 0, +2, -0.5])
def test_target_psd_out_mwperghz(with_initial_spectrum):
    """ checks that if target_psd_out_mWperGHz is defined, it is used as equalization, and it gives same result if
    computed target is the same
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             'mode 1', 50e9, 0)
    if with_initial_spectrum:
        temp = [{
                 "f_min": 191.35e12,     # align f_min , f_max on Voyager f_min, f_max and not SI !
                 "f_max": 196.1e12,
                 "baud_rate": req.baud_rate,
                 "power_dbm": with_initial_spectrum,
                 "slot_width": 50e9,
                 "roll_off": 0.15,
                 "tx_osnr": 40
                }]
        req.initial_spectrum = _spectrum_from_json(temp, equipment)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)
    # change default equalization to power spectral density
    setattr(equipment['Roadm']['default'], 'target_pch_out_db', None)
    setattr(equipment['Roadm']['default'], 'target_psd_out_mWperGHz', powerdbm2psdmwperghz(-20, 32e9))
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
    """ Create a network instance with a instance of propagated path
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req0 = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                              'mode 1', 50e9, 0)
    path0 = compute_constrained_path(network, req0)
    _ = propagate(path0, req0, equipment)
    return network


@pytest.mark.parametrize('deltap', [0, +1.2, -0.5])
def test_target_psd_out_mwperghz_deltap(deltap):
    """ checks that if target_psd_out_mWperGHz is defined, delta_p of amps is correctly updated
    Power over 1.2dBm saturate amp with this test: TODO add a test on this saturation
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             'mode 1', 50e9, deltap)
    temp = [{
             "f_min": 191.35e12,     # align f_min , f_max on Voyager f_min, f_mix and not SI !
             "f_max": 196.1e12,
             "baud_rate": req.baud_rate,
             "slot_width": 50e9,
             "roll_off": 0.15,
             "tx_osnr": 40
            }]
    req.initial_spectrum = _spectrum_from_json(temp, equipment)
    update_spectrum_power(req)
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


@pytest.mark.parametrize('case', ['SI', 'nodes'])
@pytest.mark.parametrize('deltap', [0, +2, -0.5])
@pytest.mark.parametrize('target', [-20, -21, -18])
@pytest.mark.parametrize('mode, slot_width', (['mode 1', 50e9], ['mode 2', 75e9]))
def test_equalization(case, deltap, target, mode, slot_width):
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
    target_psd = powerdbm2psdmwperghz(target, 32e9)
    ref = ref_carrier(req.power, equipment)
    if case == 'SI':
        setattr(equipment['Roadm']['default'], 'target_pch_out_db', None)
        setattr(equipment['Roadm']['default'], 'target_psd_out_mWperGHz',
                target_psd)
        network = net_setup(equipment)
    elif case == 'nodes':
        json_data = load_json(NETWORK_FILENAME)
        for el in json_data['elements']:
            if el['uid'] in roadms:
                el['params'] = {'target_psd_out_mWperGHz': target_psd}
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
    path = compute_constrained_path(network, req)
    si = create_input_spectral_information(
        req.f_min, req.f_max, req.roll_off, req.baud_rate, req.power, req.spacing, ref_carrier=ref)
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            si = el(si, degree=path[i+1].uid)
            if case in ['SI', 'nodes', 'degrees']:
                assert_allclose(powerdbm2psdmwperghz(watt2dbm(si.signal + si.ase + si.nli), si.baud_rate),
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
    setattr(equipment['Roadm']['default'], 'target_psd_out_mWperGHz', powerdbm2psdmwperghz(-20, 32e9))
    network = net_setup(equipment)
    req = create_voyager_req(equipment, 'trx Brest_KLA', 'trx Vannes_KBE', False, ['trx Vannes_KBE'], ['STRICT'],
                             'mode 1', 50e9, req_power)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)

    temp = [{
             "f_min": 191.35e12,     # align f_min , f_max on Voyager f_min, f_max and not SI !
             "f_max": 196.1e12,
             "baud_rate": req.baud_rate,
             "slot_width": 50e9,
             "roll_off": 0.15,
             "tx_osnr": 40
            }]
    req.initial_spectrum = _spectrum_from_json(temp, equipment)
    update_spectrum_power(req)
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

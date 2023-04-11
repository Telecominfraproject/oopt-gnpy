#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_propagation
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
Check that propagation example give expected results
"""

import pytest

from pathlib import Path
from networkx import dijkstra_path
from numpy import mean, sqrt, ones
from numpy.testing import assert_allclose
import re

from gnpy.core.exceptions import SpectrumError
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.utils import db2lin, dbm2watt
from gnpy.core.info import create_input_spectral_information
from gnpy.core.network import build_network
from gnpy.tools.json_io import load_network, load_equipment, network_from_json, load_json, load_gnpy_json
from gnpy.topology.request import PathRequest

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
network_file_name = DATA_DIR / 'LinkforTest.json'
eqpt_library_name = DATA_DIR / 'eqpt_config.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": load_json(DATA_DIR / "std_medium_gain_advanced_config.json")}
FIBER_SLOPE_NETWORK = DATA_DIR / 'fiber_slope' / 'test_network_fiber_freq_legacy_and_alternate_models.json'
FIBER_SLOPE_EQPT = DATA_DIR / 'fiber_slope' / 'eqpt_config_fiber_freq.json'


@pytest.fixture(params=[(96, 0.05e12), (60, 0.075e12), (45, 0.1e12), (2, 0.1e12)],
                ids=['50GHz spacing', '75GHz spacing', '100GHz spacing', '2 channels'])
# TODO in elements.py code: pytests doesn't pass with 1 channel: interpolate fail
def nch_and_spacing(request):
    """parametrize channel count vs channel spacing (Hz)"""
    yield request.param


def pathrequest(pch_dbm, p_tot_dbm):
    """create ref channel for defined power settings
    """
    params = {
        "power": dbm2watt(pch_dbm),
        "tx_power": dbm2watt(pch_dbm),
        "nb_channel": round(dbm2watt(p_tot_dbm) / dbm2watt(pch_dbm), 0)
    }
    return PathRequest(**params)


def propagation(input_power, con_in, con_out, dest):
    equipment = load_equipment(eqpt_library_name, EXTRA_CONFIGS)
    network = load_network(network_file_name, equipment)

    # parametrize the network elements with the con losses and adapt gain
    # (assumes all spans are identical)
    for e in network.nodes():
        if isinstance(e, Fiber):
            loss = e.params.loss_coef * e.params.length
            e.params.con_in = con_in
            e.params.con_out = con_out
        if isinstance(e, Edfa):
            e.operational.gain_target = loss + con_in + con_out

    build_network(network, equipment, pathrequest(0, 20))

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}

    p = input_power
    p = db2lin(p) * 1e-3
    spacing = 50e9  # THz
    si = create_input_spectral_information(f_min=191.3e12, f_max=191.3e12 + 79 * spacing, roll_off=0.15,
                                           baud_rate=32e9, spacing=spacing, tx_osnr=None,
                                           tx_power=p)
    source = next(transceivers[uid] for uid in transceivers if uid == 'trx A')
    sink = next(transceivers[uid] for uid in transceivers if uid == dest)
    path = dijkstra_path(network, source, sink)
    for el in path:
        si = el(si)
        print(el)  # remove this line when sweeping across several powers
    edfa_sample = next(el for el in path if isinstance(el, Edfa))
    nf = mean(edfa_sample.nf)

    print(f'pw: {input_power} conn in: {con_in} con out: {con_out}',
          f'OSNR@0.1nm: {round(mean(sink.osnr_ase_01nm),2)}',
          f'SNR@bandwitdth: {round(mean(sink.snr),2)}')
    return sink, nf, path


test = {'a': (-1, 1, 0), 'b': (-1, 1, 1), 'c': (0, 1, 0), 'd': (1, 1, 1)}
expected = {'a': (-2, 0, 0), 'b': (-2, 0, 1), 'c': (-1, 0, 0), 'd': (0, 0, 1)}


@pytest.mark.parametrize("dest", ['trx B', 'trx F'])
@pytest.mark.parametrize("osnr_test", ['a', 'b', 'c', 'd'])
def test_snr(osnr_test, dest):
    pw = test[osnr_test][0]
    conn_in = test[osnr_test][1]
    conn_out = test[osnr_test][2]
    sink, nf, _ = propagation(pw, conn_in, conn_out, dest)
    osnr = round(mean(sink.osnr_ase), 3)
    nli = 1.0 / db2lin(round(mean(sink.snr), 3)) - 1.0 / db2lin(osnr)
    pw = expected[osnr_test][0]
    conn_in = expected[osnr_test][1]
    conn_out = expected[osnr_test][2]
    sink, exp_nf, _ = propagation(pw, conn_in, conn_out, dest)
    expected_osnr = round(mean(sink.osnr_ase), 3)
    expected_nli = 1.0 / db2lin(round(mean(sink.snr), 3)) - 1.0 / db2lin(expected_osnr)
    # compare OSNR taking into account nf change of amps
    osnr_diff = abs(osnr - expected_osnr + nf - exp_nf)
    nli_diff = abs((nli - expected_nli) / nli)
    assert osnr_diff < 0.01 and nli_diff < 0.01


@pytest.mark.parametrize("dest", ['trx B', 'trx F'])
@pytest.mark.parametrize("cd_test", ['a', 'b', 'c', 'd'])
def test_chromatic_dispersion(cd_test, dest):
    pw = test[cd_test][0]
    conn_in = test[cd_test][1]
    conn_out = test[cd_test][2]
    sink, _, path = propagation(pw, conn_in, conn_out, dest)

    chromatic_dispersion = sink.chromatic_dispersion

    num_ch = len(chromatic_dispersion)
    expected_cd = 0
    for el in path:
        expected_cd += el.params.dispersion * el.params.length if isinstance(el, Fiber) else 0
    expected_cd = expected_cd * ones(num_ch) * 1e3
    assert chromatic_dispersion == pytest.approx(expected_cd)


@pytest.mark.parametrize("dest", ['trx B', 'trx F'])
@pytest.mark.parametrize("dgd_test", ['a', 'b', 'c', 'd'])
def test_dgd(dgd_test, dest):
    pw = test[dgd_test][0]
    conn_in = test[dgd_test][1]
    conn_out = test[dgd_test][2]
    sink, _, path = propagation(pw, conn_in, conn_out, dest)

    pmd = sink.pmd

    num_ch = len(pmd)
    expected_pmd = 0
    for el in path:
        expected_pmd += el.params.pmd_coef**2 * el.params.length if isinstance(el, Fiber) else 0
        expected_pmd += el.params.pmd**2 if isinstance(el, Roadm) else 0
    expected_pmd = sqrt(expected_pmd) * ones(num_ch) * 1e12
    assert pmd == pytest.approx(expected_pmd)


def wrong_element_propagate():
    """
    """
    data = []
    data.append({
        "error": SpectrumError,
        "json_data": {
            "elements": [{
                "uid": "Elem",
                "type": "Fiber",
                "type_variety": "SSMF",
                "params": {
                    "dispersion_per_frequency": {
                        "frequency": [
                            185.49234135667396e12,
                            186.05251641137855e12,
                            188.01312910284463e12,
                            189.99124726477024e12],
                        "value": [
                            1.60e-05,
                            1.67e-05,
                            1.7e-05,
                            1.8e-05]
                    },
                    "length": 1.02,
                    "loss_coef": 2.85,
                    "length_units": "km",
                    "att_in": 0.0,
                    "con_in": 0.0,
                    "con_out": 0.0
                }
            }],
            "connections": []
        },
        "expected_msg": 'The spectrum bandwidth exceeds the frequency interval used to define the fiber Chromatic '
                        + 'Dispersion in "Fiber Elem".\nSpectrum f_min-f_max: 191.35-196.1 THz\nChromatic Dispersion '
                        + 'f_min-f_max: 185.49-189.99 THz'
    })
    return data


@pytest.mark.parametrize('error, json_data, expected_msg',
                         [(e['error'], e['json_data'], e['expected_msg']) for e in wrong_element_propagate()])
def test_json_element(error, json_data, expected_msg):
    """
    Check that a missing key is correctly raisong the logger
    """
    equipment = load_equipment(eqpt_library_name, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    elem = next(e for e in network.nodes() if e.uid == 'Elem')
    si = create_input_spectral_information(f_min=191.3e12, f_max=196.1e12, roll_off=0.15,
                                           baud_rate=32e9, tx_power=1.0e-3, spacing=50.0e9, tx_osnr=45)
    with pytest.raises(error, match=re.escape(expected_msg)):
        _ = elem(si)


def test_fiber_slope():
    """Check that loss is as expected when there is a fiber slope definition with eqpt
    """
    equipment = load_equipment(FIBER_SLOPE_EQPT, EXTRA_CONFIGS)
    # change loss_coef_ripple and ref_frequency compared to original library to ease the test
    equipment['Fiber']['SSMF_freq'].loss_coef_ripple = [
        {
            "frequency": 185500000000000.0,
            "loss_coef_ripple_value": 0.1949626865671642 - 0.1847014925373134
        },
        {
            "frequency": 186050000000000.0,
            "loss_coef_ripple_value": 0.1921641791044776 - 0.1847014925373134
        },
        {
            "frequency": 188000000000000.0,
            "loss_coef_ripple_value": 0.1865671641791045 - 0.1847014925373134
        },
        {
            "frequency": 190000000000000.0,
            "loss_coef_ripple_value": 0.1847014925373134 - 0.1847014925373134
        },
        {
            "frequency": 191000000000000.0,
            "loss_coef_ripple_value": 0.1842350746268657 - 0.1847014925373134
        },
        {
            "frequency": 192000000000000.0,
            "loss_coef_ripple_value": 0.1847014925373134 - 0.1847014925373134
        },
        {
            "frequency": 194000000000000.0,
            "loss_coef_ripple_value": 0.1870335820895522 - 0.1847014925373134
        },
        {
            "frequency": 196000000000000.0,
            "loss_coef_ripple_value": 0.1912313432835821 - 0.1847014925373134
        },
        {
            "frequency": 198000000000000.0,
            "loss_coef_ripple_value": 0.1963619402985075 - 0.1847014925373134
        },
        {
            "frequency": 200000000000000.0,
            "loss_coef_ripple_value": 0.2024253731343283 - 0.1847014925373134
        }]
    equipment['Fiber']['SSMF_freq'].ref_frequency = 192000000000000.0
    json_data = load_gnpy_json(FIBER_SLOPE_NETWORK)
    # use 0.25/km dB for the test
    assert json_data['elements'][1]['uid'] == 'Span1'
    json_data['elements'][1]['params']['loss_coef'] = 0.25
    network = network_from_json(json_data, equipment)
    # build all the missing pieces in span element
    build_network(network, equipment, pathrequest(0, 20))
    fiber = next(e for e in network.nodes() if e.uid == 'Span1')
    si = create_input_spectral_information(f_min=191.25e12, f_max=196.05e12, roll_off=0.15,
                                           baud_rate=32e9, tx_power=1.0e-3, spacing=50.0e9, tx_osnr=45)
    si = fiber(si)
    # find the loss of the exact offset frequency and verify it matches with the scalar loss definition 0.025
    assert_allclose(si.pch[14], dbm2watt(-0.25 * 80), atol=1e-9)
    assert_allclose(si.frequency[-2], 196.0e12, atol=1e-9)
    # find the loss coef of another frequency also defined in the lut and compute the expected loss
    assert_allclose(si.pch[-2], dbm2watt(-(0.1912313432835821 - 0.1847014925373134 + 0.25) * 80.0), atol=1e-9)
    # check that to_json respects the same nb of digits for frequency and loss_coeff_value
    assert f'{fiber.to_json["params"]["loss_coef_per_frequency"][0]["loss_coef_value"]}' == '0.2602611940298508'
    # check the legacy definition on span2
    si = create_input_spectral_information(f_min=191.25e12, f_max=196.05e12, roll_off=0.15,
                                           baud_rate=32e9, tx_power=1.0e-3, spacing=50.0e9, tx_osnr=45)
    fiber = next(e for e in network.nodes() if e.uid == 'Span2')
    si = fiber(si)
    assert_allclose(si.pch, dbm2watt(-0.29 * 80), atol=1e-9)


def test_fiber_slope_no_ref():
    """Check that loss is as expected when there is a fiber slope definition with eqpt
    """
    equipment = load_equipment(FIBER_SLOPE_EQPT, EXTRA_CONFIGS)
    # change loss_coef_lut and offset frequency compared to original library to ease the test
    # ref_wavelength is 1550nm
    # @191.3 THz, loss_coeff is expected to be 0.25 dB/km
    equipment['Fiber']['SSMF_freq'].loss_coef_ripple = [
        {
            "frequency": 185500000000000.0,
            "loss_coef_ripple_value": 0.0086118364829081
        },
        {
            "frequency": 186050000000000.0,
            "loss_coef_ripple_value": 0.0058133290202215
        },
        {
            "frequency": 188000000000000.0,
            "loss_coef_ripple_value": 0.0002163140948484
        },
        {
            "frequency": 190000000000000.0,
            "loss_coef_ripple_value": -0.0016493575469427
        },
        {
            "frequency": 191000000000000.0,
            "loss_coef_ripple_value": -0.0021157754573904
        },
        {
            "frequency": 192000000000000.0,
            "loss_coef_ripple_value": -0.0016493575469427
        },
        {
            "frequency": 194000000000000.0,
            "loss_coef_ripple_value": 0.0006827320052961
        },
        {
            "frequency": 196000000000000.0,
            "loss_coef_ripple_value": 0.004880493199326
        },
        {
            "frequency": 198000000000000.0,
            "loss_coef_ripple_value": 0.0100110902142514
        },
        {
            "frequency": 200000000000000.0,
            "loss_coef_ripple_value": 0.0160745230500722
        }]
    network = load_network(FIBER_SLOPE_NETWORK, equipment)
    # build all the missing pieces in span element
    build_network(network, equipment, pathrequest(0, 20))
    fiber = next(e for e in network.nodes() if e.uid == 'Span1')
    si = create_input_spectral_information(f_min=191.25e12, f_max=196.05e12, roll_off=0.15,
                                           baud_rate=32e9, tx_power=1.0e-3, spacing=50.0e9, tx_osnr=45)
    si = fiber(si)
    # find the loss of the exact offset frequency and verify it matches with the scalar loss definition 0.025
    assert_allclose(si.pch[14], dbm2watt(-(0.251987 - 0.0016493575469427) * 80), rtol=1e-3)
    assert_allclose(si.frequency[0], 191.3e12, atol=1e-9)
    # find the loss coef of another frequency also defined in the lut and compute the expected loss
    assert_allclose(si.pch[0], dbm2watt(-0.25 * 80.0), rtol=1e-3)
    # check that to_json respects the same nb of digits for frequency and loss_coeff_value
    assert f'{fiber.to_json["params"]["loss_coef_per_frequency"][0]["loss_coef_value"]}' == '0.2605988364829081'


if __name__ == '__main__':
    from logging import getLogger, basicConfig, INFO
    logger = getLogger(__name__)
    basicConfig(level=INFO)

    for a in test:
        test_snr(a, 'trx F')
    print('\n')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

import pytest

from pathlib import Path
from networkx import dijkstra_path
from numpy import mean, sqrt, ones
import re

from gnpy.core.exceptions import SpectrumError
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.utils import db2lin
from gnpy.core.info import create_input_spectral_information
from gnpy.core.network import build_network
from gnpy.tools.json_io import load_network, load_equipment, network_from_json


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
network_file_name = DATA_DIR / 'LinkforTest.json'
eqpt_library_name = DATA_DIR / 'eqpt_config.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}


@pytest.fixture(params=[(96, 0.05e12), (60, 0.075e12), (45, 0.1e12), (2, 0.1e12)],
                ids=['50GHz spacing', '75GHz spacing', '100GHz spacing', '2 channels'])
# TODO in elements.py code: pytests doesn't pass with 1 channel: interpolate fail
def nch_and_spacing(request):
    """parametrize channel count vs channel spacing (Hz)"""
    yield request.param


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

    build_network(network, equipment, 0, 20)

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
                        + 'Dispersion in "Fiber Elem".\nSpectrum f_min-f_max: 191.35-196.1\nChromatic Dispersion '
                        + 'f_min-f_max: 185.49-189.99'
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


if __name__ == '__main__':
    from logging import getLogger, basicConfig, INFO
    logger = getLogger(__name__)
    basicConfig(level=INFO)

    for a in test:
        test_snr(a, 'trx F')
    print('\n')

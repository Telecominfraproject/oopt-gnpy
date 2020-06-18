#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

import pytest
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.utils import db2lin
from gnpy.core.info import create_input_spectral_information
from gnpy.core.network import build_network
from gnpy.tools.json_io import load_network, load_equipment
from pathlib import Path
from networkx import dijkstra_path
from numpy import mean, sqrt, ones

network_file_name = Path(__file__).parent.parent / 'tests/LinkforTest.json'
eqpt_library_name = Path(__file__).parent.parent / 'tests/data/eqpt_config.json'


@pytest.fixture(params=[(96, 0.05e12), (60, 0.075e12), (45, 0.1e12), (2, 0.1e12)],
                ids=['50GHz spacing', '75GHz spacing', '100GHz spacing', '2 channels'])
# TODO in elements.py code: pytests doesn't pass with 1 channel: interpolate fail
def nch_and_spacing(request):
    """parametrize channel count vs channel spacing (Hz)"""
    yield request.param


def propagation(input_power, con_in, con_out, dest):
    equipment = load_equipment(eqpt_library_name)
    network = load_network(network_file_name, equipment)
    build_network(network, equipment, 0, 20)

    # parametrize the network elements with the con losses and adapt gain
    # (assumes all spans are identical)
    for e in network.nodes():
        if isinstance(e, Fiber):
            loss = e.params.loss_coef * e.params.length
            e.params.con_in = con_in
            e.params.con_out = con_out
        if isinstance(e, Edfa):
            e.operational.gain_target = loss + con_in + con_out

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}

    p = input_power
    p = db2lin(p) * 1e-3
    spacing = 50e9  # THz
    si = create_input_spectral_information(191.3e12, 191.3e12 + 79 * spacing, 0.15, 32e9, p, spacing)
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


if __name__ == '__main__':
    from logging import getLogger, basicConfig, INFO
    logger = getLogger(__name__)
    basicConfig(level=INFO)

    for a in test:
        test_snr(a, 'trx F')
    print('\n')

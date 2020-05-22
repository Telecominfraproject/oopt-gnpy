#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

from gnpy.core.elements import Edfa
import numpy as np
from json import load
import pytest
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.utils import lin2db, db2lin
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power
from gnpy.core.equipment import load_equipment
from gnpy.core.network import build_network, load_network
from pathlib import Path
from networkx import dijkstra_path
from numpy import mean, sqrt
from pytest import approx

#network_file_name = 'tests/test_network.json'
network_file_name = Path(__file__).parent.parent / 'tests/LinkforTest.json'
#TODO: note that this json entries has a weird topology since EDfa1 has a possible branch on a receiver B
# this might not pass future tests/ code updates
#network_file_name = Path(__file__).parent.parent / 'examples/edfa_example_network.json'
eqpt_library_name = Path(__file__).parent.parent / 'tests/data/eqpt_config.json'

@pytest.fixture(params=[(96, 0.05e12), (60, 0.075e12), (45, 0.1e12), (2, 0.1e12)],
    ids=['50GHz spacing', '75GHz spacing', '100GHz spacing', '2 channels'])
# TODO in elements.py code: pytests doesn't pass with 1 channel: interpolate fail
def nch_and_spacing(request):
    """parametrize channel count vs channel spacing (Hz)"""
    yield request.param

def propagation(input_power, con_in, con_out,dest):
    equipment = load_equipment(eqpt_library_name)
    network = load_network(network_file_name,equipment)
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
    spacing = 50e9 # THz
    si = create_input_spectral_information(191.3e12, 191.3e12+79*spacing, 0.15, 32e9, p, spacing)
    source = next(transceivers[uid] for uid in transceivers if uid == 'trx A')
    sink = next(transceivers[uid] for uid in transceivers if uid == dest)
    path = dijkstra_path(network, source, sink)
    for el in path:
        si = el(si)
        print(el) # remove this line when sweeping across several powers
    edfa_sample = next(el for el in path if isinstance(el, Edfa))
    nf = mean(edfa_sample.nf)

    print(f'pw: {input_power} conn in: {con_in} con out: {con_out}',
          f'OSNR@0.1nm: {round(mean(sink.osnr_ase_01nm),2)}',
          f'SNR@bandwitdth: {round(mean(sink.snr),2)}')
    return sink, nf, path


test = {'a':(-1,1,0),'b':(-1,1,1),'c':(0,1,0),'d':(1,1,1)}
expected = {'a': (-2, 0, 0), 'b': (-2, 0, 1), 'c': (-1, 0, 0), 'd': (0, 0, 1)}


@pytest.mark.parametrize("dest",['trx B','trx F'])
@pytest.mark.parametrize("osnr_test", ['a', 'b', 'c', 'd'])
def test_snr(osnr_test, dest):
    pw = test[osnr_test][0]
    conn_in = test[osnr_test][1]
    conn_out = test[osnr_test][2]
    sink, nf, _ = propagation(pw, conn_in, conn_out, dest)
    osnr = round(mean(sink.osnr_ase), 3)
    nli = 1.0/db2lin(round(mean(sink.snr), 3)) - 1.0/db2lin(osnr)
    pw = expected[osnr_test][0]
    conn_in = expected[osnr_test][1]
    conn_out = expected[osnr_test][2]
    sink, exp_nf, _ = propagation(pw, conn_in, conn_out, dest)
    expected_osnr = round(mean(sink.osnr_ase), 3)
    expected_nli = 1.0/db2lin(round(mean(sink.snr), 3)) - 1.0/db2lin(expected_osnr)
    # compare OSNR taking into account nf change of amps
    osnr_diff = abs(osnr - expected_osnr + nf - exp_nf)
    nli_diff = abs((nli-expected_nli)/nli)
    assert osnr_diff < 0.01 and nli_diff < 0.01


@pytest.mark.parametrize("dest", ['trx B', 'trx F'])
@pytest.mark.parametrize("osnr_test", ['a', 'b', 'c', 'd'])
def test_chromatic_dispersion(osnr_test, dest):
    pw = test[osnr_test][0]
    conn_in = test[osnr_test][1]
    conn_out = test[osnr_test][2]
    sink, _, path = propagation(pw, conn_in, conn_out, dest)

    chromatic_dispersion = sink.chromatic_dispersion

    num_ch = len(chromatic_dispersion)
    expected_cd = 0
    for el in path:
        expected_cd += el.params.dispersion * el.params.length if isinstance(el, Fiber) else 0
    expected_cd = expected_cd * np.ones(num_ch)
    assert chromatic_dispersion == approx(expected_cd)


@pytest.mark.parametrize("dest", ['trx B', 'trx F'])
@pytest.mark.parametrize("osnr_test", ['a', 'b', 'c', 'd'])
def test_dgd(osnr_test, dest):
    pw = test[osnr_test][0]
    conn_in = test[osnr_test][1]
    conn_out = test[osnr_test][2]
    sink, _, path = propagation(pw, conn_in, conn_out, dest)

    dgd = sink.dgd

    num_ch = len(dgd)
    expected_dgd = 0
    for el in path:
        expected_dgd += el.params.pmd * sqrt(el.params.length) if isinstance(el, Fiber) else 0
        expected_dgd += el.params.dgd if isinstance(el, Roadm) else 0
    expected_dgd = expected_dgd * np.ones(num_ch) * 1e12
    # compare DGD
    assert dgd == approx(expected_dgd)


if __name__ == '__main__':
    from logging import getLogger, basicConfig, INFO
    logger = getLogger(__name__)
    basicConfig(level=INFO)

    for a in test:
        test_snr(a, 'trx F')
    print('\n')

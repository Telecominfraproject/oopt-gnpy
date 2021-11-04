# SPDX-License-Identifier: BSD-3-Clause
#
# Reading and writing JSON files for GNPy
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

""" Check request.py functions on different contexts
"""

from pathlib import Path
from copy import deepcopy
import pytest

from gnpy.core.exceptions import DisjunctionError, ServiceError
from gnpy.core.elements import Fiber, Edfa, Fused
from gnpy.core.equipment import automatic_nch
from gnpy.tools.json_io import (load_equipment, load_network, requests_from_json, disjunctions_from_json,
                                load_requests)
from gnpy.core.network import build_network
from gnpy.core.utils import lin2db
from gnpy.topology.request import (compute_path_dsjctn, ResultElement, deduplicate_disjunctions,
                                   compute_path_with_disjunction, compute_constrained_path, requests_aggregation)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILENAME = DATA_DIR / 'testTopology_auto_design_expected.json'
SERVICE_FILENAME = DATA_DIR / 'testTopology_services_expected.json'
equipment = load_equipment(EQPT_FILENAME)


@pytest.fixture()
def setup():
    """ common setup for tests: builds network, equipment and oms only once
    """
    network = load_network(NETWORK_FILENAME, equipment)
    default = equipment['SI']['default']
    p_db = default.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(default.f_min, default.f_max, default.spacing))
    build_network(network, equipment, p_db, p_total_db)
    oms_list = build_oms_list(network, equipment)
    return network, oms_list


def test_strings_and_json(setup):
    """ test that __str and __repr, to_json work. No need to test the printing layout only if it does trigger
    an error or not. try to explore all combinations
    """
    network, oms_list = setup
    data = load_requests(SERVICE_FILENAME, equipment, bidir=False, network=network, network_filename=NETWORK_FILENAME)
    requests = requests_from_json(data, equipment)
    print(requests[0].__repr__)
    print(str(requests[0]))
    requests[1].tsp_mode = None
    requests[1].baud_rate = None
    requests[4].bidir = True
    print(requests[1].__repr__)
    print(str(requests[1]))
    dsjn = disjunctions_from_json(data)
    print(dsjn[0].__repr__)
    print(str(dsjn[0]))
    dsjn = deduplicate_disjunctions(dsjn)
    paths = compute_path_dsjctn(network, equipment, requests, dsjn)
    propagated_paths, reversed_paths, reversed_propagated_paths = \
        compute_path_with_disjunction(network, equipment, requests, paths)
    pth_assign_spectrum(paths, requests, oms_list, reversed_paths)
    for path, reversed_path, request in zip(propagated_paths, reversed_propagated_paths, requests):
        print(ResultElement(request, path, reversed_path).json)
        print(ResultElement(request, path).__repr__)
        print(str(ResultElement(request, path)))
        print(ResultElement(request, path, reversed_path))

    requests[2].N = None
    requests[2].M = None
    with pytest.raises(ServiceError):
        print(ResultElement(requests[2], propagated_paths[2], reversed_paths[2]).detailed_path_json)

    requests[3].blocking_reason = 'NO_PATH'
    with pytest.raises(ServiceError):
        print(ResultElement(requests[3], propagated_paths[3], reversed_paths[3]).detailed_path_json)


class Req:
    """ Class to create limited Requests objects
    """
    def __init__(self, request_id, nodes_list, destination):
        self.request_id = request_id
        self.nodes_list = nodes_list
        self.destination = destination


def test_destination():
    """Test if code raises the exception correctly in case of the
    destination is not the last element on the nodes list.
    """
    network = []
    req = Req(1, ['a', 'b', 'c'], 'b')
    with pytest.raises(ValueError):
        compute_constrained_path(network, req)


def load_service(xls_input):
    """ common functions to the following tests
    """
    network = load_network(xls_input, equipment)
    default = equipment['SI']['default']
    p_db = default .power_dbm
    p_total_db = p_db + lin2db(automatic_nch(default.f_min, default.f_max, default.spacing))
    build_network(network, equipment, p_db, p_total_db)
    _ = build_oms_list(network, equipment)
    data = load_requests(xls_input, equipment, bidir=False, network=network, network_filename=xls_input)
    requests = requests_from_json(data, equipment)
    disjunctions = deduplicate_disjunctions(disjunctions_from_json(data))
    requests, disjunctions = requests_aggregation(requests, disjunctions)
    return network, requests, disjunctions


@pytest.mark.parametrize('xls_input', (DATA_DIR / 'test_disjunction.xls',))
def test_compare_reqs(xls_input):
    """Test if code correctly aggregates duplicate disjunctions
    in case of aggregation of duplicate requests; check that multiple disjunctions are correctly
    grouped ; check that almost but non identical requests are not agregated
    """
    network, requests, disjunctions = load_service(xls_input)
    assert requests[5].request_id == '6 | 5'
    assert disjunctions[4].disjunctions_req[1] == '6 | 5'
    assert disjunctions[5].disjunctions_req == ['7', '8']
    assert disjunctions[6].disjunctions_req == ['8', '0', '7']

    # check that almost but non identical requests are not agregated
    data = load_requests(xls_input, equipment, bidir=False, network=network, network_filename=xls_input)
    requests_before_agregation = requests_from_json(data, equipment)
    requests_id = [e.request_id for e in requests]
    for req in requests_before_agregation:
        if req.request_id not in ['6', '5']:
            assert req.request_id in requests_id


@pytest.mark.parametrize('xls_input', (DATA_DIR / 'test_disjunction.xls',))
def test_unfeasible_disjunction(xls_input):
    """ This file contains an impossible set of disjunstions:
    Check if code raises a DisjunctionError correctly at the end of step 5
    on request.py::compute_path_dsjctn.
    """
    network, requests, disjunctions = load_service(xls_input)
    with pytest.raises(DisjunctionError):
        _ = compute_path_dsjctn(network, equipment, requests, disjunctions)


@pytest.mark.parametrize('xls_input', (DATA_DIR / 'testTopology.xls',))
def test_unfeasible_requests(xls_input):
    """Test code behaviour in case of unfeasible requests (source and destination not connected,
    no mode available for the given spacing, specified mode unfeasible, no mode feasible).
    """
    network, requests, disjunctions = load_service(xls_input)
    # force unfeasible cases by changing request
    # lannion and a are not connected in the topology, so should issue NO_PATH and empty list of paths
    requests[0].destination = 'trx Lannion_CAS'
    requests[0].source = 'trx a'
    paths = compute_path_dsjctn(network, equipment, requests, disjunctions)
    # no mode can fit in 30e9: no mode is propagated
    # if no baud_rate is set, means that user did not specified one in the request
    requests[1].baud_rate = None
    requests[1].spacing = 30.0e9
    # mode cannot satisfy the requirement, mode is propagaated
    requests[2].OSNR = 100.0
    # create a fake transceiver with really high OSNR to raise NO_FEASIBLE_MODE case. algorithm will
    # explore all modes starting with max baud_rate, and max bit_rate, so last explored mode is the one
    # with lowest bit_rate and baud_rate
    requests[3].baud_rate = None
    requests[3].tsp = 'fake_type'    # based on 'vendorA_trx-type1'
    fake_modes = [{"format": "fake 4",
                   "baud_rate": 32e9,
                   "bit_rate": 100e9,
                   "OSNR": 90
                  },
                  {"format": "fake 2",
                   "baud_rate": 66e9,
                   "bit_rate": 200e9,
                   "OSNR": 100
                  },
                  {"format": "fake 3",
                   "baud_rate": 32e9,
                   "bit_rate": 200e9,
                   "OSNR": 100
                  },
                  {"format": "fake 1",
                   "baud_rate": 66e9,
                   "bit_rate": 400e9,
                   "OSNR": 100
                  }]
    equipment['Transceiver']['fake_type'] = deepcopy(equipment['Transceiver']['vendorA_trx-type1'])
    for i in range(len(equipment['Transceiver']['fake_type'].mode)):
        for key in {'format', 'baud_rate', 'OSNR', 'bit_rate'}:
            equipment['Transceiver']['fake_type'].mode[i][key] = fake_modes[i][key]
    propagated_paths, _, _ = \
        compute_path_with_disjunction(network, equipment, requests, paths)
    # differences from results without 'forced modes'
    assert len(propagated_paths[0]) == 0
    assert requests[0].blocking_reason == 'NO_PATH'
    assert len(propagated_paths[1]) == 0
    assert requests[1].blocking_reason == 'NO_FEASIBLE_BAUDRATE_WITH_SPACING'
    assert len(propagated_paths[2]) != 0
    assert requests[2].blocking_reason == 'MODE_NOT_FEASIBLE'
    assert len(propagated_paths[2]) != 0
    assert requests[3].blocking_reason == 'NO_FEASIBLE_MODE'
    assert requests[3].tsp_mode == 'fake 4'


@pytest.mark.parametrize('xls_input', (DATA_DIR / 'test_path_route_constraint.xls',))
def test_compute_route_constraint(xls_input):
    """Test code behaviour in case of route constraint (loose and strict).
    """
    network, requests, disjunctions = load_service(xls_input)
    with pytest.raises(DisjunctionError):
        _ = compute_path_dsjctn(network, equipment, requests, disjunctions)


@pytest.mark.parametrize('xls_input', (DATA_DIR / 'testTopology.xls',))
def test_reversed(xls_input):
    """ Test that reversed path is really reversed from path
    """
    network, requests, disjunctions = load_service(xls_input)
    paths = compute_path_dsjctn(network, equipment, requests, disjunctions)
    _, reversed_paths, _ = \
        compute_path_with_disjunction(network, equipment, requests, paths)
    test_path = [e.uid for e in paths[0]]
    reversed_test_path = [e.uid for e in reversed_paths[0]]
    assert test_path == ['trx Lorient_KMA', 'roadm Lorient_KMA', 'east edfa in Lorient_KMA to Vannes_KBE',
                         'fiber (Lorient_KMA → Vannes_KBE)-F055', 'west edfa in Vannes_KBE to Lorient_KMA',
                         'roadm Vannes_KBE', 'trx Vannes_KBE']
    assert reversed_test_path == ['trx Vannes_KBE', 'roadm Vannes_KBE', 'east edfa in Vannes_KBE to Lorient_KMA',
                                  'fiber (Vannes_KBE → Lorient_KMA)-F055', 'west edfa in Lorient_KMA to Vannes_KBE',
                                  'roadm Lorient_KMA', 'trx Lorient_KMA']
    # check that oms in each path has its reversed oms in the reverse path
    for path, reversed_path in zip(paths, reversed_paths):
        omses = [e.oms for e in path if isinstance(e, (Fiber, Edfa, Fused))]
        reversed_omses = [e.oms_id for e in reversed_path if isinstance(e, (Fiber, Edfa, Fused))]
        for oms in omses:
            assert oms.reversed_oms.oms_id in reversed_omses
            # check that reversed oms from reversed oms is the oms itself
            assert oms.reversed_oms.reversed_oms == oms

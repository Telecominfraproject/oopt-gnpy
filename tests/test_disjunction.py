#!/usr/bin/env python3
# Module name : test_disjunction.py
# Version:
# License: BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
checks that computed paths are disjoint as specified in the json service file
that computed paths do not loop
that include node constraints are correctly taken into account
"""

from pathlib import Path
import pytest
from gnpy.core.equipment import trx_mode_params
from gnpy.core.network import build_network
from gnpy.core.exceptions import ServiceError
from gnpy.core.utils import automatic_nch, lin2db
from gnpy.core.elements import Roadm
from gnpy.topology.request import (compute_path_dsjctn, isdisjoint, find_reversed_path, PathRequest,
                                   correct_json_route_list)
from gnpy.topology.spectrum_assignment import build_oms_list
from gnpy.tools.json_io import requests_from_json, load_requests, load_network, load_equipment, disjunctions_from_json

NETWORK_FILE_NAME = Path(__file__).parent.parent / 'tests/data/testTopology_expected.json'
SERVICE_FILE_NAME = Path(__file__).parent.parent / 'tests/data/testTopology_testservices.json'
RESULT_FILE_NAME = Path(__file__).parent.parent / 'tests/data/testTopology_testresults.json'
EQPT_LIBRARY_NAME = Path(__file__).parent.parent / 'tests/data/eqpt_config.json'


@pytest.fixture()
def serv(test_setup):
    """ common setup for service list
    """
    network, equipment = test_setup
    data = load_requests(SERVICE_FILE_NAME, equipment, bidir=False, network=network, network_filename=NETWORK_FILE_NAME)
    rqs = requests_from_json(data, equipment)
    rqs = correct_json_route_list(network, rqs)
    dsjn = disjunctions_from_json(data)
    return network, equipment, rqs, dsjn


@pytest.fixture()
def test_setup():
    """ common setup for tests: builds network, equipment and oms only once
    """
    equipment = load_equipment(EQPT_LIBRARY_NAME)
    network = load_network(NETWORK_FILE_NAME, equipment)
    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    build_oms_list(network, equipment)

    return network, equipment


def test_disjunction(serv):
    """ service_file contains sevaral combination of disjunction constraint. The test checks
        that computed paths with disjunction constraint are effectively disjoint
    """
    network, equipment, rqs, dsjn = serv
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    print(dsjn)

    dsjn_list = [d.disjunctions_req for d in dsjn]

    # assumes only pairs in dsjn list
    test = True
    for e in dsjn_list:
        rqs_id_list = [r.request_id for r in rqs]
        p1 = pths[rqs_id_list.index(e[0])][1:-1]
        p2 = pths[rqs_id_list.index(e[1])][1:-1]
        if isdisjoint(p1, p2) + isdisjoint(p1, find_reversed_path(p2)) > 0:
            test = False
            print(f'Computed path (roadms):{[e.uid for e in p1  if isinstance(e, Roadm)]}\n')
            print(f'Computed path (roadms):{[e.uid for e in p2  if isinstance(e, Roadm)]}\n')
            break
    print(dsjn_list)
    assert test


def test_does_not_loop_back(serv):
    """ check that computed paths do not loop back ie each element appears only once
    """
    network, equipment, rqs, dsjn = serv
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    test = True
    for p in pths:
        for el in p:
            p.remove(el)
            a = [e for e in p if e.uid == el.uid]
            if a:
                test = False
                break
    assert test

    # TODO : test that identical requests are correctly agregated
    # and reproduce disjunction vector as well as route constraints
    # check that requests with different parameters are not aggregated
    # check that the total agregated bandwidth is the same after aggregation
    #


def create_rq(equipment, srce, dest, bdir, nd_list, ls_list):
    """ create the usual request list according to parameters
    """
    requests_list = []
    params = {}
    params['request_id'] = 'test_request'
    params['source'] = srce
    params['bidir'] = bdir
    params['destination'] = dest
    params['trx_type'] = 'Voyager'
    params['trx_mode'] = 'mode 1'
    params['format'] = params['trx_mode']
    params['spacing'] = 50000000000.0
    params['nodes_list'] = nd_list
    params['loose_list'] = ls_list
    trx_params = trx_mode_params(equipment, params['trx_type'], params['trx_mode'], True)
    params.update(trx_params)
    params['power'] = 1.0
    f_min = params['f_min']
    f_max_from_si = params['f_max']
    params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
    params['path_bandwidth'] = 100000000000.0
    requests_list.append(PathRequest(**params))
    return requests_list


@pytest.mark.parametrize('srce, dest, result, pth, nd_list, ls_list', [
    ['a', 'trx h', 'fail', 'no_path', [], []],
    ['trx a', 'h', 'fail', 'no_path', [], []],
    ['trx a', 'trx h', 'pass', 'found_path', [], []],
    ['trx a', 'trx h', 'pass', 'found_path', ['roadm b', 'roadm a'], ['LOOSE', 'LOOSE']],
    ['trx a', 'trx h', 'pass', 'no_path', ['roadm b', 'roadm a'], ['STRICT', 'STRICT']],
    ['trx a', 'trx h', 'pass', 'found_path', ['roadm b', 'roadm c'], ['STRICT', 'STRICT']],
    ['trx a', 'trx h', 'fail', 'no_path', ['Lorient_KMA', 'roadm c'], ['STRICT', 'STRICT']],
    ['trx a', 'trx h', 'pass', 'no_path', ['roadm Lorient_KMA', 'roadm c'], ['LOOSE', 'STRICT']],
    ['trx a', 'trx h', 'pass', 'found_path', ['roadm c', 'roadm c'], ['LOOSE', 'LOOSE']],
    ['trx a', 'trx h', 'pass', 'found_path', ['roadm c', 'roadm c'], ['STRICT', 'STRICT']],
    ['trx a', 'trx h', 'pass', 'found_path', ['roadm c', 'roadm g'], ['STRICT', 'STRICT']],
    ['trx a', 'trx h', 'pass', 'found_path', ['trx a', 'roadm g'], ['STRICT', 'STRICT']],
    ['trx a', 'trx h', 'pass', 'found_path', ['trx h'], ['STRICT']],
    ['trx a', 'trx h', 'pass', 'found_path', ['roadm a'], ['STRICT']]])
def test_include_constraints(test_setup, srce, dest, result, pth, nd_list, ls_list):
    """ check that all combinations of constraints are correctly handled:
        - STRICT/LOOSE
        - correct names/incorrect names -> pass/fail
        - possible include/impossible include
        if incorrect name -> fail
        else:
                                      constraint    |one or more STRICT | all LOOSE
            ----------------------------------------------------------------------------------
            >1 path from s to d | can be applied    | found_path        | found_path
                                | cannot be applied | no_path           | found_path
            ----------------------------------------------------------------------------------
            0                   |                   |          computation stops
    """
    network, equipment = test_setup
    dsjn = []
    bdir = False
    rqs = create_rq(equipment, srce, dest, bdir, nd_list, ls_list)
    print(rqs)
    if result == 'fail':
        with pytest.raises(ServiceError):
            rqs = correct_json_route_list(network, rqs)
    else:
        rqs = correct_json_route_list(network, rqs)
        pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
        # if loose, one path can be returned
        if pths[0]:
            assert pth == 'found_path'
        else:
            assert pth == 'no_path'

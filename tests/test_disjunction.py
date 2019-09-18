#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
# Module name : test_disjunction.py
# Version : 
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
checks that computed paths are disjoint as specified in the json service file
"""

from pathlib import Path
import pytest
from gnpy.core.equipment import load_equipment, trx_mode_params, automatic_nch
from gnpy.core.network import load_network, build_network
from examples.path_requests_run import (requests_from_json , correct_route_list ,
                                        load_requests , disjunctions_from_json)
from gnpy.core.request import compute_path_dsjctn, isdisjoint , find_reversed_path
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.elements import Roadm
from gnpy.core.spectrum_assignment import build_oms_list

network_file_name = Path(__file__).parent.parent / 'tests/data/testTopology_expected.json'
service_file_name = Path(__file__).parent.parent / 'tests/data/testTopology_testservices.json'
result_file_name  = Path(__file__).parent.parent / 'tests/data/testTopology_testresults.json'
eqpt_library_name = Path(__file__).parent.parent / 'tests/data/eqpt_config.json'

@pytest.mark.parametrize("net",[network_file_name])
@pytest.mark.parametrize("eqpt", [eqpt_library_name])
@pytest.mark.parametrize("serv",[service_file_name])
def test_disjunction(net,eqpt,serv):
    data = load_requests(serv, eqpt, bidir=False)
    equipment = load_equipment(eqpt)
    network = load_network(net,equipment)
    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max 
    p_db = equipment['SI']['default'].power_dbm
    
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    build_oms_list(network, equipment)

    rqs = requests_from_json(data, equipment)
    rqs = correct_route_list(network, rqs)
    dsjn = disjunctions_from_json(data)
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    print(dsjn)

    dsjn_list = [d.disjunctions_req for d in dsjn ]

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

@pytest.mark.parametrize("net",[network_file_name])
@pytest.mark.parametrize("eqpt", [eqpt_library_name])
@pytest.mark.parametrize("serv",[service_file_name])
def test_does_not_loop_back(net,eqpt,serv):
    data = load_requests(serv, eqpt, bidir=False)
    equipment = load_equipment(eqpt)
    network = load_network(net,equipment)

    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max 
    p_db = equipment['SI']['default'].power_dbm
    
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    build_oms_list(network, equipment)

    rqs = requests_from_json(data, equipment)
    rqs = correct_route_list(network, rqs)
    dsjn = disjunctions_from_json(data)
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)

    # check that computed paths do not loop back ie each element appears only once
    test = True
    for p in pths :
        for el in p:
            p.remove(el)
            a = [e for e in p if e.uid == el.uid]
            if a :
                test = False
                break

    assert test        

    # TODO : test that identical requests are correctly agregated 
    # and reproduce disjunction vector as well as route constraints
    # check that requests with different parameters are not aggregated

    # check that the total agregated bandwidth is the same after aggregation

    #

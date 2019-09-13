#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
# Module name : test_automaticmodefeature.py
# Version : 
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
checks that empty info on mode, power, nbchannel in service file are supported
    uses combination of [mode, pow, nb channel] filled or empty defined in meshTopologyToy_services.json
    leading to feasible path or not, and check the propagate and propagate_and_optimize_mode
    return the expected based on a reference toy example

"""

from pathlib import Path
import pytest
from gnpy.core.equipment import load_equipment, trx_mode_params, automatic_nch
from gnpy.core.network import load_network, build_network
from examples.path_requests_run import requests_from_json, correct_route_list, load_requests
from gnpy.core.request import compute_path_dsjctn, propagate, propagate_and_optimize_mode
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.elements import Roadm

network_file_name = Path(__file__).parent.parent / 'tests/data/testTopology_expected.json'
service_file_name = Path(__file__).parent.parent / 'tests/data/testTopology_testservices.json'
result_file_name  = Path(__file__).parent.parent / 'tests/data/testTopology_testresults.json'
eqpt_library_name = Path(__file__).parent.parent / 'tests/data/eqpt_config.json'

@pytest.mark.parametrize("net",[network_file_name])
@pytest.mark.parametrize("eqpt", [eqpt_library_name])
@pytest.mark.parametrize("serv",[service_file_name])
@pytest.mark.parametrize("expected_mode",[['16QAM', 'PS_SP64_1', 'PS_SP64_1', 'PS_SP64_1', 'mode 2 - fake', 'mode 2', 'PS_SP64_1', 'mode 3', 'PS_SP64_1', 'PS_SP64_1', '16QAM', 'mode 1', 'PS_SP64_1', 'PS_SP64_1', 'mode 1', 'mode 2', 'mode 1', 'mode 2', 'nok']])
def test_automaticmodefeature(net,eqpt,serv,expected_mode):
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

    rqs = requests_from_json(data, equipment)
    rqs = correct_route_list(network, rqs)
    dsjn = []
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    path_res_list = []

    for i,pathreq in enumerate(rqs):

        # use the power specified in requests but might be different from the one specified for design
        # the power is an optional parameter for requests definition
        # if optional, use the one defines in eqt_config.json
        p_db = lin2db(pathreq.power*1e3)
        p_total_db = p_db + lin2db(pathreq.nb_channel)
        print(f'request {pathreq.request_id}')
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        print(f'with path constraint: {[pathreq.source]+pathreq.nodes_list}') #adding first node to be clearer on the output

        total_path = pths[i]
        print(f'Computed path (roadms):{[e.uid for e in total_path  if isinstance(e, Roadm)]}\n')
        # for debug
        # print(f'{pathreq.baud_rate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        if pathreq.baud_rate is not None:
            print(pathreq.format)
            path_res_list.append(pathreq.format)
            total_path = propagate(total_path,pathreq,equipment)
        else:
            total_path,mode = propagate_and_optimize_mode(total_path,pathreq,equipment)
            # if no baudrate satisfies spacing, no mode is returned and an empty path is returned
            # a warning is shown in the propagate_and_optimize_mode
            if mode is not None :
                print (mode['format'])
                path_res_list.append(mode['format'])
            else :
                print('nok')
                path_res_list.append('nok')
    print(path_res_list)
    assert path_res_list==expected_mode


#!/usr/bin/env python3
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
from logging import INFO
from numpy.testing import assert_allclose
import pytest

from gnpy.core.network import build_network
from gnpy.core.utils import automatic_nch, lin2db, watt2dbm
from gnpy.core.elements import Roadm
from gnpy.topology.request import compute_path_dsjctn, propagate, propagate_and_optimize_mode, correct_json_route_list
from gnpy.tools.json_io import load_network, load_equipment, requests_from_json, load_requests, load_json, \
    _equipment_from_json


data_dir = Path(__file__).parent.parent / 'tests/data'
network_file_name = data_dir / 'testTopology_expected.json'
service_file_name = data_dir / 'testTopology_testservices.json'
result_file_name = data_dir / 'testTopology_testresults.json'
eqpt_library_name = data_dir / 'eqpt_config.json'
extra_configs = {"std_medium_gain_advanced_config.json": data_dir / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": data_dir / "Juniper-BoosterHG.json"}


@pytest.mark.parametrize("net", [network_file_name])
@pytest.mark.parametrize("eqpt", [eqpt_library_name])
@pytest.mark.parametrize("serv", [service_file_name])
@pytest.mark.parametrize("expected_mode", [['16QAM', 'PS_SP64_1', 'PS_SP64_1', 'PS_SP64_1', 'mode 2 - fake', 'mode 2',
                                            'PS_SP64_1', 'mode 3', 'PS_SP64_1', 'PS_SP64_1', '16QAM', 'mode 1',
                                            'PS_SP64_1', 'PS_SP64_1', 'mode 1', 'mode 2', 'mode 1', 'mode 2', 'nok']])
def test_automaticmodefeature(net, eqpt, serv, expected_mode):
    equipment = load_equipment(eqpt, extra_configs)
    network = load_network(net, equipment)
    data = load_requests(serv, eqpt, bidir=False, network=network, network_filename=net)

    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)

    rqs = requests_from_json(data, equipment)
    rqs = correct_json_route_list(network, rqs)
    dsjn = []
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    path_res_list = []

    for i, pathreq in enumerate(rqs):

        # use the power specified in requests but might be different from the one specified for design
        # the power is an optional parameter for requests definition
        # if optional, use the one defines in eqt_config.json
        p_db = lin2db(pathreq.power * 1e3)
        p_total_db = p_db + lin2db(pathreq.nb_channel)
        print(f'request {pathreq.request_id}')
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        # adding first node to be clearer on the output
        print(f'with path constraint: {[pathreq.source]+pathreq.nodes_list}')

        total_path = pths[i]
        print(f'Computed path (roadms):{[e.uid for e in total_path  if isinstance(e, Roadm)]}\n')
        # for debug
        # print(f'{pathreq.baud_rate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        if pathreq.baud_rate is not None:
            print(pathreq.format)
            path_res_list.append(pathreq.format)
            total_path = propagate(total_path, pathreq, equipment)
        else:
            total_path, mode = propagate_and_optimize_mode(total_path, pathreq, equipment)
            # if no baudrate satisfies spacing, no mode is returned and an empty path is returned
            # a warning is shown in the propagate_and_optimize_mode
            if mode is not None:
                print(mode['format'])
                path_res_list.append(mode['format'])
            else:
                print('nok')
                path_res_list.append('nok')
    print(path_res_list)
    assert path_res_list == expected_mode


def test_propagate_and_optimize_mode(caplog):
    """Checks that the automatic mode returns the last explored mode

    Mode are explored with descending baud_rate order and descending bitrate, so the last explored mode must be mode 1
    Mode 1 GSNR is OK but pdl penalty are not OK due to high ROADM PDL. so the last explored mode is not OK
    Then the propagate_and_optimize_mode must return mode 1 and the blocking reason must be 'NO_FEASIBLE_MODE'
    """
    caplog.set_level(INFO)
    json_data = load_json(eqpt_library_name)
    voyager = next(e for e in json_data['Transceiver'] if e['type_variety'] == 'Voyager')
    # expected path min GSNR is 22.11
    # Change Voyager modes so that:
    # - highest baud rate has min OSNR > path GSNR
    # - lower baudrate with highest bitrate has min OSNR > path GSNR
    # - lower baudrate with lower bitrate has has min OSNR < path GSNR but PDL penalty is infinite
    voyager['mode'] = [
        {
            "format": "mode 1",
            "baud_rate": 32e9,
            "OSNR": 12,
            "bit_rate": 100e9,
            "roll_off": 0.15,
            "tx_osnr": 45,
            "min_spacing": 50e9,
            "penalties": [
                {
                    "chromatic_dispersion": 4e3,
                    "penalty_value": 0
                }, {
                    "chromatic_dispersion": 40e3,
                    "penalty_value": 0
                }, {
                    "pdl": 0.5,
                    "penalty_value": 1
                }, {
                    "pmd": 30,
                    "penalty_value": 0
                }],
            "cost": 1
        },
        {
            "format": "mode 3",
            "baud_rate": 32e9,
            "OSNR": 30,
            "bit_rate": 300e9,
            "roll_off": 0.15,
            "tx_osnr": 45,
            "min_spacing": 50e9,
            "cost": 1
        },
        {
            "format": "mode 2",
            "baud_rate": 66e9,
            "OSNR": 25,
            "bit_rate": 400e9,
            "roll_off": 0.15,
            "tx_osnr": 45,
            "min_spacing": 75e9,
            "cost": 1
        }]
    # change default ROADM PDL so that crossing 2 ROADMs leasd to inifinte penalty for mode 1
    eqpt_roadm = next(r for r in json_data['Roadm'] if 'type_variety' not in r)
    eqpt_roadm['pdl'] = 0.5
    equipment = _equipment_from_json(json_data, extra_configs)
    network = load_network(network_file_name, equipment)
    data = load_requests(filename=Path(__file__).parent.parent / 'tests/data/testTopology_services_expected.json',
                         eqpt=eqpt_library_name, bidir=False, network=network, network_filename=network_file_name)
    # remove the mode from request, change it to larger spacing
    data['path-request'][1]['path-constraints']['te-bandwidth']['trx_mode'] = None
    data['path-request'][1]['path-constraints']['te-bandwidth']['spacing'] = 75e9
    assert_allclose(watt2dbm(data['path-request'][1]['path-constraints']['te-bandwidth']['output-power']), 1, rtol=1e-9)
    # use the request power for design, or there will be inconsistencies with the gain
    build_network(network, equipment, 1, 21)

    rqs = requests_from_json(data, equipment)
    rqs = correct_json_route_list(network, rqs)
    [path] = compute_path_dsjctn(network, equipment, [rqs[1]], [])
    total_path, mode = propagate_and_optimize_mode(path, rqs[1], equipment)
    assert round(min(path[-1].snr_01nm), 2) == 22.22
    assert mode['format'] == 'mode 1'
    assert rqs[1].blocking_reason == 'NO_FEASIBLE_MODE'
    expected_mesg = '\tWarning! Request 1: no mode satisfies path SNR requirement.'
    # Last log records mustcontain the message about the las explored mode
    assert expected_mesg in caplog.records[-1].message

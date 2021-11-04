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
import pytest
from gnpy.core.network import build_network
from gnpy.core.utils import automatic_nch, lin2db
from gnpy.topology.request import compute_path_dsjctn, propagate, propagate_and_optimize_mode, correct_json_route_list
from gnpy.tools.json_io import load_network, load_equipment, requests_from_json, load_requests


PARENT_PATH = Path(__file__).parent.parent
NETWORK_FILE_NAME = PARENT_PATH / 'tests/data/testTopology_expected.json'
SERVICE_FILE_NAME = PARENT_PATH / 'tests/data/testTopology_testservices.json'
EQPT_LIBRARY_NAME = PARENT_PATH / 'tests/data/eqpt_config.json'


@pytest.fixture()
def test_network():
    """ Creates simulation set up
    """
    equipment = load_equipment(EQPT_LIBRARY_NAME)
    network = load_network(NETWORK_FILE_NAME, equipment)
    data = load_requests(SERVICE_FILE_NAME, EQPT_LIBRARY_NAME, bidir=False, network=network,
                         network_filename=NETWORK_FILE_NAME)

    # Build the network once using the default power defined in SI in eqpt config
    default = equipment['SI']['default']
    power_total_db = default.power_dbm + lin2db(automatic_nch(default.f_min,
                                                              default.f_max,
                                                              default.spacing))
    build_network(network, equipment, default.power_dbm, power_total_db)
    requests = requests_from_json(data, equipment)
    requests = correct_json_route_list(network, requests)
    paths = compute_path_dsjctn(network, equipment, requests, disjunctions_list=[])
    return requests, paths, equipment


@pytest.mark.parametrize("expected_mode", [['16QAM', 'PS_SP64_1', 'PS_SP64_1', \
                                            'PS_SP64_1', 'mode 2 - fake', 'mode 2', \
                                            'PS_SP64_1', 'mode 3', 'PS_SP64_1', \
                                            'PS_SP64_1', '16QAM', 'mode 1', \
                                            'PS_SP64_1', 'PS_SP64_1', 'mode 1', \
                                            'mode 2', 'mode 1', 'mode 2', 'nok']])
def test_automaticmodefeature(expected_mode, test_network):
    """ Check that each request produces the expected mode
    """
    requests, paths, equipment = test_network
    path_request_list = []

    for i, pathreq in enumerate(requests):
        total_path = paths[i]
        if pathreq.baud_rate is not None:
            path_request_list.append(pathreq.format)
            total_path = propagate(total_path, pathreq, equipment)
        else:
            total_path, mode = propagate_and_optimize_mode(total_path, pathreq, equipment)
            # if no baudrate satisfies spacing, no mode is returned and an empty path is returned
            # a warning is shown in the propagate_and_optimize_mode
            if mode is not None:
                path_request_list.append(mode['format'])
            else:
                path_request_list.append('nok')
    assert path_request_list == expected_mode


@pytest.mark.parametrize('baud_rate, spacing, min_osnr, expected_mode',
                         [(39e9, 50e9, 100, 'd'),    # 'd' has highest baud_rate and is feasible
                          (39e9, 75e9, 100, 'b'),    # 'd' does not fit in spacing, no feasible modes
                          (32e9, 50e9, 23.5, 'a')])  # 'd' provides more margin, 'a' has a better spectral efficiency
def test_path_low_osnr(test_network, baud_rate, spacing, min_osnr, expected_mode):
    """ if no mode work the function returns the last explored mode, else it returns the
    mode with the highest baudrate. If 2 modes have same rate, it selects the one with the highest
    bitrate"""
    requests, paths, equipment = test_network
    # Condition to achieve path's OSNR lower than mode's
    path_request = requests[0]
    total_path = paths[0]
    modes = [{
        'format': 'a',
        'baud_rate': 32e9,
        'OSNR': min_osnr,
        'bit_rate': 250e9,
        'roll_off': 0.15,
        'tx_osnr': 100,
        'min_spacing': 50e9,
        'cost': 1
        },
        {
        'format': 'b',
        'baud_rate': 32e9,
        'OSNR': 50,
        'bit_rate': 100e9,
        'roll_off': 0.15,
        'tx_osnr': 100,
        'min_spacing': 50e9,
        'cost': 1
        },
        {
        'format': 'd',
        'baud_rate': baud_rate,
        'OSNR': 20,
        'bit_rate': 200e9,
        'roll_off': 0.15,
        'tx_osnr': 100,
        'min_spacing': spacing,
        'cost': 1
        }]
    equipment['Transceiver']['Voyager_16QAM'].mode = modes
    total_path, mode = propagate_and_optimize_mode(total_path,
                                                   path_request,
                                                   equipment)
    assert mode['format'] == expected_mode

# SPDX-License-Identifier: BSD-3-Clause
#
# Reading and writing JSON files for GNPy
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#
""" Tests for regeneration feature
    - parsing regen in the network topology
    - parsing regen in a route list w and wo hop attribute
    - correct interpretation of the regen according to service
    - propagation result with regen identical to the last section
    - if regeneration is introduced as a preference provides a path
      with regeneration if path is not feasible otherwise
"""
from pathlib import Path
from os import unlink
import pytest
from numpy import mean
from gnpy.core.network import build_network
from gnpy.core.elements import Regenerator, Transceiver
from gnpy.core.equipment import trx_mode_params
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.core.exceptions import ServiceError
from gnpy.tools.json_io import load_equipment, load_network, requests_from_json, save_network, load_json
from gnpy.topology.request import (correct_json_route_list, correct_json_regen_list, compute_path_dsjctn,
                                   restore_regen_in_path, compute_path_with_disjunction, remove_regen_from_list)
from gnpy.topology.spectrum_assignment import build_oms_list

from tests.compare import compare_networks


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILENAME = DATA_DIR / 'threehops_regen.json'
SERVICE_FILENAME = DATA_DIR / 'regen_services.json'


@pytest.fixture()
def equipment():
    """common setup for tests: builds equipment only once
    """
    equipment = load_equipment(EQPT_FILENAME)
    # reduce the number of channels to speed up tests (propagation is taking ages for the full spectrum)
    for elem in equipment['Transceiver']:
        equipment['Transceiver'][elem].frequency['max'] = 191.9e12
    return equipment


@pytest.fixture()
def setup(equipment):
    """ common setup for tests: builds network and oms only once
    """
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    build_oms_list(network, equipment)
    return network


# parsing tests: network reading and creation
def test_network_parsing(tmpdir, setup):
    """test that autodesign creates same file as the input file (already autodesigned)
    and correctly reproduces regenerators in the autodesign
    """
    network = setup
    actual_json_output = tmpdir / 'threehops_regen_auto_design.json'
    print(actual_json_output)
    save_network(network, actual_json_output)
    actual = load_json(actual_json_output)
    # unlink(actual_json_output)
    expected = load_json(NETWORK_FILENAME)

    results = compare_networks(expected, actual)
    assert not results.elements.missing
    assert not results.elements.extra
    assert not results.elements.different
    assert not results.connections.missing
    assert not results.connections.extra
    assert not results.connections.different


# parsing tests
# reading regenerators in service json. check that a regen is correctly read out of a path constraint
# case where roadm in which the roadm is placed is also listed in the route
# case where no type is specified -> select the same transcever type from source by default
#             and source mode is specified -> select the same transcever mode from source by default
#             and source mode is not specified -> leave empty for decision
# case where the regen type is specified and the same as the source
#             and mode is specified and the same as the source
#             and mode is specified and not the same as the source
#                         and min spacing compatible with request
#                         and min spacing not compatible with request
#             and mode is not specified -> leave empty for decision
# case where the regen type is specified and different from the source
#             and mode is specified
#             and mode is not specified -> leave empty for decision
# case of several regeneration sections
# TODO case where regeneration is set as a preference

# exhaustive use case generation of all combinations
def json_data(equipment, source='trx node A', destination='trx node C'):
    """ comman data structure for the service
    """
    json_data = {
        'path-request': [{
            'request-id': '0',
            'source': source,
            'destination': destination,
            'src-tp-id': source,
            'dst-tp-id': destination,
            'bidirectional': False,
            'path-constraints': {
                'te-bandwidth': {
                    'technology': 'flexi-grid',
                    'trx_type': 'Voyager',
                    'trx_mode': 'mode 1',
                    'effective-freq-slot': [
                        {
                            'N': None,
                            'M': None
                        }
                    ],
                    'spacing': 50e9,
                    'path_bandwidth': 100e9
                }
            },
            'explicit-route-objects': {
                'route-object-include-exclude': []
            }}]}
    return json_data


def json_request(source, destination, bidir, regen_nodes, route_prev_nodes, regen_types, regen_modes, spacings):
    """ common function to create the request dict
    """
    json = json_data(equipment)['path-request'][0]
    json['source'], json['src-tp-id'], json['destination'], json['dst-tp-id'], json['bidirectional'] = \
        source, source, destination, destination, bidir
    i = 0
    for node, prev_node, regen_type, regen_mode, spacing in \
            zip(regen_nodes, route_prev_nodes, regen_types, regen_modes, spacings):
        if prev_node != '':
            hop = {
                'explicit-route-usage': 'route-include-ero',
                'index': i,
                'num-unnum-hop': {
                    'node-id': prev_node,
                    'link-tp-id': 'link-tp-id is not used',
                    'hop-type': 'STRICT'
                    }
                }
            json['explicit-route-objects']['route-object-include-exclude'].append(hop)
            i = i + 1
        hop = {
            'explicit-route-usage': 'route-include-ero',
            'index': i,
            'num-unnum-hop': {
                'node-id': node,
                'link-tp-id': 'link-tp-id is not used',
                'hop-type': 'STRICT'
                }
            }
        json['explicit-route-objects']['route-object-include-exclude'].append(hop)
        i = i + 1
        if regen_types is not None:
            hop_type = {
                "explicit-route-usage": "route-include-ero",
                "index": i,
                "regenerator": {
                    "technology": "flexi-grid",
                    "trx_type": regen_type,
                    "trx_mode": regen_mode,
                    "effective-freq-slot": [
                        {
                            "N": None,
                            "M": None
                        }
                    ],
                    "spacing": spacing
                    }
                }
            json['explicit-route-objects']['route-object-include-exclude'].append(hop_type)
            i = i + 1
    return json


@pytest.mark.parametrize('regen_type, regen_mode', [(['Voyager', 'Voyager'], ['mode 1', 'mode 1']),
                                                    (['Voyager', 'Voyager'], ['mode 2', 'mode 1']),
                                                    (['Voyager', 'Voyager'], ['mode 1', 'mode 2']),
                                                    (['Voyager', 'Voyager'], ['mode 2', 'mode 2']),
                                                    (['Voyager', 'vendorA_trx-type1'], ['mode 1', 'mode 1']),
                                                    (['vendorA_trx-type1', None], ['mode 1', None]),
                                                    ([None, 'vendorA_trx-type1'], [None, 'mode 1']),
                                                    ([None, 'Voyager'], [None, 'mode 2']),
                                                    ([None, None], [None, None])])
@pytest.mark.parametrize('bidir', [False, True])
@pytest.mark.parametrize('source, destination, regen_nodes, route_prev_nodes, check_prev_nodes, spacing', (
    ('trx node A', 'trx node C', [], [], [], []),
    ('trx node A', 'trx node C', ['regen node B'], [''], ['roadm node B'], [50e9]),
    ('trx node A', 'trx node C', ['regen node B'], [''], ['roadm node B'], [100e9]),
    ('trx node A', 'trx node C', ['regen node B'], ['roadm node B'], ['roadm node B'], [50e9]),
    ('trx node A', 'trx node D', ['regen node B'], [''], ['roadm node B'], [50e9]),
    ('trx node A', 'trx node D', ['regen node B', 'regen node C'], ['', ''], ['roadm node B', 'roadm node C'], [50e9, 50e9]),
    ('trx node A', 'trx node D', ['regen node B', 'regen node C'], ['', ''], ['roadm node B', 'roadm node C'], [50e9, 100e9]),
    ('trx node A', 'trx node D', ['regen node B', 'regen node C'], ['', ''], ['roadm node B', 'roadm node C'], [100e9, 100e9])))
def test_read_service_with_regen(setup, equipment,
                                 source, destination, regen_nodes, route_prev_nodes, check_prev_nodes, spacing,
                                 bidir,
                                 regen_type, regen_mode):
    """ creates a service with different configuration of regenerators, and use them to compute path and propagate
    verifies that created path is set according to request
    """
    network = setup
    json = json_request(source, destination, bidir,regen_nodes, route_prev_nodes, regen_type, regen_mode, spacing)
    json_data = {'path-request': [json]}
    min_spacing = []
    for node, typ, mode in zip(regen_nodes, regen_type, regen_mode):
        if typ is not None:
            min_spacing.append(trx_mode_params(equipment, typ, mode)['min_spacing'])
        else:
            min_spacing.append(
                trx_mode_params(equipment,
                                json['path-constraints']['te-bandwidth']['trx_type'],
                                json['path-constraints']['te-bandwidth']['trx_mode'])['min_spacing'])
    # check that demands with inconsistant spacing/mode are correctly raising a ServiceError
    if regen_nodes and any(min_spacing[j] > s for j, s in enumerate(spacing)):
        with pytest.raises(ServiceError):
            rqs = requests_from_json(json_data, equipment)
    elif not any(min_spacing[j] > s for j, s in enumerate(spacing)):
        rqs = requests_from_json(json_data, equipment)
        rqs = correct_json_route_list(network, rqs)
        rqs = correct_json_regen_list(network, equipment, rqs)
        rqs = remove_regen_from_list(network, rqs)
        # check that if regen type is not specifed, the regen is set to the default
        defaul_trx_type = json['path-constraints']['te-bandwidth']['trx_type']
        default_trx_mode = json['path-constraints']['te-bandwidth']['trx_mode']
        for node, typ, mode in zip(rqs[0].regen_list, regen_type, regen_mode):
            print(node)
            if typ is not None:
                assert node['trx_type'] == typ
                assert node['trx_mode'] == mode
            else:
                assert node['trx_type'] == defaul_trx_type
                assert node['trx_mode'] == default_trx_mode
        # reproduce the same process as in cli and check the path
        pths = compute_path_dsjctn(network, equipment, rqs, [])
        if destination == 'trx node C':
            expected_path = ['trx node A', 'roadm node A', 'Edfa0_roadm node A', 'fiber (node A → ila1)-',
                             'Edfa0_fiber (node A → ila1)-', 'fiber (ila1 → ila2)-', 'Edfa0_fiber (ila1 → ila2)-',
                             'fiber (ila2 → node B)-', 'Edfa0_fiber (ila2 → node B)-', 'roadm node B',
                             'Edfa1_roadm node B', 'fiber (node B → ila3)-', 'Edfa0_fiber (node B → ila3)-',
                             'fiber (ila3 → ila4)-', 'Edfa0_fiber (ila3 → ila4)-', 'fiber (ila4 → node C)-',
                             'Edfa0_fiber (ila4 → node C)-', 'roadm node C', 'trx node C']
        elif destination == 'trx node D':
            expected_path = ['trx node A', 'roadm node A', 'Edfa0_roadm node A', 'fiber (node A → ila1)-',
                             'Edfa0_fiber (node A → ila1)-', 'fiber (ila1 → ila2)-', 'Edfa0_fiber (ila1 → ila2)-',
                             'fiber (ila2 → node B)-', 'Edfa0_fiber (ila2 → node B)-', 'roadm node B',
                             'Edfa1_roadm node B', 'fiber (node B → ila3)-', 'Edfa0_fiber (node B → ila3)-',
                             'fiber (ila3 → ila4)-', 'Edfa0_fiber (ila3 → ila4)-', 'fiber (ila4 → node C)-',
                             'Edfa0_fiber (ila4 → node C)-', 'roadm node C', 'Edfa1_roadm node C',
                             'fiber (node C → ila5)-', 'Edfa0_fiber (node C → ila5)-', 'fiber (ila5 → ila6)-',
                             'Edfa0_fiber (ila5 → ila6)-', 'fiber (ila6 → node D)-', 'Edfa0_fiber (ila6 → node D)-',
                             'roadm node D', 'trx node D']
        assert [e.uid for e in pths[0]] == expected_path

        pths = restore_regen_in_path(network, rqs, pths)
        propagatedpths, reversed_pths, reversed_propagatedpths = \
            compute_path_with_disjunction(network, equipment, rqs, pths)

        # behaviour tests:
        # check that the path is correctly going through the listed regen(s) with relevant modes when specified
        # check that the bidir path is correctly implemented with a regen at the same roadm place for both directions
        all_pths = [pths[0], list(reversed(reversed_pths[0])),
                    propagatedpths[0], list(reversed(reversed_propagatedpths[0]))]
        for pth in all_pths:
            reg_nodes = [n.uid for n in pth if isinstance(n, Regenerator)]
            reg_prev_nodes = [pth[i - 1].uid for i, n in enumerate(pth) if isinstance(n, Regenerator)]
            reg_next_nodes = [pth[i + 1].uid for i, n in enumerate(pth) if isinstance(n, Regenerator)]
            for rnode, prev_node, next_node, reg, check_prev_node in \
                    zip(reg_nodes, reg_prev_nodes, reg_next_nodes, regen_nodes, check_prev_nodes):
                # assert regen is correctly part of the path
                assert rnode == reg
                # check that the regen in path is correctly placed between the two roadm instances
                assert check_prev_node == prev_node
                assert check_prev_node == next_node
        j = 0
        for elem in propagatedpths[0]:
            if isinstance(elem, Regenerator):
                # check that  the regen caracteristics correspond to the one specified
                if regen_type[j] is None:
                    reg_type = defaul_trx_type
                else:
                    reg_type = regen_type[j]
                if regen_mode[j] is None:
                    reg_mode = default_trx_mode
                else:
                    reg_mode = regen_mode[j]
                # check the baudrate only, use the first carrier
                assert elem.baud_rate[0] == trx_mode_params(equipment, reg_type, reg_mode)['baud_rate']
                j = j + 1


@pytest.mark.parametrize('regen', (['regen node B'], ['regen node B', 'regen node C']))
def test_regen_section(setup, equipment, regen):
    """ check that the regenerated SNR of each section gives the same result as a the list of independant requests
    for each section (inc txOSNR and add-drop OSNR)
    """
    network = setup
    temp1 = ['' for e in regen]
    temp2 = [None for e in regen]
    json = json_request('trx node A', 'trx node D', True, regen, temp1, temp2, temp2, temp2)
    rqs = requests_from_json({'path-request': [json]}, equipment)
    rqs = correct_json_regen_list(network, equipment, rqs)
    rqs = remove_regen_from_list(network, rqs)
    pths = compute_path_dsjctn(network, equipment, rqs, [])
    expected_path = ['trx node A', 'roadm node A', 'Edfa0_roadm node A', 'fiber (node A → ila1)-',
                     'Edfa0_fiber (node A → ila1)-', 'fiber (ila1 → ila2)-', 'Edfa0_fiber (ila1 → ila2)-',
                     'fiber (ila2 → node B)-', 'Edfa0_fiber (ila2 → node B)-', 'roadm node B',
                     'Edfa1_roadm node B', 'fiber (node B → ila3)-', 'Edfa0_fiber (node B → ila3)-',
                     'fiber (ila3 → ila4)-', 'Edfa0_fiber (ila3 → ila4)-', 'fiber (ila4 → node C)-',
                     'Edfa0_fiber (ila4 → node C)-', 'roadm node C', 'Edfa1_roadm node C',
                     'fiber (node C → ila5)-', 'Edfa0_fiber (node C → ila5)-', 'fiber (ila5 → ila6)-',
                     'Edfa0_fiber (ila5 → ila6)-', 'fiber (ila6 → node D)-', 'Edfa0_fiber (ila6 → node D)-',
                     'roadm node D', 'trx node D']
    assert [e.uid for e in pths[0]] == expected_path
    pths = restore_regen_in_path(network, rqs, pths)

    propagatedpths, reversed_pths, reversed_propagatedpths = compute_path_with_disjunction(network, equipment, rqs, pths)
    path = propagatedpths[0]
    sections = [e for e in path[1:] if isinstance(e, Transceiver)]
    expected_sections = []
    source = 'trx node A'
    for elem in sections:
        dest = elem.uid.replace('regen', 'trx')
        print(source, dest)
        json_section_data = json_data(equipment, source, dest)
        expected_rqs = requests_from_json(json_section_data, equipment)
        # expected_rqs = correct_json_route_list(network, expected_rqs)
        expected_pths = compute_path_dsjctn(network, equipment, expected_rqs, [])
        expected_propagated, _, _ = compute_path_with_disjunction(network, equipment, expected_rqs, expected_pths)
        assert elem.snr_01nm == expected_propagated[0][-1].snr_01nm
        source = dest

# select a place for a regen:
# simple algorithm: same type/mode as source
# if source type and mode are selected and path is not feasible find a regen place where a regenerator exists
# an unfeasible OMS means that no regen can solve this, so the path is set to unfeasible
# if source mode is not specified, use another criterium from the user. default is highest baudrate/highest bitrate

# check before and after propagation and for reversed path

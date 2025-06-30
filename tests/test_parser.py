#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_parser
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""Adding tests to check the parser non regression

convention of naming of test files:
- ..._expected.json for the reference output
tests:
- generation of topology json
- reading of Eqpt sheet w and W/ power mode
- consistency of autodesign
- generation of service list based on service sheet
- writing of results in csv
- writing of results in json (same keys)
"""

from pathlib import Path
from os import unlink
import shutil
from pandas import read_csv
from xlrd import open_workbook
import pytest
from copy import deepcopy
from gnpy.core.utils import automatic_nch, dbm2watt
from gnpy.core.network import build_network, add_missing_elements_in_network
from gnpy.core.exceptions import ServiceError, ConfigurationError
from gnpy.topology.request import (jsontocsv, requests_aggregation, compute_path_dsjctn, deduplicate_disjunctions,
                                   compute_path_with_disjunction, ResultElement, PathRequest)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.convert import convert_file
from gnpy.tools.json_io import (load_json, load_network, save_network, load_equipment, requests_from_json,
                                disjunctions_from_json, network_to_json, network_from_json)
from gnpy.tools.service_sheet import read_service_sheet, correct_xls_route_list, Request_element, Request

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}
equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)


def pathrequest(pch_dbm: float, p_tot_dbm: float = None, nb_channels: int = None):
    """create ref channel for defined power settings
    """
    params = {
        "power": dbm2watt(pch_dbm),
        "tx_power": dbm2watt(pch_dbm),
        "nb_channel": nb_channels if nb_channels else round(dbm2watt(p_tot_dbm) / dbm2watt(pch_dbm), 0),
        'request_id': None,
        'trx_type': None,
        'trx_mode': None,
        'source': None,
        'destination': None,
        'bidir': False,
        'nodes_list': [],
        'loose_list': [],
        'format': '',
        'baud_rate': None,
        'bit_rate': None,
        'roll_off': None,
        'OSNR': None,
        'penalties': None,
        'path_bandwidth': None,
        'effective_freq_slot': None,
        'f_min': None,
        'f_max': None,
        'spacing': None,
        'min_spacing': None,
        'cost': None,
        'equalization_offset_db': None,
        'tx_osnr': None
    }
    return PathRequest(**params)


@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'CORONET_Global_Topology.xlsx': DATA_DIR / 'CORONET_Global_Topology_expected.json',
    DATA_DIR / 'testTopology.xls': DATA_DIR / 'testTopology_expected.json',
    DATA_DIR / 'perdegreemeshTopologyExampleV2.xls': DATA_DIR / 'perdegreemeshTopologyExampleV2_expected.json'

}.items())
def test_excel_json_generation(tmpdir, xls_input, expected_json_output):
    """tests generation of topology json"""
    xls_copy = Path(tmpdir) / xls_input.name
    shutil.copyfile(xls_input, xls_copy)
    convert_file(xls_copy)

    actual_json_output = xls_copy.with_suffix('.json')
    actual = load_json(actual_json_output)
    unlink(actual_json_output)
    assert actual == load_json(expected_json_output)

# assume xls entries
# test that the build network gives correct results in gain mode


@pytest.mark.parametrize('xls_input,expected_json_output',
                         {DATA_DIR / 'CORONET_Global_Topology.xlsx':
                          DATA_DIR / 'CORONET_Global_Topology_auto_design_expected.json',
                          DATA_DIR / 'testTopology.xls':
                          DATA_DIR / 'testTopology_auto_design_expected.json',
                          }.items())
def test_auto_design_generation_fromxlsgainmode(tmpdir, xls_input, expected_json_output):
    """tests generation of topology json and that the build network gives correct results in gain mode"""
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = load_network(xls_input, equipment)
    add_missing_elements_in_network(network, equipment)
    # in order to test the Eqpt sheet and load gain target,
    # change the power-mode to False (to be in gain mode)
    equipment['Span']['default'].power_mode = False
    # Build the network once using the default power defined in SI in eqpt config

    p_db = equipment['SI']['default'].power_dbm
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
    actual_json_output = tmpdir / xls_input.with_name(xls_input.stem + '_auto_design').with_suffix('.json').name
    save_network(network, actual_json_output)
    actual = load_json(actual_json_output)
    unlink(actual_json_output)
    assert actual == load_json(expected_json_output)

# test that autodesign creates same file as an input file already autodesigned


@pytest.mark.parametrize('json_input, power_mode',
                         {DATA_DIR / 'CORONET_Global_Topology_auto_design_expected.json':
                          False,
                          DATA_DIR / 'testTopology_auto_design_expected.json':
                          False,
                          DATA_DIR / 'perdegreemeshTopologyExampleV2_auto_design_expected.json':
                          True
                          }.items())
def test_auto_design_generation_fromjson(tmpdir, json_input, power_mode):
    """test that autodesign creates same file as an input file already autodesigned"""
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = load_network(json_input, equipment)
    # in order to test the Eqpt sheet and load gain target,
    # change the power-mode to False (to be in gain mode)
    equipment['Span']['default'].power_mode = power_mode
    # Build the network once using the default power defined in SI in eqpt config

    p_db = equipment['SI']['default'].power_dbm
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)
    add_missing_elements_in_network(network, equipment)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
    actual_json_output = tmpdir / json_input.with_name(json_input.stem + '_auto_design').with_suffix('.json').name
    save_network(network, actual_json_output)
    actual = load_json(actual_json_output)
    unlink(actual_json_output)
    assert actual == load_json(json_input)

# test services creation


@pytest.mark.parametrize('xls_input, expected_json_output', {
    DATA_DIR / 'testTopology.xls': DATA_DIR / 'testTopology_services_expected.json',
    DATA_DIR / 'testService.xls': DATA_DIR / 'testService_services_expected.json'
}.items())
def test_excel_service_json_generation(xls_input, expected_json_output):
    """test services creation"""
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = load_network(DATA_DIR / 'testTopology.xls', equipment)
    # Build the network once using the default power defined in SI in eqpt config
    p_db = equipment['SI']['default'].power_dbm
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
    from_xls = read_service_sheet(xls_input, equipment, network, network_filename=DATA_DIR / 'testTopology.xls')
    assert from_xls == load_json(expected_json_output)

    # TODO verify that requested bandwidth is not zero !

# test xls answers creation


@pytest.mark.parametrize('json_input',
    (DATA_DIR / 'testTopology_response.json', )
)
def test_csv_response_generation(tmpdir, json_input):
    """tests if generated csv is consistant with expected generation same columns (order not important)"""
    json_data = load_json(json_input)
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    csv_filename = Path(tmpdir / json_input.name).with_suffix('.csv')
    with open(csv_filename, 'w', encoding='utf-8') as fcsv:
        jsontocsv(json_data, equipment, fcsv)

    expected_csv_filename = json_input.parent / (json_input.stem + '_expected.csv')

    # expected header
    # csv_header = \
    # [
    #  'response-id',
    #  'source',
    #  'destination',
    #  'path_bandwidth',
    #  'Pass?',
    #  'nb of tsp pairs',
    #  'total cost',
    #  'transponder-type',
    #  'transponder-mode',
    #  'OSNR-0.1nm',
    #  'SNR-0.1nm',
    #  'SNR-bandwidth',
    #  'baud rate (Gbaud)',
    #  'input power (dBm)',
    #  'path',
    #  'spectrum (N,M)',
    #  'reversed path OSNR-0.1nm',
    #  'reversed path SNR-0.1nm',
    #  'reversed path SNR-bandwidth'
    # ]

    resp = read_csv(csv_filename)
    print(resp)
    unlink(csv_filename)
    expected_resp = read_csv(expected_csv_filename)
    print(expected_resp)
    resp_header = list(resp.head(0))
    expected_resp_header = list(expected_resp.head(0))
    # check that headers are the same
    resp_header.sort()
    expected_resp_header.sort()
    print('headers are differents')
    print(resp_header)
    print(expected_resp_header)
    assert resp_header == expected_resp_header

    # for each header checks that the output are as expected
    resp.sort_values(by=['response-id'])
    expected_resp.sort_values(by=['response-id'])

    for column in expected_resp:
        assert list(resp[column].fillna('')) == list(expected_resp[column].fillna(''))
        print('results are different')
        print(list(resp[column]))
        print(list(expected_resp[column]))
        print(type(list(resp[column])[-1]))


# test json answers creation
@pytest.mark.parametrize('xls_input, expected_response_file', {
    DATA_DIR / 'testTopology.xls': DATA_DIR / 'testTopology_response.json',
}.items())
def test_json_response_generation(xls_input, expected_response_file):
    """tests if json response is correctly generated for all combinations of requests"""

    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = load_network(xls_input, equipment)
    p_db = equipment['SI']['default'].power_dbm

    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))

    data = read_service_sheet(xls_input, equipment, network)
    # change one of the request with bidir option to cover bidir case as well
    data['path-request'][2]['bidirectional'] = True

    oms_list = build_oms_list(network, equipment)
    rqs = requests_from_json(data, equipment)
    dsjn = disjunctions_from_json(data)
    dsjn = deduplicate_disjunctions(dsjn)
    rqs, dsjn = requests_aggregation(rqs, dsjn)
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    propagatedpths, reversed_pths, reversed_propagatedpths = \
        compute_path_with_disjunction(network, equipment, rqs, pths, redesign=True)
    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)

    result = []
    for i, pth in enumerate(propagatedpths):
        # test ServiceError handling : when M is None at this point, the
        # json result should not be created if there is no blocking reason
        if i == 1:
            my_rq = deepcopy(rqs[i])
            my_rq.M = None
            my_rq.N = None
            with pytest.raises(ServiceError):
                ResultElement(my_rq, pth, reversed_propagatedpths[i]).json

            my_rq.blocking_reason = 'NO_SPECTRUM'
            ResultElement(my_rq, pth, reversed_propagatedpths[i]).json

        result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))

    temp = {
        'response': [n.json for n in result]
    }

    expected = load_json(expected_response_file)

    for i, response in enumerate(temp['response']):
        if i == 2:
            # compare response must be False because z-a metric is missing
            # (request with bidir option to cover bidir case)
            assert expected['response'][i] != response
            print(f'response {response["response-id"]} should not match')
            expected['response'][2]['path-properties']['z-a-path-metric'] = [
                {'metric-type': 'SNR-bandwidth', 'accumulative-value': 22.809999999999999},
                {'metric-type': 'SNR-0.1nm', 'accumulative-value': 26.890000000000001},
                {'metric-type': 'OSNR-bandwidth', 'accumulative-value': 26.239999999999998},
                {'metric-type': 'OSNR-0.1nm', 'accumulative-value': 30.32},
                {'metric-type': 'reference_power', 'accumulative-value': 0.0012589254117941673},
                {'metric-type': 'path_bandwidth', 'accumulative-value': 60000000000.0}]
            # test should be OK now
        else:
            assert expected['response'][i] == response


@pytest.mark.parametrize('source, destination, route_list, hoptype, expected_correction', [
    ('Brest_KLA', 'Vannes_KBE',
        'roadm Brest_KLA | roadm Lannion_CAS | roadm Lorient_KMA | roadm Vannes_KBE',
        'no',
        ['roadm Brest_KLA', 'roadm Lannion_CAS', 'roadm Lorient_KMA', 'roadm Vannes_KBE']),
    ('Brest_KLA', 'Vannes_KBE',
        'trx Brest_KLA | roadm Lannion_CAS | roadm Lorient_KMA | roadm Vannes_KBE',
        'No',
        ['roadm Lannion_CAS', 'roadm Lorient_KMA', 'roadm Vannes_KBE']),
    ('Lannion_CAS', 'Rennes_STA', 'trx Rennes_STA', 'yes', []),
    ('Lannion_CAS', 'Lorient_KMA', 'toto', 'yes', []),
    ('Lannion_CAS', 'Lorient_KMA', 'toto', 'no', 'Fail'),
    ('Lannion_CAS', 'Lorient_KMA', 'Corlay | Loudeac | Lorient_KMA', 'yes',
        ['west fused spans in Corlay', 'west fused spans in Loudeac', 'roadm Lorient_KMA']),
    ('Lannion_CAS', 'Lorient_KMA', 'Ploermel | Vannes_KBE', 'yes',
        ['east edfa in Ploermel to Vannes_KBE', 'roadm Vannes_KBE']),
    ('Rennes_STA', 'Brest_KLA', 'Vannes_KBE | Quimper | Brest_KLA', 'yes',
        ['roadm Vannes_KBE', 'west edfa in Quimper to Lorient_KMA', 'roadm Brest_KLA']),
    ('Brest_KLA', 'Rennes_STA', 'Brest_KLA | Quimper | Lorient_KMA', 'yes',
        ['roadm Brest_KLA', 'east edfa in Quimper to Lorient_KMA', 'roadm Lorient_KMA']),
    ('Brest_KLA', 'trx Rennes_STA', '', 'yes', 'Fail'),
    ('trx Brest_KLA', 'Rennes_STA', '', 'yes', 'Fail'),
    ('trx Brest_KLA', 'trx Rennes_STA', '', 'yes', 'Fail'),
    ('Brest_KLA', 'trx Rennes_STA', '', 'no', 'Fail'),
    ('Brest_KLA', 'Rennes_STA', 'trx Rennes_STA', 'no', []),
    ('Brest_KLA', 'Rennes_STA', None, '', []),
    ('Brest_KLA', 'Rennes_STA', 'Brest_KLA | Quimper | Ploermel', 'yes',
        ['roadm Brest_KLA']),
    ('Brest_KLA', 'Rennes_STA', 'Brest_KLA | Quimper | Ploermel', 'no',
        ['roadm Brest_KLA']),
    ('Brest_KLA', 'Rennes_STA', 'Brest_KLA | trx Quimper', 'yes', ['roadm Brest_KLA']),
    ('Brest_KLA', 'Rennes_STA', 'Brest_KLA | trx Lannion_CAS', 'yes', ['roadm Brest_KLA']),
    ('Brest_KLA', 'Rennes_STA', 'Brest_KLA | trx Lannion_CAS', 'no', 'Fail')
])
def test_excel_ila_constraints(source, destination, route_list, hoptype, expected_correction):
    """add different kind of constraints to test all correct_route cases"""
    service_xls_input = DATA_DIR / 'testTopology.xls'
    network_json_input = DATA_DIR / 'testTopology_auto_design_expected.json'
    network = load_network(network_json_input, equipment)
    # increase length of one span to trigger automatic fiber splitting included by autodesign
    # so that the test also covers this case
    next(node for node in network.nodes() if node.uid == 'fiber (Brest_KLA → Quimper)-').length = 200000
    next(node for node in network.nodes() if node.uid == 'fiber (Quimper → Brest_KLA)-').length = 200000
    default_si = equipment['SI']['default']
    p_db = default_si.power_dbm
    nb_channels = automatic_nch(default_si.f_min, default_si.f_max, default_si.spacing)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
    # create params for a xls request based on input (note that this not the same type as PathRequest)
    params = {
        'request_id': '0',
        'source': source,
        'destination': destination,
        'trx_type': 'Voyager',
        'spacing': 50,
        'nodes_list': route_list,
        'is_loose': hoptype,
        'nb_channel': 0,
        'power': 0,
        'path_bandwidth': 0,
    }
    request = Request_element(Request(**params), equipment, False)

    if expected_correction != 'Fail':
        [request] = correct_xls_route_list(service_xls_input, network, [request])
        assert request.nodes_list == expected_correction
    else:
        with pytest.raises(ServiceError):
            [request] = correct_xls_route_list(service_xls_input, network, [request])


@pytest.mark.parametrize('route_list, hoptype, expected_amp_route', [
    ('node1 | siteE | node2', 'no',
     ['roadm node1', 'west edfa in siteE', 'roadm node2']),
    ('node2 | siteE | node1', 'no',
     ['roadm node2', 'east edfa in siteE', 'roadm node1']),
    ('node1 | siteF | node2', 'no',
     ['roadm node1', 'west edfa in siteF', 'roadm node2']),
    ('node1 | siteA | siteB', 'yes',
     ['roadm node1', 'west edfa in siteA']),
    ('node1 | siteA | siteB | node2', 'yes',
     ['roadm node1', 'west edfa in siteA', 'west edfa in siteB', 'roadm node2']),
    ('node1 | siteC | node2', 'yes',
     ['roadm node1', 'east edfa in siteC', 'roadm node2']),
    ('node1 | siteD | node2', 'no',
     ['roadm node1', 'west edfa in siteD to node1', 'roadm node2']),
    ('roadm node1 | Edfa_booster_roadm node1_to_fiber (node1 → siteE)-CABLES#19 | west edfa in siteE | node2',
     'no',
     ['roadm node1', 'Edfa_booster_roadm node1_to_fiber (node1 → siteE)-CABLES#19',
      'west edfa in siteE', 'roadm node2'])])
def test_excel_ila_constraints2(route_list, hoptype, expected_amp_route):
    """Check different cases for ILA constraints definition
    """
    network_xls_input = DATA_DIR / 'ila_constraint.xlsx'
    network = load_network(network_xls_input, equipment)
    add_missing_elements_in_network(network, equipment)
    default_si = equipment['SI']['default']
    p_db = default_si.power_dbm
    nb_channels = automatic_nch(default_si.f_min, default_si.f_max, default_si.spacing)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
    # create params for a request based on input

    params = {
        'request_id': '0',
        'source': 'node1',
        'destination': 'node2',
        'trx_type': 'Voyager',
        'mode': None,
        'spacing': 50,
        'nodes_list': route_list,
        'is_loose': hoptype,
        'nb_channel': 80,
        'power': 0,
        'path_bandwidth': 100,
    }
    request = Request_element(Request(**params), equipment, False)
    [request] = correct_xls_route_list(network_xls_input, network, [request])
    assert request.nodes_list == expected_amp_route


def setup_per_degree(case):
    """common setup for degree: returns the dict network for different cases"""
    json_network = load_json(DATA_DIR / 'testTopology_expected.json')
    json_network_auto = load_json(DATA_DIR / 'testTopology_auto_design_expected.json')
    if case == 'no':
        return json_network
    elif case == 'all':
        return json_network_auto
    elif case == 'Lannion_CAS and all':
        elem = next(e for e in json_network['elements'] if e['uid'] == 'roadm Lannion_CAS')
        elem['params'] = {'per_degree_pch_out_db': {
            "east edfa in Lannion_CAS to Corlay": -17,
            "east edfa in Lannion_CAS to Stbrieuc": -18,
            "east edfa in Lannion_CAS to Morlaix": -21}}
        return json_network
    elif case == 'Lannion_CAS and one':
        elem = next(e for e in json_network['elements'] if e['uid'] == 'roadm Lannion_CAS')
        elem['params'] = {'per_degree_pch_out_db': {
            "east edfa in Lannion_CAS to Corlay": -17,
            "east edfa in Lannion_CAS to Stbrieuc": -18}}
        return json_network


@pytest.mark.parametrize('case', ['no', 'all', 'Lannion_CAS and all', 'Lannion_CAS and one'])
def test_target_pch_out_db_global(case):
    """check that per degree attributes are correctly created with global values if none are given"""
    json_network = setup_per_degree(case)
    per_degree = {}
    for elem in json_network['elements']:
        if 'type' in elem.keys() and elem['type'] == 'Roadm' and 'params' in elem.keys() \
            and 'per_degree_pch_out_db' in elem['params']:
            # records roadms that have a per degree target
            per_degree[elem['uid']] = {k: v for k, v in elem['params']['per_degree_pch_out_db'].items()}
    network = network_from_json(json_network, equipment)
    # Build the network once using the default power defined in SI in eqpt config
    # power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max,
                                equipment['SI']['default'].spacing)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))

    data = network_to_json(network)
    for elem in data['elements']:
        if 'type' in elem.keys() and elem['type'] == 'Roadm':
            # check that power target attributes exist and are filled with correct values
            # first check that global 'target_pch_out_db' is correctly filled
            assert elem['params']['target_pch_out_db'] == equipment['Roadm']['default'].target_pch_out_db
            for degree, power in elem['params']['per_degree_pch_out_db'].items():
                if elem['uid'] not in per_degree.keys():
                    # second: check that per degree 'target_pch_out_db' is correctly filled with global value
                    # when there was no per degree specification on network input
                    assert power == equipment['Roadm']['default'].target_pch_out_db
                else:
                    if degree not in per_degree[elem['uid']].keys():
                        # third: check that per degree 'target_pch_out_db' is correctly filled with global value
                        # on degrees that had no specification when other degrees are filled
                        assert power == equipment['Roadm']['default'].target_pch_out_db
                    else:
                        # fourth: check that per degree 'target_pch_out_db' is correctly filled with specified values
                        assert power == per_degree[elem['uid']][degree]


def all_rows(sh, start=0):
    """reads excel sheet row per row"""
    return (sh.row(x) for x in range(start, sh.nrows))


class Amp:
    """Node element contains uid, list of connected nodes and eqpt type"""

    def __init__(self, uid, to_node, eqpt=None, west=None):
        self.uid = uid
        self.to_node = to_node
        self.eqpt = eqpt
        self.west = west


def test_eqpt_creation(tmpdir):
    """tests that convert correctly creates equipment according to equipment sheet
    including all cominations in testTopologyconvert.xls: if a line exists the amplifier
    should be created even if no values are provided.
    """
    xls_input = DATA_DIR / 'testTopologyconvert.xls'

    xls_copy = Path(tmpdir) / xls_input.name
    shutil.copyfile(xls_input, xls_copy)
    convert_file(xls_copy)

    actual_json_output = xls_copy.with_suffix('.json')
    actual = load_json(actual_json_output)
    unlink(actual_json_output)

    connections = {elem['from_node']: elem['to_node'] for elem in actual['connections']}
    jsonconverted = {}
    for elem in actual['elements']:
        if 'type' in elem.keys() and elem['type'] == 'Edfa':
            print(elem['uid'])
            if 'type_variety' in elem.keys():
                jsonconverted[elem['uid']] = Amp(elem['uid'], connections[elem['uid']], elem['type_variety'])
            else:
                jsonconverted[elem['uid']] = Amp(elem['uid'], connections[elem['uid']])

    with open_workbook(xls_input) as wobo:
        # reading Eqpt sheet assuming header is node A, Node Z, amp variety
        # fused should not be recorded as an amp
        eqpt_sheet = wobo.sheet_by_name('Eqpt')
        raw_eqpts = {}
        for row in all_rows(eqpt_sheet, start=5):
            if row[0].value not in raw_eqpts.keys():
                raw_eqpts[row[0].value] = Amp(row[0].value, [row[1].value], [row[2].value], [row[7].value])
            else:
                raw_eqpts[row[0].value].to_node.append(row[1].value)
                raw_eqpts[row[0].value].eqpt.append(row[2].value)
                raw_eqpts[row[0].value].west.append(row[7].value)
    # create the possible names similarly to what convert should do
    possiblename = [f'east edfa in {xlsname} to {node}' for xlsname, value in raw_eqpts.items()
                    for i, node in enumerate(value.to_node) if value.eqpt[i] != 'fused'] +\
                   [f'west edfa in {xlsname} to {node}' for xlsname, value in raw_eqpts.items()
                    for i, node in enumerate(value.to_node) if value.west[i] != 'fused']
    # check that all lines in eqpt sheet correctly converts to an amp element
    for name in possiblename:
        assert name in jsonconverted.keys()
    # check that all amp in the converted files corresponds to an eqpt line
    for ampuid in jsonconverted.keys():
        assert ampuid in possiblename


def test_service_json_constraint_order():
    """test that the constraints are read in correct order"""

    unsorted_request = {
        "request-id": "unsorted",
        "source": "trx Brest_KLA",
        "destination": "trx Vannes_KBE",
        "src-tp-id": "trx Brest_KLA",
        "dst-tp-id": "trx Vannes_KBE",
        "bidirectional": False,
        "path-constraints": {
            "te-bandwidth": {
                "technology": "flexi-grid",
                "trx_type": "Voyager",
                "trx_mode": "mode 1",
                "spacing": 50000000000.0,
                "output-power": 0.001,
                "path_bandwidth": 10000000000.0
            }
        },
        "explicit-route-objects": {
            "route-object-include-exclude": [
                {
                    "explicit-route-usage": "route-include-ero",
                    "index": 2,
                    "num-unnum-hop": {
                        "node-id": "roadm Lorient_KMA",
                        "link-tp-id": "link-tp-id is not used",
                        "hop-type": "STRICT"
                    }
                },
                {
                    "explicit-route-usage": "route-include-ero",
                    "index": 3,
                    "num-unnum-hop": {
                        "node-id": "roadm Vannes_KBE",
                        "link-tp-id": "link-tp-id is not used",
                        "hop-type": "STRICT"
                    }
                },
                {
                    "explicit-route-usage": "route-include-ero",
                    "index": 1,
                    "num-unnum-hop": {
                        "node-id": "roadm Lannion_CAS",
                        "link-tp-id": "link-tp-id is not used",
                        "hop-type": "LOOSE"
                    }
                },
                {
                    "explicit-route-usage": "route-include-ero",
                    "index": 0,
                    "num-unnum-hop": {
                        "node-id": "roadm Brest_KLA",
                        "link-tp-id": "link-tp-id is not used",
                        "hop-type": "STRICT"
                    }
                }
            ]
        }
    }

    data = {'path-request': [unsorted_request]}
    rqs = requests_from_json(data, equipment)
    assert rqs[0].nodes_list == ['roadm Brest_KLA', 'roadm Lannion_CAS', 'roadm Lorient_KMA', 'roadm Vannes_KBE']
    assert rqs[0].loose_list == ['STRICT', 'LOOSE', 'STRICT', 'STRICT']


@pytest.mark.parametrize('type_variety, target_pch_out_db, correct_variety', [(None, -20, True),
                                                                              ('example_test', -18, True),
                                                                              ('example', None, False)])
def test_roadm_type_variety(type_variety, target_pch_out_db, correct_variety):
    """Checks that if element has no variety, the default one is applied, and if it has one
    that the type_variety is correctly applied
    """
    json_data = {
        "elements": [{
            "uid": "roadm Oakland",
            "type": "Roadm",
        }],
        "connections": []
    }
    expected_roadm = {
        "uid": "roadm Oakland",
        "type": "Roadm",
        "params": {
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            }
        },
        'metadata': {
            'location': {
                'city': None,
                'latitude': 0,
                'longitude': 0,
                'region': None
            }
        }
    }
    if type_variety is not None:
        json_data['elements'][0]['type_variety'] = type_variety
        expected_roadm['type_variety'] = type_variety
    else:
        # Do not add type variety in json_data to test that it creates a 'default' type_variety
        expected_roadm['type_variety'] = 'default'
    expected_roadm['params']['target_pch_out_db'] = target_pch_out_db
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    if correct_variety:
        network = network_from_json(json_data, equipment)
        roadm = [n for n in network.nodes()][0]
        assert roadm.to_json == expected_roadm
    else:
        with pytest.raises(ConfigurationError):
            network = network_from_json(json_data, equipment)

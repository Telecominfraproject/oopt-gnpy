#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2018-06-15

""" Adding tests to check the parser non regression
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
from tests.compare import compare_networks, compare_services
from copy import deepcopy
from gnpy.core.utils import automatic_nch, lin2db
from gnpy.core.network import build_network
from gnpy.core.exceptions import ServiceError
from gnpy.topology.request import (jsontocsv, requests_aggregation, compute_path_dsjctn, deduplicate_disjunctions,
                                   compute_path_with_disjunction, ResultElement, PathRequest)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.convert import convert_file
from gnpy.tools.json_io import (load_json, load_network, save_network, load_equipment, requests_from_json,
                                disjunctions_from_json, network_to_json, network_from_json)
from gnpy.tools.service_sheet import read_service_sheet, correct_xls_route_list

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
eqpt_filename = DATA_DIR / 'eqpt_config.json'
equipment = load_equipment(eqpt_filename)


@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'CORONET_Global_Topology.xlsx': DATA_DIR / 'CORONET_Global_Topology_expected.json',
    DATA_DIR / 'testTopology.xls': DATA_DIR / 'testTopology_expected.json',
    DATA_DIR / 'perdegreemeshTopologyExampleV2.xls': DATA_DIR / 'perdegreemeshTopologyExampleV2_expected.json'

}.items())
def test_excel_json_generation(tmpdir, xls_input, expected_json_output):
    """ tests generation of topology json
    """
    xls_copy = Path(tmpdir) / xls_input.name
    shutil.copyfile(xls_input, xls_copy)
    convert_file(xls_copy)

    actual_json_output = xls_copy.with_suffix('.json')
    actual = load_json(actual_json_output)
    unlink(actual_json_output)
    expected = load_json(expected_json_output)

    results = compare_networks(expected, actual)
    assert not results.elements.missing
    assert not results.elements.extra
    assert not results.elements.different
    assert not results.connections.missing
    assert not results.connections.extra
    assert not results.connections.different

# assume xls entries
# test that the build network gives correct results in gain mode


@pytest.mark.parametrize('xls_input,expected_json_output',
                         {DATA_DIR / 'CORONET_Global_Topology.xlsx':
                          DATA_DIR / 'CORONET_Global_Topology_auto_design_expected.json',
                          DATA_DIR / 'testTopology.xls':
                          DATA_DIR / 'testTopology_auto_design_expected.json',
                          }.items())
def test_auto_design_generation_fromxlsgainmode(tmpdir, xls_input, expected_json_output):
    """ tests generation of topology json
        test that the build network gives correct results in gain mode
    """
    equipment = load_equipment(eqpt_filename)
    network = load_network(xls_input, equipment)
    # in order to test the Eqpt sheet and load gain target,
    # change the power-mode to False (to be in gain mode)
    equipment['Span']['default'].power_mode = False
    # Build the network once using the default power defined in SI in eqpt config

    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    actual_json_output = tmpdir / xls_input.with_name(xls_input.stem + '_auto_design').with_suffix('.json').name
    save_network(network, actual_json_output)
    actual = load_json(actual_json_output)
    unlink(actual_json_output)
    expected = load_json(expected_json_output)

    results = compare_networks(expected, actual)
    assert not results.elements.missing
    assert not results.elements.extra
    assert not results.elements.different
    assert not results.connections.missing
    assert not results.connections.extra
    assert not results.connections.different

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
    """test that autodesign creates same file as an input file already autodesigned
    """
    equipment = load_equipment(eqpt_filename)
    network = load_network(json_input, equipment)
    # in order to test the Eqpt sheet and load gain target,
    # change the power-mode to False (to be in gain mode)
    equipment['Span']['default'].power_mode = power_mode
    # Build the network once using the default power defined in SI in eqpt config

    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    actual_json_output = tmpdir / json_input.with_name(json_input.stem + '_auto_design').with_suffix('.json').name
    save_network(network, actual_json_output)
    actual = load_json(actual_json_output)
    unlink(actual_json_output)
    expected = load_json(json_input)

    results = compare_networks(expected, actual)
    assert not results.elements.missing
    assert not results.elements.extra
    assert not results.elements.different
    assert not results.connections.missing
    assert not results.connections.extra
    assert not results.connections.different

# test services creation


@pytest.mark.parametrize('xls_input, expected_json_output', {
    DATA_DIR / 'testTopology.xls': DATA_DIR / 'testTopology_services_expected.json',
    DATA_DIR / 'testService.xls': DATA_DIR / 'testService_services_expected.json'
}.items())
def test_excel_service_json_generation(xls_input, expected_json_output):
    """ test services creation
    """
    equipment = load_equipment(eqpt_filename)
    network = load_network(DATA_DIR / 'testTopology.xls', equipment)
    # Build the network once using the default power defined in SI in eqpt config
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    from_xls = read_service_sheet(xls_input, equipment, network, network_filename=DATA_DIR / 'testTopology.xls')
    expected = load_json(expected_json_output)

    results = compare_services(expected, from_xls)
    assert not results.requests.missing
    assert not results.requests.extra
    assert not results.requests.different
    assert not results.synchronizations.missing
    assert not results.synchronizations.extra
    assert not results.synchronizations.different

    # TODO verify that requested bandwidth is not zero !

# test xls answers creation


@pytest.mark.parametrize('json_input',
    (DATA_DIR / 'testTopology_response.json', )
)
def test_csv_response_generation(tmpdir, json_input):
    """ tests if generated csv is consistant with expected generation
        same columns (order not important)
    """
    json_data = load_json(json_input)
    equipment = load_equipment(eqpt_filename)
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


def compare_response(exp_resp, act_resp):
    """ False if the keys are different in the nested dicts as well
    """
    print(exp_resp)
    print(act_resp)
    test = True
    for key in act_resp.keys():
        if key not in exp_resp.keys():
            print(f'{key} is not expected')
            return False
        if isinstance(act_resp[key], dict):
            test = compare_response(exp_resp[key], act_resp[key])
    if test:
        for key in exp_resp.keys():
            if key not in act_resp.keys():
                print(f'{key} is expected')
                return False
            if isinstance(exp_resp[key], dict):
                test = compare_response(exp_resp[key], act_resp[key])

    # at this point exp_resp and act_resp have the same keys. Check if their values are the same
    for key in act_resp.keys():
        if not isinstance(act_resp[key], dict):
            if exp_resp[key] != act_resp[key]:
                print(f'expected value :{exp_resp[key]}\n actual value: {act_resp[key]}')
                return False
    return test


# test json answers creation
@pytest.mark.parametrize('xls_input, expected_response_file', {
    DATA_DIR / 'testTopology.xls': DATA_DIR / 'testTopology_response.json',
}.items())
def test_json_response_generation(xls_input, expected_response_file):
    """ tests if json response is correctly generated for all combinations of requests
    """

    equipment = load_equipment(eqpt_filename)
    network = load_network(xls_input, equipment)
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)

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
        compute_path_with_disjunction(network, equipment, rqs, pths)
    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)

    result = []
    for i, pth in enumerate(propagatedpths):
        # test ServiceError handling : when M is zero at this point, the
        # json result should not be created if there is no blocking reason
        if i == 1:
            my_rq = deepcopy(rqs[i])
            my_rq.M = 0
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
            assert not compare_response(expected['response'][i], response)
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
            assert compare_response(expected['response'][i], response)
            print(f'response {response["response-id"]} is not correct')

# test the correspondance names dict in case of excel input
# test that using the created json network still works with excel input
# test all configurations of names: trx names, roadm, fused, ila and fiber
# as well as splitted case

# initial network is based on the couple testTopology.xls/ testTopology_auto_design_expected.json
# with added constraints to cover more test cases


@pytest.mark.parametrize('source, destination, route_list, hoptype, expected_correction', [
    ('trx Brest_KLA', 'trx Vannes_KBE',
        'roadm Brest_KLA | roadm Lannion_CAS | roadm Lorient_KMA | roadm Vannes_KBE',
        'STRICT',
        ['roadm Brest_KLA', 'roadm Lannion_CAS', 'roadm Lorient_KMA', 'roadm Vannes_KBE']),
    ('trx Brest_KLA', 'trx Vannes_KBE',
        'trx Brest_KLA | roadm Lannion_CAS | roadm Lorient_KMA | roadm Vannes_KBE',
        'STRICT',
        ['roadm Lannion_CAS', 'roadm Lorient_KMA', 'roadm Vannes_KBE']),
    ('trx Lannion_CAS', 'trx Rennes_STA', 'trx Rennes_STA', 'LOOSE', []),
    ('trx Lannion_CAS', 'trx Lorient_KMA', 'toto', 'LOOSE', []),
    ('trx Lannion_CAS', 'trx Lorient_KMA', 'toto', 'STRICT', 'Fail'),
    ('trx Lannion_CAS', 'trx Lorient_KMA', 'Corlay | Loudeac | Lorient_KMA', 'LOOSE',
        ['west fused spans in Corlay', 'west fused spans in Loudeac', 'roadm Lorient_KMA']),
    ('trx Lannion_CAS', 'trx Lorient_KMA', 'Ploermel | Vannes_KBE', 'LOOSE',
        ['east edfa in Ploermel to Vannes_KBE', 'roadm Vannes_KBE']),
    ('trx Rennes_STA', 'trx Brest_KLA', 'Vannes_KBE | Quimper | Brest_KLA', 'LOOSE',
        ['roadm Vannes_KBE', 'west edfa in Quimper to Lorient_KMA', 'roadm Brest_KLA']),
    ('trx Brest_KLA', 'trx Rennes_STA', 'Brest_KLA | Quimper | Lorient_KMA', 'LOOSE',
        ['roadm Brest_KLA', 'east edfa in Quimper to Lorient_KMA', 'roadm Lorient_KMA']),
    ('Brest_KLA', 'trx Rennes_STA', '', 'LOOSE', 'Fail'),
    ('trx Brest_KLA', 'Rennes_STA', '', 'LOOSE', 'Fail'),
    ('Brest_KLA', 'Rennes_STA', '', 'LOOSE', 'Fail'),
    ('Brest_KLA', 'trx Rennes_STA', '', 'STRICT', 'Fail'),
    ('trx Brest_KLA', 'trx Rennes_STA', 'trx Rennes_STA', 'STRICT', []),
    ('trx Brest_KLA', 'trx Rennes_STA', None, '', []),
    ('trx Brest_KLA', 'trx Rennes_STA', 'Brest_KLA | Quimper | Ploermel', 'LOOSE',
        ['roadm Brest_KLA']),
    ('trx Brest_KLA', 'trx Rennes_STA', 'Brest_KLA | Quimper | Ploermel', 'STRICT',
        ['roadm Brest_KLA']),
    ('trx Brest_KLA', 'trx Rennes_STA', 'Brest_KLA | trx Quimper', 'LOOSE', ['roadm Brest_KLA']),
    ('trx Brest_KLA', 'trx Rennes_STA', 'Brest_KLA | trx Lannion_CAS', 'LOOSE', ['roadm Brest_KLA']),
    ('trx Brest_KLA', 'trx Rennes_STA', 'Brest_KLA | trx Lannion_CAS', 'STRICT', 'Fail')
])
def test_excel_ila_constraints(source, destination, route_list, hoptype, expected_correction):
    """ add different kind of constraints to test all correct_route cases
    """
    service_xls_input = DATA_DIR / 'testTopology.xls'
    network_json_input = DATA_DIR / 'testTopology_auto_design_expected.json'
    equipment = load_equipment(eqpt_filename)
    network = load_network(network_json_input, equipment)
    # increase length of one span to trigger automatic fiber splitting included by autodesign
    # so that the test also covers this case
    next(node for node in network.nodes() if node.uid == 'fiber (Brest_KLA → Quimper)-').length = 200000
    next(node for node in network.nodes() if node.uid == 'fiber (Quimper → Brest_KLA)-').length = 200000
    default_si = equipment['SI']['default']
    p_db = default_si.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(default_si.f_min, default_si.f_max, default_si.spacing))
    build_network(network, equipment, p_db, p_total_db)
    # create params for a request based on input
    nodes_list = route_list.split(' | ') if route_list is not None else []
    params = {
        'request_id': '0',
        'source': source,
        'bidir': False,
        'destination': destination,
        'trx_type': '',
        'trx_mode': '',
        'format': '',
        'spacing': '',
        'nodes_list': nodes_list,
        'loose_list': [hoptype for node in nodes_list] if route_list is not None else '',
        'f_min': 0,
        'f_max': 0,
        'baud_rate': 0,
        'OSNR': None,
        'bit_rate': None,
        'cost': None,
        'roll_off': 0,
        'tx_osnr': 0,
        'min_spacing': None,
        'nb_channel': 0,
        'power': 0,
        'path_bandwidth': 0,
        'effective_freq_slot': None
    }
    request = PathRequest(**params)

    if expected_correction != 'Fail':
        [request] = correct_xls_route_list(service_xls_input, network, [request])
        assert request.nodes_list == expected_correction
    else:
        with pytest.raises(ServiceError):
            [request] = correct_xls_route_list(service_xls_input, network, [request])


def setup_per_degree(case):
    """ common setup for degree: returns the dict network for different cases
    """
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
    """ check that per degree attributes are correctly created with global values if none are given
    """
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
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max,
                                             equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)

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
    """ reads excel sheet row per row
    """
    return (sh.row(x) for x in range(start, sh.nrows))


class Amp:
    """ Node element contains uid, list of connected nodes and eqpt type
    """

    def __init__(self, uid, to_node, eqpt=None, west=None):
        self.uid = uid
        self.to_node = to_node
        self.eqpt = eqpt
        self.west = west


def test_eqpt_creation(tmpdir):
    """ tests that convert correctly creates equipment according to equipment sheet
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

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

from json import load
from pathlib import Path
from os import unlink
from pandas import read_csv
import pytest
from tests.compare import compare_networks, compare_services
from copy import deepcopy
from gnpy.core.utils import lin2db
from gnpy.core.network import save_network, build_network
from gnpy.core.convert import convert_file
from gnpy.core.service_sheet import convert_service_sheet
from gnpy.core.equipment import load_equipment, automatic_nch
from gnpy.core.network import load_network
from gnpy.core.request import (jsontocsv, requests_aggregation,
                               compute_path_dsjctn, Result_element)
from gnpy.core.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.core.exceptions import ServiceError
from examples.path_requests_run import (requests_from_json, disjunctions_from_json,
                                        correct_route_list, correct_disjn,
                                        compute_path_with_disjunction)

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
eqpt_filename = DATA_DIR / 'eqpt_config.json'


@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'CORONET_Global_Topology.xls':   DATA_DIR / 'CORONET_Global_Topology_expected.json',
    DATA_DIR / 'testTopology.xls':     DATA_DIR / 'testTopology_expected.json',
    }.items())
def test_excel_json_generation(xls_input, expected_json_output):
    """ tests generation of topology json
    """
    convert_file(xls_input)

    actual_json_output = xls_input.with_suffix('.json')
    with open(actual_json_output, encoding='utf-8') as f:
        actual = load(f)
    unlink(actual_json_output)

    with open(expected_json_output, encoding='utf-8') as f:
        expected = load(f)

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
                         {DATA_DIR / 'CORONET_Global_Topology.xls':\
                          DATA_DIR / 'CORONET_Global_Topology_auto_design_expected.json',
                          DATA_DIR / 'testTopology.xls':\
                          DATA_DIR / 'testTopology_auto_design_expected.json',
                         }.items())
def test_auto_design_generation_fromxlsgainmode(xls_input, expected_json_output):
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
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    save_network(xls_input, network)

    actual_json_output = f'{str(xls_input)[0:len(str(xls_input))-4]}_auto_design.json'

    with open(actual_json_output, encoding='utf-8') as f:
        actual = load(f)
    unlink(actual_json_output)

    with open(expected_json_output, encoding='utf-8') as f:
        expected = load(f)

    results = compare_networks(expected, actual)
    assert not results.elements.missing
    assert not results.elements.extra
    assert not results.elements.different
    assert not results.connections.missing
    assert not results.connections.extra
    assert not results.connections.different

#test that autodesign creates same file as an input file already autodesigned
@pytest.mark.parametrize('json_input,expected_json_output',
                         {DATA_DIR / 'CORONET_Global_Topology_auto_design_expected.json':\
                          DATA_DIR / 'CORONET_Global_Topology_auto_design_expected.json',
                          DATA_DIR / 'testTopology_auto_design_expected.json':\
                          DATA_DIR / 'testTopology_auto_design_expected.json',
                         }.items())
def test_auto_design_generation_fromjson(json_input, expected_json_output):
    """test that autodesign creates same file as an input file already autodesigned
    """
    equipment = load_equipment(eqpt_filename)
    network = load_network(json_input, equipment)
    # in order to test the Eqpt sheet and load gain target,
    # change the power-mode to False (to be in gain mode)
    equipment['Span']['default'].power_mode = False
    # Build the network once using the default power defined in SI in eqpt config

    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    save_network(json_input, network)

    actual_json_output = f'{str(json_input)[0:len(str(json_input))-5]}_auto_design.json'

    with open(actual_json_output, encoding='utf-8') as f:
        actual = load(f)
    unlink(actual_json_output)

    with open(expected_json_output, encoding='utf-8') as f:
        expected = load(f)

    results = compare_networks(expected, actual)
    assert not results.elements.missing
    assert not results.elements.extra
    assert not results.elements.different
    assert not results.connections.missing
    assert not results.connections.extra
    assert not results.connections.different

# test services creation
@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'testTopology.xls':     DATA_DIR / 'testTopology_services_expected.json',
    DATA_DIR / 'testService.xls':     DATA_DIR / 'testService_services_expected.json'
    }.items())
def test_excel_service_json_generation(xls_input, expected_json_output):
    """ test services creation
    """
    convert_service_sheet(xls_input, eqpt_filename)

    actual_json_output = f'{str(xls_input)[:-4]}_services.json'
    with open(actual_json_output, encoding='utf-8') as f:
        actual = load(f)
    unlink(actual_json_output)

    with open(expected_json_output, encoding='utf-8') as f:
        expected = load(f)

    results = compare_services(expected, actual)
    assert not results.requests.missing
    assert not results.requests.extra
    assert not results.requests.different
    assert not results.synchronizations.missing
    assert not results.synchronizations.extra
    assert not results.synchronizations.different

    # TODO verify that requested bandwidth is not zero !

# test xls answers creation
@pytest.mark.parametrize('json_input, csv_output', {
    DATA_DIR / 'testTopology_response.json':     DATA_DIR / 'testTopology_response',
}.items())
def test_csv_response_generation(json_input, csv_output):
    """ tests if generated csv is consistant with expected generation
        same columns (order not important)
    """
    with open(json_input) as jsonfile:
        json_data = load(jsonfile)
    equipment = load_equipment(eqpt_filename)
    csv_filename = str(csv_output)+'.csv'
    with open(csv_filename, 'w', encoding='utf-8') as fcsv:
        jsontocsv(json_data, equipment, fcsv)

    expected_csv_filename = str(csv_output)+'_expected.csv'

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
        if not key in exp_resp.keys():
            print(f'{key} is not expected')
            return False
        if isinstance(act_resp[key], dict):
            test = compare_response(exp_resp[key], act_resp[key])
    if test:
        for key in exp_resp.keys():
            if not key in act_resp.keys():
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
    DATA_DIR / 'testTopology.xls':     DATA_DIR / 'testTopology_response.json',
}.items())
def test_json_response_generation(xls_input, expected_response_file):
    """ tests if json response is correctly generated for all combinations of requests
    """
    data = convert_service_sheet(xls_input, eqpt_filename)
    # change one of the request with bidir option to cover bidir case as well
    data['path-request'][2]['bidirectional'] = True

    equipment = load_equipment(eqpt_filename)
    network = load_network(xls_input, equipment)
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    oms_list = build_oms_list(network, equipment)
    rqs = requests_from_json(data, equipment)
    rqs = correct_route_list(network, rqs)
    dsjn = disjunctions_from_json(data)
    dsjn = correct_disjn(dsjn)
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
            error_handled = False
            try:
                temp_result = {
                    'response': Result_element(my_rq, pth, reversed_propagatedpths[i]).json}
            except ServiceError:
                error_handled = True
            if not error_handled:
                print('Service error with M=0 not correctly handled')
                raise AssertionError()
            error_handled = False
            my_rq.blocking_reason = 'NO_SPECTRUM'
            try:
                temp_result = {
                    'response': Result_element(my_rq, pth, reversed_propagatedpths[i]).json}
                print(temp_result)
            except ServiceError:
                error_handled = True
            if error_handled:
                print('Service error with NO_SPECTRUM blocking reason not correctly handled')
                raise AssertionError()

        result.append(Result_element(rqs[i], pth, reversed_propagatedpths[i]))

    temp = {
        'response': [n.json for n in result]
    }
    # load expected result and compare keys and values

    with open(expected_response_file) as jsonfile:
        expected = load(jsonfile)
        # since we changes bidir attribute of request#2, need to add the corresponding
        # metric in response

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

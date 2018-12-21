#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
# Module name : test_automaticmodefeature.py
# Version : 
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
checks that example files of example folder work correctly and give same results

"""

from pathlib import Path
import pytest
from gnpy.core.equipment import load_equipment,  automatic_nch
from gnpy.core.network import load_network, build_network
from gnpy.core.request import requests_aggregation, compute_path_dsjctn, Result_element
from examples.path_requests_run import (requests_from_json , correct_route_list ,
                                        load_requests , disjunctions_from_json, correct_disjn,
                                        compute_path_with_disjunction,path_result_json)
from gnpy.core.utils import db2lin, lin2db
from json import load
from tests.compare import compare_paths

TEST_DIR = Path(__file__).parent.parent
DATA_DIR = TEST_DIR/ 'tests/data/'
EXAMPLE_DIR = TEST_DIR / 'examples'
eqpt_filename = EXAMPLE_DIR / 'eqpt_config.json'


@pytest.mark.parametrize('xls_input,expected_json_output', {
    EXAMPLE_DIR / 'meshTopologyExampleV2.xls':     DATA_DIR / 'result1.json',
    EXAMPLE_DIR / 'meshTopologyToy.xls':     DATA_DIR / 'result2.json'}.items())
def test_path_requests_run_example_files(xls_input, expected_json_output):
    print(xls_input)
    print(expected_json_output)
    data = load_requests(xls_input,eqpt_filename)
    equipment = load_equipment(eqpt_filename)
    network = load_network(xls_input,equipment)

    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max 
    p_db = equipment['SI']['default'].power_dbm
    
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)

    rqs = requests_from_json(data, equipment)

    # check that request ids are unique. Non unique ids, may 
    # mess the computation : better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for a in list(set(all_ids)):
            all_ids.remove(a)
        msg = f'Requests id {all_ids} are not unique'
        logger.critical(msg)
        exit()
    rqs = correct_route_list(network, rqs)

    dsjn = disjunctions_from_json(data)
    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = correct_disjn(dsjn)
        
    # Aggregate demands with same exact constraints
    rqs,dsjn = requests_aggregation(rqs,dsjn)
    # TODO export novel set of aggregated demands in a json file
    
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    propagatedpths = compute_path_with_disjunction(network, equipment, rqs, pths)

    result = []
    # assumes that list of rqs and list of propgatedpths have same order
    for i,p in enumerate(propagatedpths):
        result.append(Result_element(rqs[i],p))
    actual = path_result_json(result)
    with open(expected_json_output, encoding='utf-8') as f:
        expected = load(f)

    results = compare_paths(expected, actual)
    assert not results.paths.missing
    assert not results.paths.extra
    assert not results.paths.different

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2018-06-15

from gnpy.core.elements import Edfa
import numpy as np
from json import load
import pytest
from gnpy.core import network_from_json
from gnpy.core.elements import Transceiver, Fiber, Edfa
from gnpy.core.utils import lin2db, db2lin
from gnpy.core.info import SpectralInformation, Channel, Power
from gnpy.core.network import save_network, build_network
from tests.compare import compare_networks, compare_services
from gnpy.core.convert import convert_file
from gnpy.core.service_sheet import convert_service_sheet
from gnpy.core.equipment import load_equipment, automatic_nch
from gnpy.core.network import load_network
from pathlib import Path
import filecmp
from os import unlink

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
eqpt_filename = DATA_DIR / 'eqpt_config.json'

# adding tests to check the parser non regression
# convention of naming of test files:
#
#    - ..._expected.json for the reference output

@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'excelTestFile.xls':             DATA_DIR / 'excelTestFile_expected.json',
    DATA_DIR / 'CORONET_Global_Topology.xls':   DATA_DIR / 'CORONET_Global_Topology_expected.json',
    DATA_DIR / 'meshTopologyExampleV2.xls':     DATA_DIR / 'meshTopologyExampleV2_expected.json',
    DATA_DIR / 'meshTopologyExampleV2Eqpt.xls': DATA_DIR / 'meshTopologyExampleV2Eqpt_expected.json',
 }.items())
def test_excel_json_generation(xls_input, expected_json_output):
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

# assume json entries
# test that the build network gives correct results
# TODO !!
@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'excelTestFile.xls':             DATA_DIR / 'excelTestFile_auto_design_expected.json',
    DATA_DIR / 'CORONET_Global_Topology.xls':   DATA_DIR / 'CORONET_Global_Topology_auto_design_expected.json',
    DATA_DIR / 'meshTopologyExampleV2.xls':     DATA_DIR / 'meshTopologyExampleV2_auto_design_expected.json',
    DATA_DIR / 'meshTopologyExampleV2Eqpt.xls': DATA_DIR / 'meshTopologyExampleV2Eqpt_auto_design_expected.json',
 }.items())
def test_auto_design_generation(xls_input, expected_json_output):
    equipment = load_equipment(eqpt_filename)
    network = load_network(xls_input,equipment)

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
# test services creation

@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'excelTestFile.xls':             DATA_DIR / 'excelTestFile_services_expected.json',
    DATA_DIR / 'meshTopologyExampleV2.xls':     DATA_DIR / 'meshTopologyExampleV2_services_expected.json',
    DATA_DIR / 'meshTopologyExampleV2Eqpt.xls': DATA_DIR / 'meshTopologyExampleV2Eqpt_services_expected.json',
}.items())
def test_excel_service_json_generation(xls_input, expected_json_output):
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

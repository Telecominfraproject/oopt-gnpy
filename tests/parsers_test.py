#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2018-06-15

from gnpy.core.elements import Edfa
import numpy as np
from json import load, dumps
import pytest
from gnpy.core import network_from_json
from gnpy.core.elements import Transceiver, Fiber, Edfa
from gnpy.core.utils import lin2db, db2lin
from gnpy.core.info import SpectralInformation, Channel, Power
from examples.compare_json import compare_network_file, compare_service_file, compare_result_file
from gnpy.core.convert import convert_file
from examples.convert_service_sheet import convert_service_sheet
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
    result, _ = compare_network_file(expected_json_output, actual_json_output)
    unlink(actual_json_output)
    assert result

# assume json entries
# test that the build network gives correct results
# TODO !!

@pytest.mark.parametrize('xls_input,expected_json_output', {
    DATA_DIR / 'excelTestFile.xls':             DATA_DIR / 'excelTestFile_services_expected.json',
    DATA_DIR / 'meshTopologyExampleV2.xls':     DATA_DIR / 'meshTopologyExampleV2_services_expected.json',
    DATA_DIR / 'meshTopologyExampleV2Eqpt.xls': DATA_DIR / 'meshTopologyExampleV2Eqpt_services_expected.json',
}.items())
def test_excel_service_json_generation(xls_input, expected_json_output):
    convert_service_sheet(xls_input, eqpt_filename)
    actual_json_output = f'{str(xls_input)[:-4]}_services.json'
    result, _ = compare_service_file(expected_json_output, actual_json_output)
    unlink(actual_json_output)
    assert result

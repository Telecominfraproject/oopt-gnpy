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


network_file_name = 'tests/test_network.json'
eqpt_library_name = 'examples/eqpt_config.json'

# adding tests to check the parser non regression
# convention of naming of test files:
# 
#    - ..._expected.json for the reference output

excel_filename = ['tests/excelTestFile.xls',
 'examples/CORONET_Global_Topology.xls',
 'tests/meshTopologyExampleV2.xls',
 'tests/meshTopologyExampleV2Eqpt.xls']
network_test_filenames = {
 'tests/excelTestFile.xls'             : 'tests/excelTestFile_expected.json',
 'examples/CORONET_Global_Topology.xls': 'tests/CORONET_Global_Topology_expected.json',
 'tests/meshTopologyExampleV2.xls'     : 'tests/meshTopologyExampleV2_expected.json',
 'tests/meshTopologyExampleV2Eqpt.xls' : 'tests/meshTopologyExampleV2Eqpt_expected.json'}
@pytest.mark.parametrize("inputfile",excel_filename)
def test_excel_json_generation(inputfile) :
    convert_file(Path(inputfile)) 
    # actual
    json_filename = f'{inputfile[:-3]}json'
    # expected
    expected_filename = network_test_filenames[inputfile]
     
    assert compare_network_file(expected_filename,json_filename)[0] is True

# assume json entries
# test that the build network gives correct results     
# TODO !!

excel_filename = ['tests/excelTestFile.xls',
 'tests/meshTopologyExampleV2.xls',
 'tests/meshTopologyExampleV2Eqpt.xls']
service_test_filenames = {
 'tests/excelTestFile.xls'             : 'tests/excelTestFile_services_expected.json',
 'tests/meshTopologyExampleV2.xls'     : 'tests/meshTopologyExampleV2_services_expected.json',
 'tests/meshTopologyExampleV2Eqpt.xls' : 'tests/meshTopologyExampleV2Eqpt_services_expected.json'}
@pytest.mark.parametrize("inputfile",excel_filename)
def test_excel_service_json_generation(inputfile) :
    convert_service_sheet(Path(inputfile),eqpt_library_name) 
    # actual
    json_filename = f'{inputfile[:-4]}_services.json'
    # expected
    test_filename = service_test_filenames[inputfile]
     
    assert compare_service_file(test_filename,json_filename)[0] is True
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2023-09-29
"""
@author: esther.lerouzic
checks all possibilities of this function

"""

from pathlib import Path
import pytest

from gnpy.core.equipment import trx_mode_params
from gnpy.core.exceptions import EquipmentConfigError
from gnpy.tools.json_io import load_equipment, load_json, _equipment_from_json


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_LIBRARY_NAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILE_NAME = DATA_DIR / 'testTopology_expected.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}


@pytest.mark.parametrize('trx_type, trx_mode, error_message, no_error, expected_result',
    [('', '', False, True, "SI"),
     ('', '', True, False, 'Could not find transponder "" in equipment library'),
     ('vendorA_trx-type1', '', True, False,
      'Could not find transponder "vendorA_trx-type1" with mode "" in equipment library'),
     ('vendorA_trx-type1', '', False, True, "SI"),
     ('', 'mode 1', True, False, 'Could not find transponder "" in equipment library'),
     ('', 'mode 1', False, True, "SI"),
     ('vendorA_trx-type1', 'mode 2', True, True, 'mode 2'),
     ('vendorA_trx-type1', 'mode 2', False, True, 'mode 2'),
     ('wrong type', '', True, False, 'Could not find transponder "wrong type" in equipment library'),
     ('wrong type', '', False, True, 'SI'),
     ('vendorA_trx-type1', 'wrong mode', True, False,
      'Could not find transponder "vendorA_trx-type1" with mode "wrong mode" in equipment library'),
     ('vendorA_trx-type1', 'wrong mode', False, True, 'SI'),
     ('wrong type', 'wrong mode', True, False, 'Could not find transponder "wrong type" in equipment library'),
     ('wrong type', 'wrong mode', False, True, 'SI'),
     ('vendorA_trx-type1', None, True, True, 'None'),
     ('vendorA_trx-type1', None, False, True, 'None'),
     (None, None, True, False, 'Could not find transponder "None" in equipment library'),
     (None, None, False, True, 'SI'),
     (None, 'mode 2', True, False, 'Could not find transponder "None" in equipment library'),
     (None, 'mode 2', False, True, 'SI'),
     ])
def test_trx_mode_params(trx_type, trx_mode, error_message, no_error, expected_result):
    """Checks all combinations of trx_type and mode
    """
    possible_results = {}
    possible_results["SI"] = {
        'OSNR': None,
        'baud_rate': 32000000000.0,
        'bit_rate': None,
        'cost': None,
        'equalization_offset_db': 0,
        'f_max': 196100000000000.0,
        'f_min': 191300000000000.0,
        'min_spacing': None,
        'penalties': {},
        'roll_off': 0.15,
        'spacing': 50000000000.0,
        'tx_osnr': 100,
    }
    possible_results["mode 2"] = {
        'format': 'mode 2',
        'baud_rate': 64e9,
        'OSNR': 15,
        'bit_rate': 200e9,
        'roll_off': 0.15,
        'tx_osnr': 100,
        'equalization_offset_db': 0,
        'min_spacing': 75e9,
        'f_max': 196100000000000.0,
        'f_min': 191350000000000.0,
        'penalties': {},
        'cost': 1
    }
    possible_results["None"] = {
        'format': 'undetermined',
        'baud_rate': None,
        'OSNR': None,
        'bit_rate': None,
        'roll_off': None,
        'tx_osnr': None,
        'equalization_offset_db': 0,
        'min_spacing': None,
        'f_max': 196100000000000.0,
        'f_min': 191350000000000.0,
        'penalties': None,
        'cost': None
    }
    equipment = load_equipment(EQPT_LIBRARY_NAME, EXTRA_CONFIGS)
    if no_error:
        trx_params = trx_mode_params(equipment, trx_type, trx_mode, error_message)
        print(trx_params)
        assert trx_params == possible_results[expected_result]
    else:
        with pytest.raises(EquipmentConfigError, match=expected_result):
            _ = trx_mode_params(equipment, trx_type, trx_mode, error_message)


@pytest.mark.parametrize('baudrate, spacing, error_message',
    [(60e9, 50e9, 'Inconsistency in equipment library:\n Transponder "vendorB_trx-type1" mode "wrong mode" '
                  + 'has baud rate 60.00 GHz greater than min_spacing 50.00.'),
     (32e9, 50, 'Inconsistency in equipment library:\n Transponder "vendorB_trx-type1" mode "wrong mode" '
                + 'has baud rate 32.00 GHz greater than min_spacing 0.00.')])
def test_wrong_baudrate_spacing(baudrate, spacing, error_message):
    """Checks wrong values for baudrate and spacing correctly raise an error
    """
    json_data = load_json(EQPT_LIBRARY_NAME)
    wrong_transceiver = {
        'type_variety': 'vendorB_trx-type1',
        'frequency': {
            'min': 191.35e12,
            'max': 196.1e12
        },
        'mode': [{
            'format': 'PS_SP64_1',
            'baud_rate': 32e9,
            'OSNR': 11,
            'bit_rate': 100e9,
            'roll_off': 0.15,
            'tx_osnr': 100,
            'min_spacing': 50e9,
            'cost': 1,
            'penalties': [{
                'chromatic_dispersion': 80000,
                'penalty_value': 0.5
            }, {
                'pmd': 120,
                'penalty_value': 0.5}],
            'equalization_offset_db': 0
        }, {
            'format': 'wrong mode',
            'baud_rate': baudrate,
            'OSNR': 11,
            'bit_rate': 100e9,
            'roll_off': 0.15,
            'tx_osnr': 40,
            'min_spacing': spacing,
            'cost': 1,
            'penalties': [{
                'chromatic_dispersion': 80000,
                'penalty_value': 0.5
            }, {
                'pmd': 120,
                'penalty_value': 0.5}],
            'equalization_offset_db': 0}]
    }
    json_data['Transceiver'].append(wrong_transceiver)
    equipment = _equipment_from_json(json_data, EXTRA_CONFIGS)

    with pytest.raises(EquipmentConfigError, match=error_message):
        _ = trx_mode_params(equipment, 'vendorB_trx-type1', 'wrong mode', error_message=False)

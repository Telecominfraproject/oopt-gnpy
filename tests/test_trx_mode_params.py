#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_trx_mode_params
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
checks all possibilities of this function
"""

from pathlib import Path
from numpy.testing import assert_allclose
import pytest

from gnpy.core.equipment import trx_mode_params
from gnpy.core.exceptions import EquipmentConfigError
from gnpy.tools.json_io import load_equipment, load_json, _equipment_from_json, requests_from_json
from gnpy.tools.default_edfa_config import DEFAULT_EXTRA_CONFIG
from gnpy.core.elements import Transceiver
from gnpy.core.info import create_input_spectral_information


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_LIBRARY_NAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILE_NAME = DATA_DIR / 'testTopology_expected.json'


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
        'tx_channel_power_min': None,
        'tx_channel_power_max': None,
        'rx_channel_power_min': None,
        'rx_channel_power_max': None,
        'rx_ref_channel_power': None
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
        'cost': None,
        'tx_channel_power_min': None,
        'tx_channel_power_max': None,
        'rx_channel_power_min': None,
        'rx_channel_power_max': None,
        'rx_ref_channel_power': None
    }
    equipment = load_equipment(EQPT_LIBRARY_NAME, DEFAULT_EXTRA_CONFIG)
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
    equipment = _equipment_from_json(json_data, DEFAULT_EXTRA_CONFIG)

    with pytest.raises(EquipmentConfigError, match=error_message):
        _ = trx_mode_params(equipment, 'vendorB_trx-type1', 'wrong mode', error_message=False)


def test_transceiver_power_range():
    """
    """

    trx_lib = {
        "SI": [{
            "type_variety": "default",
            "f_min": 191.3e12,
            "f_max": 196.1e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            "tx_power_dbm": -1,
            "power_range_db": [
                0,
                0,
                0.5
            ],
            "roll_off": 0.15,
            "tx_osnr": 100,
            "sys_margins": 0,
            "use_si_channel_count_for_design": False
        }],
        "Transceiver": [{
            "type_variety": "Voyager",
            "frequency": {
                "min": 191.35e12,
                "max": 196.1e12
            },
            "mode": [{
                "format": "mode 1",
                "baud_rate": 32e9,
                "OSNR": 11,
                "bit_rate": 100e9,
                "roll_off": 0.15,
                "tx_osnr": 40,
                "min_spacing": 50e9,
                "tx-channel-power-min": 0,
                "tx-channel-power-max": 5,
                "rx-channel-power-min": -29,
                "rx-channel-power-max": -19,
                "rx-ref-channel-power": -20,
                "penalties": [{
                    "rx_power": -20,
                    "penalty_value": 0
                }, {
                    "rx_power": -21,
                    "penalty_value": 1
                }, {
                    "rx_power": -25,
                    "penalty_value": 2
                }],
                "cost": 1
            }]
        }]
    }

    request_data = {
        "path-request": [{
            "request-id": "0",
            "source": "trx Lorient_KMA",
            "destination": "trx Vannes_KBE",
            "src-tp-id": "trx Lorient_KMA",
            "dst-tp-id": "trx Vannes_KBE",
            "bidirectional": False,
            "path-constraints": {
                "te-bandwidth": {
                    "technology": "flexi-grid",
                    "trx_type": "Voyager",
                    "trx_mode": "mode 1",
                    "spacing": 50e9,
                    "path_bandwidth": 100e9
                }
            }
        }]
    }

    eqpt_trx = _equipment_from_json(trx_lib, DEFAULT_EXTRA_CONFIG)

    # transceiver
    trx = Transceiver(uid='transceiver_1')

    # request
    [rq] = requests_from_json(request_data, eqpt_trx)

    assert rq.tx_channel_power_min == 0
    assert rq.tx_channel_power_max == 5
    assert rq.rx_channel_power_min == -29
    assert rq.rx_channel_power_max == -19
    assert rq.rx_ref_channel_power == -20

    # peigne WDM
    spectral_info = create_input_spectral_information(
        f_min=rq.f_min, f_max=rq.f_max, roll_off=rq.roll_off, baud_rate=rq.baud_rate,
        spacing=rq.spacing, tx_osnr=rq.tx_osnr, tx_power=rq.tx_power, delta_pdb=rq.offset_db)

    assert trx.rx_power_dbm is None
    trx(spectral_info)
    assert_allclose(trx.rx_power_dbm, -1, 1e-3)
    print(trx)

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
import pytest
from numpy.testing import assert_allclose
from numpy import inf

from gnpy.core.equipment import trx_mode_params
from gnpy.core.exceptions import EquipmentConfigError
from gnpy.tools.json_io import load_equipment, load_json, _equipment_from_json, requests_from_json
from gnpy.tools.default_edfa_config import DEFAULT_EXTRA_CONFIG
from gnpy.core.elements import Transceiver
from gnpy.core.info import create_input_spectral_information
from gnpy.core.utils import dbm2watt

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_LIBRARY_NAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILE_NAME = DATA_DIR / 'testTopology_expected.json'


def generate_trx_lib(key: str) -> dict:
    """ Generates a transciever library with a dynamic mode configuration based on the provided key.

    :param key:  A string corresponding to mode scenario.
    :type key : str
    :return: A dictionary containing the mode configuration, including parameters such as
            'format', 'baud_rate', 'OSNR', etc
    :rtype: dict
    """

    return {
        "SI": [{
            "type_variety": "default",
            "f_min": 191.3e12,
            "f_max": 196.1e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            "power_range_db": [0, 0, 0.5],
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
            "mode": [mode(key)]
        }]
    }


def generate_request_data(key: str, tx_power: float) -> dict:
    """
    Generate request data dictionary for a specific transceiver mode and power.

    :param key: The mode key for the transceiver configuration.
    :type key: str
    :param rx_power: The transceiver power in dBm.
    :type rx_power: float
    :return: A dictionary ormatted for request processing, including the path request details.
    :rtype: dict
    """
    return {
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
                    "trx_mode": key,
                    "spacing": 50e9,
                    "tx_power": dbm2watt(tx_power),
                    "path_bandwidth": 100e9
                }
            }
        }]
    }


@pytest.mark.parametrize('trx_type, trx_mode, error_message, no_error, expected_result', [
    ('', '', False, True, "SI"),
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


def mode(key: str) -> dict:
    """
    Generate a mode configuration dictionary based on the provided key.

    :param key: The key indicating the mode scenario, which determines the configuration details.
    :type key: str
    :return: A dictionary containing configuration parameters such as format, baud rate, OSNR,
        bit rate, roll-off, tx_osnr, min_spacing, penalties, and optional rx channel power boundaries.
    :rtype: dict
    The returned dictionary includes various settings for a transceiver mode, with optional
    rx channel power boundaries depending on the key provided.
    """
    penalties_cases = {
        "penalties given": [
            {"rx_power_dbm": -20, "penalty_value": 0},
            {"rx_power_dbm": -21, "penalty_value": 1},
            {"rx_power_dbm": -25, "penalty_value": 2}
        ]
    }

    rx_cases = {
        "rx_min and rx_max equals to rx_power_dbm boundaries": {"rx-channel-power-min": -25,
                                                                "rx-channel-power-max": -20,
                                                                "penalties": penalties_cases["penalties given"]},

        "rx-channel-power-min below boundaries": {"rx-channel-power-min": -28,
                                                  "rx-channel-power-max": -20,
                                                  "penalties": penalties_cases["penalties given"]},
        "rx-channel-power-min above boundaries": {"rx-channel-power-min": -24,
                                                  "rx-channel-power-max": -20,
                                                  "penalties": penalties_cases["penalties given"]},
        "rx-channel-power-max below boundaries": {"rx-channel-power-min": -25,
                                                  "rx-channel-power-max": -21,
                                                  "penalties": penalties_cases["penalties given"]},
        "rx-channel-power-max above boundaries": {"rx-channel-power-min": -25,
                                                  "rx-channel-power-max": -15,
                                                  "penalties": penalties_cases["penalties given"]},
        "no rx_min and rx_max given": {"penalties": penalties_cases["penalties given"]},
        "only rx_min given": {"rx-channel-power-min": -25,
                              "penalties": penalties_cases["penalties given"]},
        "only rx_max given": {"rx-channel-power-max": -15,
                              "penalties": penalties_cases["penalties given"]},
        "no penalty given": {"rx-channel-power-min": -25,
                             "rx-channel-power-max": -15},
        "no penalty given and only rx_min": {"rx-channel-power-min": -25},
        "no penalty given and only rx_max": {"rx-channel-power-max": -15},
        "no rx_penalties given": {
            "rx-channel-power-min": -25,
            "rx-channel-power-max": -15,
            "penalties": [{
                'chromatic_dispersion': 80000,
                'penalty_value': 0.5
            }, {
                'pmd': 120,
                'penalty_value': 0.5}]}
    }

    return {
        "format": key,
        "baud_rate": 32e9,
        "OSNR": 11,
        "bit_rate": 100e9,
        "roll_off": 0.15,
        "tx_osnr": 40,
        "min_spacing": 50e9,
        **rx_cases[key],
        "cost": 1}


@pytest.mark.parametrize("key, min_expected_value, max_expected_value",
                         [("rx_min and rx_max equals to rx_power_dbm boundaries", -25, -20),
                          ("rx-channel-power-max above boundaries", -25, -15),
                          ("no rx_min and rx_max given", -25, -20),
                          ("only rx_min given", -25, -20),
                          ("only rx_max given", -25, -15)])
def test_transceiver_power_range_pass(key: str, min_expected_value: float, max_expected_value: float):
    """
    :param key: Test case identifier describing the scenario being tested
    :type key: str
    :param min_expected_value: Expected value for rx_channel_power_min after processing
    :type min_expected_value: float
    :param max_expected_value: Expected value for rx_channel_power_max after processing
    :type max_expected_value: float
    :return: None
    :rtype: None
    """

    trx_lib = generate_trx_lib(key)
    tx_power = -23
    request_data = generate_request_data(key, tx_power)
    eqpt_trx = _equipment_from_json(trx_lib, DEFAULT_EXTRA_CONFIG)

    [rq] = requests_from_json(request_data, eqpt_trx)

    assert rq.rx_channel_power_min == min_expected_value
    assert rq.rx_channel_power_max == max_expected_value


@pytest.mark.parametrize("key, message_error",
                         [("rx-channel-power-min below boundaries", "rx_channel_power_min value must be set"
                          " to the minimum rx_power_dbm value defined in penalties"),
                          ("rx-channel-power-min above boundaries", "rx_channel_power_min value must be set"
                          " to the minimum rx_power_dbm value defined in penalties"),
                          ("rx-channel-power-max below boundaries", "rx_channel_power_max value must be set"
                          " to the maximum rx_power_dbm value defined in penalties")
                          ])
def test_transceiver_power_range_fail(key: str, message_error: str):
    """
    :param key: Test case identifier describing the failure scenario
    :type key: str
    :param min_expected_value: Invalid rx_channel_power_min value that should trigger error
    :type min_expected_value: float
    :param max_expected_value: Invalid rx_channel_power_max value that should trigger error
    :type max_expected_value: float
    :param message_error: Expected error message in the ValueError exception
    :type message_error: str
    :return: None
    :rtype: None
    :raises ValueError: When rx power range values are inconsistent with penalty data
    """
    trx_lib = generate_trx_lib(key)

    with pytest.raises(ValueError) as exc_info:
        _equipment_from_json(trx_lib, DEFAULT_EXTRA_CONFIG)
    assert message_error in str(exc_info.value)


@pytest.mark.parametrize("key, tx_power, expected_penalty",
                         [("rx_min and rx_max equals to rx_power_dbm boundaries", -28, inf),
                          ("rx_min and rx_max equals to rx_power_dbm boundaries", -23, 1.5),
                          ("rx_min and rx_max equals to rx_power_dbm boundaries", -20.9, 0.9),
                          ("rx-channel-power-max above boundaries", -18, 0),
                          ("rx-channel-power-max above boundaries", -14, inf),
                          ("no penalty given", -16, 0),
                          ("no penalty given", -14, inf),
                          ("no penalty given", -24, 0),
                          ("no penalty given and only rx_min", -24, 0),
                          ("no penalty given and only rx_min", -30, inf),
                          ("no penalty given and only rx_max", -14, inf),
                          ("no penalty given and only rx_max", -30, 0),
                          ("no rx_penalties given", -24, 0)
                          ])
def test_transceiver_calc_penalties(key: str, tx_power: float, expected_penalty: float):
    """
    Test the penalty calculation for a transceiver based on received power.
    :param key: Test case identifier describing the scenario being tested
    :type key: str
    :param tx_power: The tx power in dBm (corresponds to rx_power received on the transceiver).
    :type tx_power: float
    :param expected_penalty: The expected penalty value corresponding to the rx_power.
    :type expected_penalty: float
    :return: None
    :rtype: None

    This test generates a transceiver configuration and request data based on the
    provided rx_power, then calculates penalties and asserts that the penalty
    matches the expected value within a tolerance.
    """

    trx_lib = generate_trx_lib(key)
    request_data = generate_request_data(key, tx_power)

    eqpt_trx = _equipment_from_json(trx_lib, DEFAULT_EXTRA_CONFIG)

    [rq] = requests_from_json(request_data, eqpt_trx)

    trx = Transceiver(uid='transceiver_1')

    spectral_info = create_input_spectral_information(
        f_min=rq.f_min, f_max=rq.f_max, roll_off=rq.roll_off, baud_rate=rq.baud_rate,
        spacing=rq.spacing, tx_osnr=rq.tx_osnr, tx_power=rq.tx_power, delta_pdb=rq.offset_db)

    # no propagation yet, no received power at this step
    assert trx.rx_power_dbm is None

    # simulation of back to back transceiver, received power to tx power
    trx(spectral_info)
    trx.calc_penalties(rq.penalties)

    assert_allclose(trx.rx_power_dbm, tx_power, 1e-3)
    assert_allclose(trx.penalties.get('rx_power_dbm'), expected_penalty, 1e-3)

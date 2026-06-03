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
from numpy import inf, zeros, mean
from gnpy.core.equipment import trx_mode_params
from gnpy.core.exceptions import EquipmentConfigError
from gnpy.tools.json_io import load_equipment, load_json, _equipment_from_json, requests_from_json, load_network
from gnpy.core.elements import Transceiver, Edfa
from gnpy.core.info import create_input_spectral_information
from gnpy.core.utils import dbm2watt, lin2db
from gnpy.core.network import build_network
from gnpy.tools.default_edfa_config import DEFAULT_EXTRA_CONFIG
from gnpy.topology.request import PathRequest
from gnpy.tools.cli_examples import load_common_data, load_requests, designed_network, planning


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
DATA_DIR_TRX = DATA_DIR / 'trx'
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
                    "spacing": 200e9,
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
    ('', 'mode 1', True, False, 'Could not find transponder "" in equipment library'),
    ('', 'mode 1', False, True, "SI"),
    ('vendorA_trx-type1', 'mode 2', True, True, 'mode 2'),
    ('vendorA_trx-type1', 'mode 2', False, True, 'mode 2'),
    ('wrong type', '', True, False, 'Could not find transponder "wrong type" in equipment library'),
    ('vendorA_trx-type1', 'wrong mode', True, False,
     'Could not find transponder "vendorA_trx-type1" with mode "wrong mode" in equipment library'),
    ('wrong type', 'wrong mode', True, False, 'Could not find transponder "wrong type" in equipment library'),
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
        'detailed_rx': {},
        'roll_off': 0.15,
        'spacing': 50000000000.0,
        'tx_osnr': 100,
        'tx_channel_power_min_dbm': None,
        'tx_channel_power_max_dbm': None,
        'rx_channel_power_min_dbm': None,
        'rx_channel_power_max_dbm': None
    }
    possible_results["mode 2"] = {
        'format': 'mode 2',
        'baud_rate': 64e9,
        'OSNR': 15,
        'detailed_rx': {},
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
        'detailed_rx': {},
        'tx_osnr': None,
        'equalization_offset_db': 0,
        'min_spacing': None,
        'f_max': 196100000000000.0,
        'f_min': 191350000000000.0,
        'penalties': None,
        'cost': None,
        'tx_channel_power_min_dbm': None,
        'tx_channel_power_max_dbm': None,
        'rx_channel_power_min_dbm': None,
        'rx_channel_power_max_dbm': None
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
    detailed_penalties = {
        "BER_threshold": 0.068,
        "prx_ref_dbm": -10.0,
        "k1": 0.5,
        "k2": 0.5,
        "snr_prx_db_0.1nm": 65 + lin2db(32 / 12.5),  # in 0.1nm
        "snr_trx_db_0.1nm": 9 + lin2db(32 / 12.5),
    }

    rx_cases = {
        "no penalty given and only rx_min": {"rx-channel-power-min-dbm": -25},
        "no penalty given and only rx_max": {"rx-channel-power-max-dbm": -15},
        "no rx_penalties given": {
            "rx-channel-power-min-dbm": -25,
            "rx-channel-power-max-dbm": -15
        },
        "rx_penalties given": {
            "rx-channel-power-min-dbm": -22,
            "rx-channel-power-max-dbm": -10,
            "detailed_rx": detailed_penalties
        }
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
        "tx-channel-power-min-dbm": 0,
        "tx-channel-power-max-dbm": 5,
        "penalties": [{
            'chromatic_dispersion': 80000,
            'penalty_value': 0.5
        }, {
            'pmd': 120,
            'penalty_value': 0.5}],
        "cost": 1}


@pytest.mark.parametrize("key, min_expected_value, max_expected_value",
                         [("no rx_penalties given", -25, -15),
                          ("no penalty given and only rx_min", -25, None),
                          ("no penalty given and only rx_max", None, -15),
                          ("rx_penalties given", -22, -10)])
def test_transceiver_power_range_pass(key: str, min_expected_value: float, max_expected_value: float):
    """
     Verify the RX range(rx_channel_power_min/max) are correctly read from the mode/request.
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

    assert rq.rx_channel_power_min_dbm == min_expected_value
    assert rq.rx_channel_power_max_dbm == max_expected_value


def pathrequest(pch_dbm, p_tot_dbm):
    """create ref channel for defined power settings
    """
    params = {
        "power": dbm2watt(pch_dbm),
        "tx_power": dbm2watt(pch_dbm),
        "nb_channel": round(dbm2watt(p_tot_dbm) / dbm2watt(pch_dbm), 0)
    }
    return PathRequest(**params)


def setup_edfa_variable_gain():
    """init edfa class by reading test_network.json file
    remove all gain and nf ripple"""
    equipment = load_equipment(EQPT_LIBRARY_NAME, DEFAULT_EXTRA_CONFIG)
    network = load_network(NETWORK_FILE_NAME, equipment)
    build_network(network, equipment, pathrequest(0, 20))
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][0]
    edfa.gain_ripple = zeros(96)
    edfa.interpol_nf_ripple = zeros(96)
    return edfa


@pytest.mark.parametrize("key, tx_power, expected_penalty",
                         [("no rx_penalties given", -16, 0),
                          ("no rx_penalties given", -14, inf),
                          ("no rx_penalties given", -24, 0),
                          ("no penalty given and only rx_min", -24, 0),
                          ("no penalty given and only rx_min", -30, inf),
                          ("no penalty given and only rx_max", -14, inf),
                          ("no penalty given and only rx_max", -30, 0),
                          ("rx_penalties given", -22.1, inf),
                          ("rx_penalties given", -20, 0),
                          ("rx_penalties given", -9, inf),
                          ])
def test_transceiver_check_boundaries_penalties(key: str, tx_power: float, expected_penalty: float):
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

    assert rq.tx_channel_power_min_dbm == 0
    assert rq.tx_channel_power_max_dbm == 5

    trx1 = Transceiver(uid='transceiver_1')
    trx2 = Transceiver(uid='transceiver_2')
    edfa = setup_edfa_variable_gain()
    edfa.effective_gain = 15
    edfa.out_voa = 15
    spectral_info = create_input_spectral_information(
        f_min=rq.f_min, f_max=rq.f_max, roll_off=rq.roll_off, baud_rate=rq.baud_rate,
        spacing=rq.spacing, tx_osnr=rq.tx_osnr, tx_power=rq.tx_power, delta_pdb=rq.offset_db)

    # no propagation yet, no received power at this step
    assert trx1.rx_power_dbm is None

    # simulation of back to back transceiver with added noise (EDFA), received power to tx power
    spectral_info = trx1(spectral_info)  # emitted signal
    spectral_info = edfa(spectral_info)
    spectral_info = trx2(spectral_info)  # received signal (back-to-back)
    assert trx2.rx_power_dbm is not None

    rx_power = float(mean(trx2.rx_power_dbm))
    assert rx_power is not None

    # osnr_ase_01nm_expected = 58 + tx_power - NF
    osnr_ase_01nm_expected = 58 + rx_power - 10
    assert_allclose(trx2.osnr_ase_01nm, osnr_ase_01nm_expected, 1e-1)  # formula 58+pin-NF

    trx2.calc_penalties(rq.penalties, rq.rx_channel_power_min_dbm, rq.rx_channel_power_max_dbm)
    assert_allclose(trx2.rx_power_dbm, tx_power, 1e-2)
    assert_allclose(trx2.penalties.get('rx_power_dbm'), expected_penalty, atol=1e-2)


@pytest.mark.parametrize("key, rx_power, expected_snr",
                         [("rx_penalties given", -10.1, 12.96),
                          ("rx_penalties given", -21.9, 11.64),
                          ("rx_penalties given", -22, 11.62)
                          ])
def test_receiver_noise_contribution(key: str, rx_power: float, expected_snr: float):
    """
    Test the expected snr for a transceiver based on received power.
    :param key: Test case identifier describing the scenario being tested
    :type key: str
    :param rx_power: The rx power in dBm (corresponds to rx_power received on the transceiver).
    :type rx_power: float
    :param expected_snr: The expected snr for the rx_power.
    :type expected_snr: float
    :return: None
    :rtype: None

    This test generates a transceiver configuration and request data based on the
    provided rx_power, then calculates rx_snr contribution and asserts that the expected snr
    matches the expected value within a tolerance.
    """

    trx_lib = generate_trx_lib(key)
    # tx_power is fixed to -10dbm
    request_data = generate_request_data(key, -10)

    eqpt_trx = _equipment_from_json(trx_lib, DEFAULT_EXTRA_CONFIG)

    [rq] = requests_from_json(request_data, eqpt_trx)

    trx1 = Transceiver(uid='transceiver_1')
    trx2 = Transceiver(uid='transceiver_2')
    edfa = setup_edfa_variable_gain()
    edfa.effective_gain = 15
    edfa.out_voa = 5 - rx_power
    spectral_info = create_input_spectral_information(
        f_min=rq.f_min, f_max=rq.f_max, roll_off=rq.roll_off, baud_rate=rq.baud_rate,
        spacing=rq.spacing, tx_osnr=rq.tx_osnr, tx_power=rq.tx_power, delta_pdb=rq.offset_db)

    # no propagation yet, no received power at this step
    assert trx1.rx_power_dbm is None

    # simulation of back to back transceiver with added noise (EDFA), received power to tx power
    spectral_info = trx1(spectral_info)  # emitted signal
    spectral_info = edfa(spectral_info)
    spectral_info = trx2(spectral_info)  # received signal (back-to-back)
    assert trx2.rx_power_dbm is not None

    # nf = 10
    # the contribution of noise at the input of the receiver is unchanged
    # osnr_ase_01nm_expected = 58 + tx_power - NF
    osnr_ase_01nm_expected = 58 + -10 - 10
    assert_allclose(trx2.osnr_ase_01nm, osnr_ase_01nm_expected, 1e-1)  # formula 58+pin-NF

    assert_allclose(trx2.rx_power_dbm, rx_power, atol=1e-2)

    trx2.update_rx_snr(rq.detailed_rx)
    snr_after = trx2.esnr_01nm
    assert_allclose(snr_after, expected_snr, atol=1e-2)


def test_detailed_rx():
    (equipment, network) = \
        load_common_data(DATA_DIR_TRX / 'eqpt_config_with_detailed_rx.json',
                         None, None,
                         DATA_DIR_TRX / 'topology.json', None, None)
    network, _, _ = designed_network(equipment, network)
    data = load_requests(DATA_DIR_TRX / 'services_with_detailed_rx.json', equipment, bidir=True,
                         network=network, network_filename=DATA_DIR_TRX / 'topology.json')
    oms_list, propagatedpths, reversed_propagatedpths, rqs, _, result = \
        planning(network, equipment, data, redesign=False)
    assert not hasattr(rqs[0], "blocking_reason")
    assert rqs[1].blocking_reason is not None
    assert rqs[2].blocking_reason is not None
    assert propagatedpths[1][-1].penalties['rx_power_dbm'][0] == float('inf')
    assert propagatedpths[2][-1].penalties['rx_power_dbm'][0] is not float('inf')
    assert reversed_propagatedpths[2][-1].penalties['rx_power_dbm'][0] == float('inf')

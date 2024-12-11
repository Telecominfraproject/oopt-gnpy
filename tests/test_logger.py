# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from pathlib import Path
import re
import pytest

from gnpy.core.exceptions import ConfigurationError, ServiceError, EquipmentConfigError, ParametersError, \
    NetworkTopologyError
from gnpy.tools.json_io import SI, Roadm, Amp, load_equipment, requests_from_json, network_from_json, \
    load_network, load_requests
from gnpy.tools.convert import xls_to_json_data

TEST_DIR = Path(__file__).parent
EQPT_FILENAME = TEST_DIR / 'data/eqpt_config.json'
MULTIBAND_EQPT_FILENAME = TEST_DIR / 'data/eqpt_config_multiband.json'
DATA_DIR = TEST_DIR / 'data'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}


def test_jsonthing(caplog):
    """Check that a missing key correctly raises an info
    """
    json_data = {

        "baud_rate": 32e9,
        "f_max": 196.1e12,
        "spacing": 50e9,
        "power_dbm": 0,
        "power_range_db": [0, 0, 1],
        "roll_off": 0.15,
        "tx_osnr": 40,
        "sys_margins": 2
    }
    _ = SI(**json_data)
    expected_msg = '\n\tWARNING missing f_min attribute in eqpt_config.json[SI]\n' \
                   + '\tdefault value is f_min = 191350000000000.0'
    assert expected_msg in caplog.text


def wrong_equipment():
    """Creates list of malformed equipments
    """
    data = []
    data.append({
        "error": EquipmentConfigError,
        "equipment": Roadm,
        "json_data": {
            "target_pch_out_db": -20,
            "target_out_mWperSlotWidth": 3.125e-4,
            "add_drop_osnr": 38,
            "pmd": 0,
            "pdl": 0,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            }
        },
        "expected_msg": "Only one equalization type should be set in ROADM, found: target_pch_out_db,"
                        + " target_out_mWperSlotWidth"
    })
    data.append({
        "error": EquipmentConfigError,
        "equipment": Roadm,
        "json_data": {
            "add_drop_osnr": 38,
            "pmd": 0,
            "pdl": 0,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            }
        },
        "expected_msg": "No equalization type set in ROADM"
    })
    return data


@pytest.mark.parametrize('error, equipment, json_data, expected_msg',
                         [(e['error'], e['equipment'], e['json_data'], e['expected_msg']) for e in wrong_equipment()])
def test_wrong_equipment(caplog, error, equipment, json_data, expected_msg):
    """
    """
    with pytest.raises(EquipmentConfigError, match=expected_msg):
        _ = equipment(**json_data)


@pytest.mark.parametrize('xls_service_filename, xls_topo_filename, expected_msg',
                         [('wrong_service.xlsx', 'testTopology.xls',
                           "Service error: Request Id: 0 - could not find tsp : 'Voyager' with mode: 'Mode 10' "
                           + "in eqpt library \nComputation stopped."),
                          ('wrong_service_type.xlsx', 'testTopology.xls',
                           "Service error: Request Id: 0 - could not find tsp : 'Galileo' with mode: 'mode 1' "
                           + "in eqpt library \nComputation stopped.")])
def test_wrong_xls_service(xls_service_filename, xls_topo_filename, expected_msg):
    """
    """
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = load_network(DATA_DIR / xls_topo_filename, equipment)
    with pytest.raises(ServiceError, match=expected_msg):
        _ = load_requests(DATA_DIR / xls_service_filename, equipment, False, network, DATA_DIR / xls_topo_filename)


def wrong_amp():
    """Creates list of malformed equipments
    """
    data = []
    data.append({
        "error": EquipmentConfigError,
        "json_data": {
            "type_variety": "test_fixed_gain",
            "type_def": "fixed_gain",
            "gain_flatmax": 21,
            "gain_min": 20,
            "p_max": 21,
            "allowed_for_design": True
        },
        "expected_msg": "missing nf0 value input for amplifier: test_fixed_gain in equipment config"
    })
    data.append({
        "error": EquipmentConfigError,
        "json_data": {
            "type_variety": "test",
            "type_def": "variable_gain",
            "gain_flatmax": 25,
            "gain_min": 15,
            "p_max": 21,
            "nf_min": 5.8,
            "out_voa_auto": False,
            "allowed_for_design": True
        },
        "expected_msg": "missing nf_min or nf_max value input for amplifier: test in equipment config"
    })
    data.append({
        "error": EquipmentConfigError,
        "json_data": {
            "type_variety": "medium+high_power",
            "type_def": "dual_stage",
            "gain_min": 25,
            "preamp_variety": "std_medium_gain",
            "allowed_for_design": False
        },
        "expected_msg": "missing preamp/booster variety input for amplifier: medium+high_power in equipment config"
    })
    return data


@pytest.mark.parametrize('error, json_data, expected_msg',
                         [(e['error'], e['json_data'], e['expected_msg']) for e in wrong_amp()])
def test_wrong_amp(error, json_data, expected_msg):
    """
    """
    with pytest.raises(error, match=re.escape(expected_msg)):
        _ = Amp.from_json(EQPT_FILENAME, **json_data)


def wrong_requests():
    """Creates list of malformed requests
    """
    data = []
    data.append({
        'error': ConfigurationError,
        'json_data': {
            "path-request": [{
                "request-id": "imposed_mode",
                "source": "trx Brest_KLA",
                "destination": "trx Vannes_KBE",
                "src-tp-id": "trx Brest_KLA",
                "dst-tp-id": "trx Vannes_KBE",
                "bidirectional": False,
                "path-constraints": {
                    "te-bandwidth": {
                        "technology": "flexi-grid",
                        "trx_type": "test_offset",
                        "trx_mode": "mode 3",
                        "spacing": 75000000000.0,
                        "path_bandwidth": 100000000000.0
                    }
                }
            }]
        },
        'expected_msg': 'Equipment Config error in imposed_mode: '
                        + 'Could not find transponder "test_offset" in equipment library'
    })
    data.append({
        'error': ServiceError,
        'json_data': {
            "path-request": [{
                "request-id": "Missing_type",
                "source": "trx Brest_KLA",
                "destination": "trx Vannes_KBE",
                "src-tp-id": "trx Brest_KLA",
                "dst-tp-id": "trx Vannes_KBE",
                "bidirectional": False,
                "path-constraints": {
                    "te-bandwidth": {
                        "technology": "flexi-grid",
                        "trx_type": None,
                        "spacing": 75000000000.0,
                        "path_bandwidth": 100000000000.0
                    }
                }
            }]},
        'expected_msg': 'Request Missing_type has no transceiver type defined'
    })
    data.append({
        'error': ServiceError,
        'json_data': {
            "path-request": [{
                "request-id": "wrong_spacing",
                "source": "trx Brest_KLA",
                "destination": "trx Vannes_KBE",
                "src-tp-id": "trx Brest_KLA",
                "dst-tp-id": "trx Vannes_KBE",
                "bidirectional": False,
                "path-constraints": {
                    "te-bandwidth": {
                        "technology": "flexi-grid",
                        "trx_type": "Voyager",
                        "trx_mode": "mode 2",
                        "spacing": 50000000000.0,
                        "path_bandwidth": 100000000000.0
                    }
                }
            }]},
        'expected_msg': 'Request wrong_spacing has spacing below transponder Voyager mode 2 min spacing'
        + ' value 75.0GHz.\nComputation stopped'
    })
    data.append({
        'error': ServiceError,
        'json_data': {
            "path-request": [{
                "request-id": "Wrong_nb_channel",
                "source": "trx Brest_KLA",
                "destination": "trx Vannes_KBE",
                "src-tp-id": "trx Brest_KLA",
                "dst-tp-id": "trx Vannes_KBE",
                "bidirectional": False,
                "path-constraints": {
                    "te-bandwidth": {
                        "technology": "flexi-grid",
                        "trx_type": "Voyager",
                        "trx_mode": "mode 2",
                        "spacing": 75000000000.0,
                        "max-nb-of-channel": 150,
                        "path_bandwidth": 100000000000.0
                    }
                }
            }]},
        'expected_msg': 'Requested channel number 150, baud rate 66.0 GHz'
                        + ' and requested spacing 75.0GHz is not consistent with frequency range'
                        + ' 191.35 THz, 196.1 THz.'
                        + ' Max recommanded nb of channels is 63.'
    })
    data.append({
        'error': ServiceError,
        'json_data': {
            "path-request": [{
                "request-id": "Wrong_M",
                "source": "trx Brest_KLA",
                "destination": "trx Vannes_KBE",
                "src-tp-id": "trx Brest_KLA",
                "dst-tp-id": "trx Vannes_KBE",
                "bidirectional": False,
                "path-constraints": {
                    "te-bandwidth": {
                        "technology": "flexi-grid",
                        "trx_type": "Voyager",
                        "trx_mode": "mode 2",
                        "spacing": 75000000000.0,
                        "effective-freq-slot": [
                            {
                              "N": -208,
                              "M": 4
                            }
                        ],
                        "path_bandwidth": 100000000000.0
                    }
                }
            }]},
        'expected_msg': 'Requested M [{\'N\': -208, \'M\': 4}] number of slots for request Wrong_M '
                        + 'support 0 nb of channels while 1 are required to support request 100.0 Gbit/s'
                        + ' with Voyager mode 2'
    })
    return data


@pytest.mark.parametrize('error, json_data, expected_msg',
                         [(e['error'], e['json_data'], e['expected_msg']) for e in wrong_requests()])
def test_json_request(error, json_data, expected_msg):
    """
    Check that a missing key is correctly raisong the logger
    """
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)

    with pytest.raises(error, match=re.escape(expected_msg)):
        _ = requests_from_json(json_data, equipment)


def wrong_element():
    """
    """
    data = []
    data.append({
        "error": ConfigurationError,
        "json_data": {
            "elements": [{
                "uid": "roadm SITE2",
                "type": "Roadm",
                "params": {
                    "target_pch_out_db": -20,
                    "target_out_mWperSlotWidth": 3.125e-4,
                },
                "metadata": {
                    "location": {
                        "latitude": 2.0,
                        "longitude": 3.0,
                        "city": "SITE2",
                        "region": "RLD"
                    }
                }
            }],
            "connections": []
        },
        "expected_msg": "ROADM roadm SITE2: invalid equalization settings"
    })
    data.append({
        "error": ConfigurationError,
        "json_data": {
            "elements": [{
                "uid": "east edfa in ILA2 to SITE2",
                "type": "Edfa",
                "type_variety": "not_valid_variety",
                "metadata": {
                    "location": {
                        "latitude": 2.0,
                        "longitude": 0.0,
                        "city": "ILA2",
                        "region": "RLD"
                    }
                }
            }],
            "connections": []
        },
        "expected_msg": "The Edfa of variety type not_valid_variety was not recognized:"
                        + "\nplease check it is properly defined in the eqpt_config json file"
    })
    data.append({
        "error": ParametersError,
        "json_data": {
            "elements": [{
                "uid": "fiber (ILA2 → ILA1)",
                "type": "Fiber",
                "type_variety": "SSMF",
                "params": {
                    "length": 100.0,
                    "loss_coef": 0.2,
                    "att_in": 0,
                    "con_in": 0,
                    "con_out": 0
                },
                "metadata": {
                    "location": {
                        "latitude": 2.0,
                        "longitude": 1.5,
                        "city": None,
                        "region": None
                    }
                }
            }],
            "connections": []
        },
        "expected_msg": "Config error in fiber (ILA2 → ILA1): "
                        + "Fiber configurations json must include \'length_units\'. Configuration: "
                        + "{\'length\': 100.0, \'loss_coef\': 0.2, \'att_in\': 0, \'con_in\': 0, \'con_out\': 0, "
                        + "\'type_variety\': \'SSMF\', \'dispersion\': 1.67e-05, \'effective_area\': 8.3e-11, "
                        + "\'pmd_coef\': 1.265e-15}"
    })
    return data


@pytest.mark.parametrize('error, json_data, expected_msg',
                         [(e['error'], e['json_data'], e['expected_msg']) for e in wrong_element()])
def test_json_network(error, json_data, expected_msg):
    """
    Check that a missing key is correctly raisong the logger
    """
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    with pytest.raises(error, match=re.escape(expected_msg)):
        _ = network_from_json(json_data, equipment)


@pytest.mark.parametrize('input_filename, expected_msg',
    [(DATA_DIR / 'wrong_topo_node.xlsx', 'XLS error: The following nodes are not referenced from the Links sheet.'
                                         + ' If unused, remove them from the Nodes sheet:\n - toto'),
     (DATA_DIR / 'wrong_topo_link.xlsx', 'XLS error: The Links sheet references nodes that are not defined in the '
                                         + 'Nodes sheet:\n - ALB -> toto'),
     (DATA_DIR / 'wrong_topo_link_header.xlsx', 'missing header Node Z'),
     (DATA_DIR / 'wrong_topo_eqpt.xlsx', 'XLS error: The Eqpt sheet refers to nodes that are not defined in the '
                                         + 'Nodes sheet:\n - toto'),
     (DATA_DIR / 'wrong_topo_duplicate_node.xlsx', 'Duplicate city: Counter({\'ALB\': 2, \'CHA_3\': 1})'),
     (DATA_DIR / 'wrong_topo_duplicate_eqpt.xlsx', 'XLS error: Duplicate lines in Eqpt sheet: - ALB -> CHA_3'),
     (DATA_DIR / 'wrong_topo_bad_eqpt.xlsx', 'XLS error: The Eqpt sheet references links that are not defined '
                                             + 'in the Links sheet:\n - toto -> CHA_3'),
     (DATA_DIR / 'wrong_duplicate_link_reverse.xlsx', 'XLS error: links  - (\'ila\', \'siteb\') are duplicate'),
     (DATA_DIR / 'wrong_duplicate_eqpt_ila_reverse.xlsx', 'XLS error: Duplicate ILA eqpt definition in Eqpt sheet:'
                                                          + ' - ila')])
def test_wrong_xlsx(input_filename, expected_msg):
    """Check that error and logs are correctly working
    """
    with pytest.raises(NetworkTopologyError, match=re.escape(expected_msg)):
        _ = xls_to_json_data(input_filename)


@pytest.mark.parametrize('input_filename, expected_msg',
    [(DATA_DIR / 'wrong_node_type.xlsx', 'invalid node type (ILA) specified in Lannion_CAS, replaced by ROADM\n')])
def test_log_wrong_xlsx(caplog, input_filename, expected_msg):
    """Check that logs are correctly working
    """
    _ = xls_to_json_data(input_filename)
    assert expected_msg in caplog.text


def wrong_configs():
    wrong_config = [[{
        "uid": "Edfa1",
        "type": "Edfa",
        "type_variety": "std_medium_gain_multiband",
        "operational": {
            "gain_target": 21,
            "delta_p": 3.0,
            "out_voa": 3.0,
            "in_voa": 0.0,
            "tilt_target": 0.0,
            "f_min": 186.2,
            "f_max": 190.2
        }},
        ParametersError]
    ]
    wrong_config.append([{
        "uid": "[83/WR-2-4-SIG=>930/WRT-1-2-SIG]-PhysicalConn/2056",
        "type": "Fiber",
        "type_variety": "SSMF",
        "params": {
            "dispersion_per_frequency": {
                "frequency": [
                    185.49234135667396e12,
                    186.05251641137855e12,
                    188.01312910284463e12,
                    189.99124726477024e12],
                "value": [
                    1.60e-05,
                    1.67e-05,
                    1.7e-05,
                    1.8e-05]
            },
            "loss_coef": 2.85,
            "length_units": "km",
            "att_in": 0.0,
            "con_in": 0.0,
            "con_out": 0.0
        }},
        ParametersError
    ])
    wrong_config.append([{
        "uid": "east edfa in Lannion_CAS to Stbrieuc",
        "metadata": {
            "location": {
                "city": "Lannion_CAS",
                "region": "RLD",
                "latitude": 2.0,
                "longitude": 0.0
            }
        },
        "type": "Multiband_amplifier",
        "type_variety": "std_low_gain_multiband",
        "amplifiers": [{
            "type_variety": "std_low_gain",
            "operational": {
                "gain_target": 20.0,
                "delta_p": 0,
                "tilt_target": 0,
                "out_voa": 0
            }
        }, {
            "type_variety": "std_low_gain_L_ter",
            "operational": {
                "gain_target": 20.0,
                "delta_p": 1,
                "tilt_target": 0,
                "out_voa": 1
            }
        }]},
        ConfigurationError])
    wrong_config.append([{
        "uid": "east edfa in Lannion_CAS to Stbrieuc",
        "metadata": {
            "location": {
                "city": "Lannion_CAS",
                "region": "RLD",
                "latitude": 2.0,
                "longitude": 0.0
            }
        },
        "type": "Edfa",
        "type_variety": "std_low_gain_multiband",
        "amplifiers": [{
            "type_variety": "std_low_gain",
            "operational": {
                "gain_target": 20.0,
                "delta_p": 0,
                "tilt_target": 0,
                "out_voa": 0
            }
        }, {
            "type_variety": "std_low_gain_L",
            "operational": {
                "gain_target": 20.0,
                "delta_p": 1,
                "tilt_target": 0,
                "out_voa": 1
            }
        }]},
        ParametersError])
    wrong_config.append([{
        "uid": "east edfa in Lannion_CAS to Stbrieuc",
        "metadata": {
            "location": {
                "city": "Lannion_CAS",
                "region": "RLD",
                "latitude": 2.0,
                "longitude": 0.0
            }
        },
        "type": "Multiband_amplifier",
        "type_variety": "std_low_gain_multiband",
        "amplifiers": [{
            "type_variety": "std_low_gain",
            "operational": {
                "gain_target": 20.0,
                "delta_p": 0,
                "tilt_target": 0,
                "out_voa": 0
            }
        }, {
            "type_variety": "std_low_gain_L",
            "operational": {
                "gain_target": 20.0,
                "delta_p": 1,
                "tilt_target": 0,
                "out_voa": 1
            }
        }, {
            "type_variety": "std_low_gain",
            "operational": {
                "gain_target": 20.0,
                "delta_p": 1,
                "tilt_target": 0,
                "out_voa": 1
            }
        }]},
        ParametersError])
    return wrong_config


@pytest.mark.parametrize('el_config, error_type', wrong_configs())
def test_wrong_multiband(el_config, error_type):

    equipment = load_equipment(MULTIBAND_EQPT_FILENAME, EXTRA_CONFIGS)
    fused_config = {
        "uid": "[83/WR-2-4-SIG=>930/WRT-1-2-SIG]-Tl/9300",
        "type": "Fused",
        "params": {
            "loss": 20
        }
    }
    json_data = {
        "elements": [
            el_config,
            fused_config
        ],
        "connections": []
    }
    with pytest.raises(error_type):
        _ = network_from_json(json_data, equipment)

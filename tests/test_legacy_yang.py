#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_legacy_yang
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
Test conversion legacy to yang utils
====================================
check that combinations of inputs are correctly converted
"""
from pathlib import Path
import json
import subprocess    # nosec
import os
import pytest

from gnpy.tools.json_io import load_gnpy_json
from gnpy.tools.cli_examples import transmission_main_example, path_requests_run
from gnpy.tools.convert_legacy_yang import legacy_to_yang, yang_to_legacy
from gnpy.tools.yang_convert_utils import convert_delta_power_range


SRC_ROOT = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data' / 'convert'
EXAMPLE_DIR = SRC_ROOT / 'gnpy' / 'example-data'


@pytest.mark.parametrize('yang_input, expected_output, expected_autodesign, eqpt', [
    ('GNPy_yang_formatted-edfa_example_network.json', 'GNPy_legacy_formatted-edfa_example_network_expected.json',
     'edfa_example_network_autodesign_expected.json', None),
    ('GNPy_yang_formatted-testTopology_auto_design_expected.json',
     'GNPy_legacy_formatted-testTopology_auto_design_expected.json',
     'testTopology_auto_design_expected.json',
     'eqpt_config.json')])
def test_gnpy(tmpdir, yang_input, expected_output, expected_autodesign, eqpt):
    """Convert back from yang to legacy format, and checks that GNPy can run on the converted file
    """
    with open(DATA_DIR / yang_input, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        converted = yang_to_legacy(json_data)
    with open(DATA_DIR / expected_output, 'r', encoding='utf-8') as f:
        expected = json.load(f)
    assert converted == expected
    with open(tmpdir / 'essaitopo.json', 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    args = [str(tmpdir / 'essaitopo.json'), '--save-network', str(tmpdir / 'autodesign.json')]
    if eqpt is not None:
        args.append('-e')
        args.append(str(DATA_DIR / eqpt))
    transmission_main_example(args)
    expected = load_gnpy_json(DATA_DIR / expected_autodesign)
    actual = load_gnpy_json(tmpdir / 'autodesign.json')
    assert actual == expected


@pytest.mark.parametrize('topo_input, service_input, expected_autodesign, eqpt', [
    ('testTopology_auto_design.json', 'testTopology_testservices.json',
     'testTopology_auto_design_expected.json', 'eqpt_config.json')])
def test_gnpy_path_requests_run(tmpdir, topo_input, service_input, expected_autodesign, eqpt):
    """Convert to and back from legacy- to yang to legacy format and checks that conversion runs with gnpy
    """
    with open(DATA_DIR / topo_input, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        converted = yang_to_legacy(legacy_to_yang(json_data))
    with open(tmpdir / 'essaitopo.json', 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    with open(DATA_DIR / service_input, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        converted = yang_to_legacy(legacy_to_yang(json_data))
    with open(tmpdir / 'essaiservice.json', 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    with open(DATA_DIR / eqpt, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        converted = yang_to_legacy(legacy_to_yang(json_data))
    with open(tmpdir / 'essaieqpt.json', 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    args = [str(tmpdir / 'essaitopo.json'), str(tmpdir / 'essaiservice.json'),
            '-e', str(tmpdir / 'essaieqpt.json'), '--save-network', str(tmpdir / 'autodesign.json')]
    path_requests_run(args)
    expected = load_gnpy_json(DATA_DIR / expected_autodesign)
    actual = load_gnpy_json(tmpdir / 'autodesign.json')
    assert actual == expected


def test_gnpy_eqpt():
    """Convert back from yang to legacy format and checks that conversion runs with gnpy
    """
    json_data = {
        "Edfa": [{
            "type_variety": "std_medium_gain",
            "type_def": "gnpy-eqpt-config:variable_gain",
            "gain_flatmax": "26.0",
            "gain_min": "15",
            "p_max": "21",
            "nf_min": "6",
            "nf_max": "10",
            "out_voa_auto": False,
            "allowed_for_design": True}],
        "Fiber": [{
            "type_variety": "SSMF",
            "dispersion": "0.0000167",
            "effective_area": "0.0000000000830",
            "pmd_coef": "0.000000000000001265"}],
        "Span": [{
            "power_mode": False,
            "delta_power_range_dict_db": {"min_value": "0.0", "max_value": "0.0", "step": "0.5"},
            "max_fiber_lineic_loss_for_raman": "0.25",
            "target_extended_gain": "2.5",
            "max_length": "150",
            "length_units": "km",
            "max_loss": "28",
            "padding": "10",
            "EOL": "0",
            "con_in": "0",
            "con_out": "0"}],
        "Roadm": [{
            "type_variety": "example_test",
            "target_pch_out_db": "-18.0",
            "add_drop_osnr": "35.0",
            "pmd": "0.000000000001",
            "pdl": "0.5",
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            },
            "roadm-path-impairments": []}],
        "SI": [{
            "f_min": "191300000000000.0",
            "f_max": "196100000000000.0",
            "baud_rate": "32000000000.0",
            "spacing": "50000000000.0",
            "power_dbm": "0.0",
            "power_range_dict_db": {"min_value": "0.0", "max_value": "0.0", "step": "0.5"},
            "roll_off": "0.15",
            "tx_osnr": "100.0",
            "sys_margins": "0.0"}],
        "Transceiver": [{
            "type_variety": "vendorA_trx-type1",
            "frequency": {
                "min": "191350000000000",
                "max": "196100000000000"
            },
            "mode": [{
                "format": "PS_SP64_1",
                "baud_rate": "32000000000",
                "OSNR": "11",
                "bit_rate": "100000000000.0",
                "roll_off": "0.15",
                "tx_osnr": "100",
                "min_spacing": "50000000000",
                "cost": "1"
            }, {
                "format": "PS_SP64_2",
                "baud_rate": "64000000000",
                "OSNR": "15",
                "bit_rate": "200000000000",
                "roll_off": "0.15",
                "tx_osnr": "100",
                "min_spacing": "75000000000",
                "cost": "1"
            }
            ]}]}

    expected_data = {
        "Edfa": [{
            "type_variety": "std_medium_gain",
            "type_def": "variable_gain",
            "gain_flatmax": 26.0,
            "gain_min": 15.0,
            "p_max": 21.0,
            "nf_min": 6.0,
            "nf_max": 10.0,
            "out_voa_auto": False,
            "allowed_for_design": True}],
        "Fiber": [{
            "type_variety": "SSMF",
            "dispersion": 1.67e-05,
            "effective_area": 83e-12,
            "pmd_coef": 1.265e-15}],
        "Span": [{
            "power_mode": False,
            "delta_power_range_db": [0.0, 0.0, 0.5],
            "max_fiber_lineic_loss_for_raman": 0.25,
            "target_extended_gain": 2.5,
            "max_length": 150.0,
            "length_units": "km",
            "max_loss": 28.0,
            "padding": 10.0,
            "EOL": 0.0,
            "con_in": 0.0,
            "con_out": 0.0}],
        "Roadm": [{
            "type_variety": "example_test",
            "target_pch_out_db": -18.0,
            "add_drop_osnr": 35.0,
            "pmd": 1e-12,
            "pdl": 0.5,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": []
            },
            "roadm-path-impairments": []}],
        "SI": [{
            "f_min": 191.3e12,
            "f_max": 196.1e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0.0,
            "power_range_db": [0.0, 0.0, 0.5],
            "roll_off": 0.15,
            "tx_osnr": 100.0,
            "sys_margins": 0.0}],
        "Transceiver": [{
            "type_variety": "vendorA_trx-type1",
            "frequency": {
                "min": 191.35e12,
                "max": 196.1e12
            },
            "mode": [{
                "format": "PS_SP64_1",
                "baud_rate": 32e9,
                "OSNR": 11,
                "bit_rate": 100e9,
                "roll_off": 0.15,
                "tx_osnr": 100,
                "min_spacing": 50e9,
                "cost": 1
            }, {
                "format": "PS_SP64_2",
                "baud_rate": 64e9,
                "OSNR": 15,
                "bit_rate": 200e9,
                "roll_off": 0.15,
                "tx_osnr": 100,
                "min_spacing": 75e9,
                "cost": 1
            }]
        }]
    }

    converted = yang_to_legacy(json_data)
    assert converted == expected_data


@pytest.mark.parametrize('input, expected_output', [
    (EXAMPLE_DIR / 'edfa_example_network.json', 'GNPy_yang_formatted-edfa_example_network_expected.json'),
    ('testTopology_auto_design.json', 'GNPy_yang_formatted-testTopology_auto_design_expected.json'),
    ('GNPy_yang_formatted-testTopology_auto_design_expected.json',
     'GNPy_yang_formatted-testTopology_auto_design_expected.json'),
    ('testTopology_testservices.json', 'testTopology_testservices_expected.json'),
    ('testTopology_testservices_expected.json', 'testTopology_testservices_expected.json'),
    ('eqpt_config.json', 'GNPy_yang_formatted-eqpt_config_expected.json'),
    ('extra_eqpt_config.json', 'GNPy_yang_formatted-extra_eqpt_config_expected.json'),
    ('GNPy_yang_formatted-eqpt_config_expected.json', 'GNPy_yang_formatted-eqpt_config_expected.json'),
    ('sim_params.json', 'GNPy_yang_formatted-sim_params_expected.json'),
    ('GNPy_yang_formatted-sim_params_expected.json', 'GNPy_yang_formatted-sim_params_expected.json'),
    ('spectrum.json', 'GNPy_yang_formatted-spectrum_expected.json'),
    ('GNPy_yang_formatted-spectrum_expected.json', 'GNPy_yang_formatted-spectrum_expected.json'),
    ('testTopology_response.json', 'GNPy_yang_formatted-testTopology_response.json'),
    ('std_medium_gain_advanced_config.json', 'GNPy_yang_formatted-edfa-config_expected.json')])
def test_gnpy_convert(input, expected_output):
    """Convert legacy gnpy format to yang format and checks that conversion is as expected
    """
    with open(DATA_DIR / input, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        actual = legacy_to_yang(json_data)
    with open(DATA_DIR / expected_output, 'r', encoding='utf-8') as f:
        expected = json.load(f)
    assert actual == expected


@pytest.mark.parametrize("args, fileout", (
    [['--legacy-to-yang', EXAMPLE_DIR / 'edfa_example_network.json', '-o'], 'essaitopo.json'],
    [['--legacy-to-yang', DATA_DIR / 'testTopology_auto_design.json', '-o'], 'essaitopo2.json'],
    [['--legacy-to-yang', DATA_DIR / 'eqpt_config.json', '-o'], 'essaieqpt.json'],
    [['--legacy-to-yang', DATA_DIR / 'testTopology_testservices.json', '-o'], 'essaireq.json']))
def test_example_invocation(tmpdir, args, fileout):
    """test the main function of converter"""
    os.environ['PYTHONPATH'] = str(SRC_ROOT)
    proc = subprocess.run(
        ('python', SRC_ROOT / 'gnpy' / 'tools' / 'convert_legacy_yang.py', *args, tmpdir / fileout),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, universal_newlines=True)     # nosec
    assert proc.stderr == ''


def test_span_with_delta_power_range_dict_db():
    """Test when span already has delta_power_range_dict_db"""
    input_data = {
        'Span': [{
            "power_mode": False,
            "max_fiber_lineic_loss_for_raman": 0.25,
            "target_extended_gain": 2.5,
            "max_length": 150,
            "length_units": "km",
            "max_loss": 28,
            "padding": 10,
            "EOL": 0,
            "con_in": 0,
            "con_out": 0,
            'delta_power_range_dict_db': {
                'min_value': -5,
                'max_value': 5,
                'step': 0.1
            }
        }]
    }
    result = convert_delta_power_range(input_data)
    assert result == input_data


def test_span_missing_both_power_ranges():
    """Test when span is missing both power range formats"""
    input_data = {
        'Span': [{
            "power_mode": False,
            "max_fiber_lineic_loss_for_raman": 0.25,
            "target_extended_gain": 2.5,
            "max_length": 150,
            "length_units": "km",
            "max_loss": 28,
            "padding": 10,
            "EOL": 0,
            "con_in": 0,
            "con_out": 0
        }]
    }
    with pytest.raises(KeyError) as exc:
        convert_delta_power_range(input_data)
    assert 'delta_power_range or delta_power_range_dict_db missing' in str(exc.value)


def test_si_with_power_range_dict_db():
    """Test when SI already has power_range_dict_db"""
    input_data = {
        'SI': [{
            "f_min": 191.3e12,
            "f_max": 196.1e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            "roll_off": 0.15,
            "tx_osnr": 100,
            "sys_margins": 0,
            'power_range_dict_db': {
                'min_value': -10,
                'max_value': 10,
                'step': 0.5
            }
        }]
    }
    result = convert_delta_power_range(input_data)
    assert result == input_data


def test_si_missing_both_power_ranges():
    """Test when SI is missing both power range formats"""
    input_data = {
        'SI': [{
            "f_min": 191.3e12,
            "f_max": 196.1e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            "roll_off": 0.15,
            "tx_osnr": 100,
            "sys_margins": 0
        }]
    }
    with pytest.raises(KeyError) as exc:
        convert_delta_power_range(input_data)
    assert 'power_range_db or power_range_dict_db missing' in str(exc.value)


def test_successful_conversion():
    """Test successful conversion of both Span and SI"""
    input_data = {
        'Span': [{
            "power_mode": False,
            'delta_power_range_db': [-5, 5, 0.1],
            "max_fiber_lineic_loss_for_raman": 0.25,
            "target_extended_gain": 2.5,
            "max_length": 150,
            "length_units": "km",
            "max_loss": 28,
            "padding": 10,
            "EOL": 0,
            "con_in": 0,
            "con_out": 0
        }],
        'SI': [{
            "f_min": 191.3e12,
            "f_max": 196.1e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            'power_range_db': [-10, 10, 0.5],
            "roll_off": 0.15,
            "tx_osnr": 100,
            "sys_margins": 0
        }, {
            "type_variety": "LBAND",
            "f_min": 186.0e12,
            "f_max": 190.0e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            'power_range_db': [0, 0, 0.5],
            "roll_off": 0.15,
            "tx_osnr": 100,
            "sys_margins": 0
        }]
    }

    expected_output = {
        'Span': [{
            "power_mode": False,
            "max_fiber_lineic_loss_for_raman": 0.25,
            "target_extended_gain": 2.5,
            "max_length": 150,
            "length_units": "km",
            "max_loss": 28,
            "padding": 10,
            "EOL": 0,
            "con_in": 0,
            "con_out": 0,
            'delta_power_range_dict_db': {
                'min_value': -5,
                'max_value': 5,
                'step': 0.1
            }
        }],
        'SI': [{
            "f_min": 191.3e12,
            "f_max": 196.1e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            'power_range_dict_db': {
                'min_value': -10,
                'max_value': 10,
                'step': 0.5
            },
            "roll_off": 0.15,
            "tx_osnr": 100,
            "sys_margins": 0
        }, {
            "type_variety": "LBAND",
            "f_min": 186.0e12,
            "f_max": 190.0e12,
            "baud_rate": 32e9,
            "spacing": 50e9,
            "power_dbm": 0,
            'power_range_dict_db': {
                'min_value': 0,
                'max_value': 0,
                'step': 0.5
            },
            "roll_off": 0.15,
            "tx_osnr": 100,
            "sys_margins": 0
        }]
    }

    result = convert_delta_power_range(input_data)
    assert result == expected_output

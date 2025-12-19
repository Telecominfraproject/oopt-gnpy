#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_gain_mode
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
checks behaviour of gain mode
- if all amps have their gains set, check that these gains are used, even if power_dbm or req_power change
- check that saturation is correct in gain mode

"""

from pathlib import Path

from gnpy.tools.json_io import load_json, load_eqpt_topo_from_json, load_gnpy_json, _equipment_from_json
import networkx as nx


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILENAME = DATA_DIR / 'LinkforTest.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": load_json(DATA_DIR / "std_medium_gain_advanced_config.json")}


def test_load_eqpt_topo_from_json():
    eqpt = load_json(EQPT_FILENAME)
    topology = load_json(NETWORK_FILENAME)
    extra_equipments = None
    extra_configs = EXTRA_CONFIGS

    equipment, network = load_eqpt_topo_from_json(eqpt, topology, extra_equipments, extra_configs)

    assert isinstance(equipment, dict)
    assert isinstance(network, nx.DiGraph)

    assert len(network.nodes) > 0
    assert len(network.edges) > 0


def test_load_eqpt_topo_from_json_default_extra():
    eqpt = load_json(EQPT_FILENAME)
    topology = load_json(NETWORK_FILENAME)

    equipment, network = load_eqpt_topo_from_json(eqpt, topology)

    assert isinstance(equipment, dict)
    assert isinstance(network, nx.DiGraph)
    # Check that some expected nodes exist (example: 'A', 'B')
    assert len(network.nodes) > 0
    assert len(network.edges) > 0


def test_equipment_other_name():

    equipment_json = load_gnpy_json(EQPT_FILENAME)
    amp = {
        "type_variety": "std_high_gain",
        "other_name": ["coucouamp"],
        "type_def": "variable_gain",
        "gain_flatmax": 35,
        "gain_min": 25,
        "p_max": 21,
        "nf_min": 5.5,
        "nf_max": 7,
        "out_voa_auto": False,
        "allowed_for_design": True}
    transceiver = {
        "type_variety": "vendorA_trx-type2",
        "other_name": ["coucou", "123"],
        "frequency": {
            "min": 191.35e12,
            "max": 196.1e12},
        "mode": [{
            "format": "mode 1",
            "other_name": ["abc", "123"],
            "baud_rate": 32e9,
            "OSNR": 11,
            "bit_rate": 100e9,
            "roll_off": 0.15,
            "tx_osnr": 40,
            "min_spacing": 37.5e9,
            "penalties": [
                {
                    "chromatic_dispersion": 1440.0,
                    "penalty_value": 0.0
                },
                {
                    "chromatic_dispersion": 1600.0,
                    "penalty_value": 1.0
                }
            ],
            "cost": 1
        }, {
            "format": "mode 2",
            "baud_rate": 66e9,
            "OSNR": 15,
            "bit_rate": 200e9,
            "roll_off": 0.15,
            "tx_osnr": 40,
            "min_spacing": 75e9,
            "cost": 1}]}
    equipment_json['Transceiver'].append(transceiver)
    equipment_json['Edfa'].append(amp)
    equipment = _equipment_from_json(equipment_json, EXTRA_CONFIGS)
    keys = ['type_def', 'gain_flatmax', 'gain_min', 'p_max', 'out_voa_auto', 'allowed_for_design',
            'f_min', 'f_max', 'bands', 'dgt', 'nf_fit_coeff', 'nf_ripple']
    for key in keys:
        assert getattr(equipment['Edfa']['std_high_gain'], key) == getattr(equipment['Edfa']['coucouamp'], key)
    # other_name should not be a key of amp
    for amp in equipment['Edfa']:
        assert 'other_name' not in amp
    # check that copies are correctly created for coucou other_name
    assert getattr(equipment['Transceiver']['vendorA_trx-type2'], 'frequency') == \
        getattr(equipment['Transceiver']['coucou'], 'frequency')
    assert getattr(equipment['Transceiver']['vendorA_trx-type2'], 'frequency') == \
        getattr(equipment['Transceiver']['123'], 'frequency')
    # check that modes are correctly create and eventually duplicated if they have other_name
    modes = {m['format']: m for m in equipment['Transceiver']['vendorA_trx-type2'].mode}
    other_modes = {m['format']: m for m in equipment['Transceiver']['coucou'].mode}
    # modes should be identical except for format and other_name
    keys = ['baud_rate', 'OSNR', 'bit_rate', 'roll_off', 'tx_osnr', 'min_spacing', 'cost', 'penalties']
    for key in keys:
        assert modes['mode 2'][key] == other_modes['mode 2'][key]
        assert modes['abc'][key] == modes['mode 1'][key]
        assert modes['abc'][key] == other_modes['123'][key]
    # other_name should not be a key of mode
    for mode in dict(modes, **other_modes).values():
        assert 'other_name' not in mode
    # other_name should not be a key of trx
    for trx_type in equipment['Transceiver']:
        assert 'other_name' not in trx_type

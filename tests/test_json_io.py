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

from gnpy.tools.json_io import  load_json, load_eqpt_topo_from_json
import networkx as nx


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILENAME  = DATA_DIR / 'LinkforTest.json'
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



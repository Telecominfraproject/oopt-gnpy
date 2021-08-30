# SPDX-License-Identifier: BSD-3-Clause
#
# Reading and writing JSON files for GNPy
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#
""" Tests for regeneration feature
    - parsing regen in the network topology
    - parsing regen in a route list w and wo hop attribute
    - correct interpretation of the regen according to service
    - propagation result with regen identical to the last section
    - if regeneration is introduced as a preference provides a path
      with regeneration if path is not feasible otherwise
"""
from pathlib import Path
from os import unlink
import pytest
from numpy import mean
from gnpy.core.network import build_network
from gnpy.core.elements import Regenerator, Transceiver
from gnpy.core.equipment import trx_mode_params
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.core.exceptions import ServiceError
from gnpy.tools.json_io import load_equipment, load_network, requests_from_json, save_network, load_json
from gnpy.topology.spectrum_assignment import build_oms_list

from tests.compare import compare_networks


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILENAME = DATA_DIR / 'threehops_regen.json'
SERVICE_FILENAME = DATA_DIR / 'regen_services.json'


@pytest.fixture()
def equipment():
    """common setup for tests: builds equipment only once
    """
    equipment = load_equipment(EQPT_FILENAME)
    # reduce the number of channels to speed up tests (propagation is taking ages for the full spectrum)
    for elem in equipment['Transceiver']:
        equipment['Transceiver'][elem].frequency['max'] = 191.9e12
    return equipment


@pytest.fixture()
def setup(equipment):
    """ common setup for tests: builds network and oms only once
    """
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    build_oms_list(network, equipment)
    return network

# parsing tests: network reading and creation
def test_network_parsing(tmpdir, setup):
    """test that autodesign creates same file as the input file (already autodesigned)
    and correctly reproduces regenerators in the autodesign
    """
    network = setup
    actual_json_output = tmpdir / 'threehops_regen_auto_design.json'
    print(actual_json_output)
    save_network(network, actual_json_output)
    actual = load_json(actual_json_output)
    # unlink(actual_json_output)
    expected = load_json(NETWORK_FILENAME)

    results = compare_networks(expected, actual)
    assert not results.elements.missing
    assert not results.elements.extra
    assert not results.elements.different
    assert not results.connections.missing
    assert not results.connections.extra
    assert not results.connections.different

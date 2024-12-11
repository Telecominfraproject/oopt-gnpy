#!/usr/bin/env python3
# Module name: test_path_computation_functions.py
# Version:
# License: BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic

"""

from pathlib import Path
import pytest
from gnpy.core.network import build_network
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.topology.request import explicit_path
from gnpy.topology.spectrum_assignment import build_oms_list
from gnpy.tools.json_io import load_equipment, load_network, requests_from_json


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILENAME = DATA_DIR / 'testTopology_auto_design_expected.json'
SERVICE_FILENAME = DATA_DIR / 'testTopology_services_expected.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}
equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)


@pytest.fixture()
def setup_without_oms():
    """ common setup for tests: builds network, equipment and oms only once
    """
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    return network


def some_request(explicit_route):
    """Create a request with an explicit route
    """
    route = {
        "route-object-include-exclude": [
            {
                "explicit-route-usage": "route-include-ero",
                "index": i,
                "num-unnum-hop": {
                    "node-id": node_id,
                    "link-tp-id": "link-tp-id is not used",
                    "hop-type": "STRICT"
                }
            } for i, node_id in enumerate(explicit_route)
        ]
    }
    return {
        "path-request": [{
            "request-id": "2",
            "source": explicit_route[0],
            "destination": explicit_route[-1],
            "src-tp-id": explicit_route[0],
            "dst-tp-id": explicit_route[-1],
            "bidirectional": False,
            "path-constraints": {
                "te-bandwidth": {
                    "technology": "flexi-grid",
                    "trx_type": "Voyager",
                    "trx_mode": "mode 1",
                    "spacing": 75000000000.0,
                    "path_bandwidth": 100000000000.0
                }
            },
            "explicit-route-objects": route}
        ]
    }


@pytest.mark.parametrize('setup', ["with_oms", "without_oms"])
@pytest.mark.parametrize('explicit_route, expected_path', [
    (['trx Brest_KLA', 'trx Vannes_KBE'], None),
    # path contains one element per oms
    (['trx Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'trx Lannion_CAS'],
     ['trx Brest_KLA', 'roadm Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'fiber (Brest_KLA → Morlaix)-F060',
      'east fused spans in Morlaix', 'fiber (Morlaix → Lannion_CAS)-F059', 'west edfa in Lannion_CAS to Morlaix',
      'roadm Lannion_CAS', 'trx Lannion_CAS']),
    # path contains several elements per oms
    (['trx Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'west edfa in Lannion_CAS to Morlaix',
      'roadm Lannion_CAS', 'trx Lannion_CAS'],
     ['trx Brest_KLA', 'roadm Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'fiber (Brest_KLA → Morlaix)-F060',
      'east fused spans in Morlaix', 'fiber (Morlaix → Lannion_CAS)-F059', 'west edfa in Lannion_CAS to Morlaix',
      'roadm Lannion_CAS', 'trx Lannion_CAS']),
    # path contains all elements
    (['trx Brest_KLA', 'roadm Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'fiber (Brest_KLA → Morlaix)-F060',
      'east fused spans in Morlaix', 'fiber (Morlaix → Lannion_CAS)-F059', 'west edfa in Lannion_CAS to Morlaix',
      'roadm Lannion_CAS', 'trx Lannion_CAS'],
     ['trx Brest_KLA', 'roadm Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'fiber (Brest_KLA → Morlaix)-F060',
      'east fused spans in Morlaix', 'fiber (Morlaix → Lannion_CAS)-F059', 'west edfa in Lannion_CAS to Morlaix',
      'roadm Lannion_CAS', 'trx Lannion_CAS']),
    # path conteains element for only 1 oms (2 oms path)
    (['trx Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'trx Rennes_STA'], None),
    # path contains roadm edges for all OMS, but no element of the OMS
    (['trx Brest_KLA', 'roadm Brest_KLA', 'roadm Lannion_CAS', 'trx Lannion_CAS'], None),
    # path contains one element for all 3 OMS
    (['trx Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'east edfa in Lannion_CAS to Corlay',
      'east edfa in Lorient_KMA to Vannes_KBE', 'trx Vannes_KBE'],
     ['trx Brest_KLA', 'roadm Brest_KLA', 'east edfa in Brest_KLA to Morlaix', 'fiber (Brest_KLA → Morlaix)-F060',
      'east fused spans in Morlaix', 'fiber (Morlaix → Lannion_CAS)-F059', 'west edfa in Lannion_CAS to Morlaix',
      'roadm Lannion_CAS', 'east edfa in Lannion_CAS to Corlay', 'fiber (Lannion_CAS → Corlay)-F061',
      'west fused spans in Corlay', 'fiber (Corlay → Loudeac)-F010', 'west fused spans in Loudeac',
      'fiber (Loudeac → Lorient_KMA)-F054', 'west edfa in Lorient_KMA to Loudeac', 'roadm Lorient_KMA',
      'east edfa in Lorient_KMA to Vannes_KBE', 'fiber (Lorient_KMA → Vannes_KBE)-F055',
      'west edfa in Vannes_KBE to Lorient_KMA', 'roadm Vannes_KBE', 'trx Vannes_KBE'])])
def test_explicit_path(setup, setup_without_oms, explicit_route, expected_path):
    """tests that explicit path correctly returns the full path if it is possible else that it returns None
    """
    network = setup_without_oms
    if setup == "with_oms":
        # OMS are initiated in elements, so that explicit path can be verified
        build_oms_list(network, equipment)
    else:
        # OMS are not initiated, explicit path can not be computed.
        expected_path = None
    json_data = some_request(explicit_route)
    [req] = requests_from_json(json_data, equipment)

    node_list = []
    for node in req.nodes_list:
        node_list.append(next(el for el in network if el.uid == node))
    source = node_list[0]
    destination = node_list[-1]
    if expected_path is None:
        assert explicit_path(node_list, source, destination, network) is None
    else:
        actual_path = [e.uid for e in explicit_path(node_list, source, destination, network)]
        assert actual_path == expected_path

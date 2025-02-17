from pathlib import Path
import pytest

from gnpy.tools.json_io import results_to_json, load_requests, load_json, load_equipment, load_network
from tests.test_disjunction import serv, test_setup
from gnpy.tools.worker_utils import planning

DATA_DIR = Path(__file__).parent.parent / 'tests/data'
NETWORK_FILE_NAME = DATA_DIR / 'testTopology_expected.json'
SERVICE_FILE_NAME = DATA_DIR / 'testTopology_testservices.json'
RESULT_FILE_NAME = DATA_DIR / 'testTopology_testresults.json'
EQPT_LIBRARY_NAME = DATA_DIR / 'eqpt_config.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}



def test_results_to_json(serv):
    equipment =  load_equipment(EQPT_LIBRARY_NAME)
    network = load_network(NETWORK_FILE_NAME, equipment)

    service = load_json(SERVICE_FILE_NAME)

    _, _, _, _, _, result = planning(network, equipment, service)

    result_json = results_to_json(result)
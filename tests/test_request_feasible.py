from pathlib import Path
import pytest

from gnpy.core.network import build_network
from gnpy.core.utils import automatic_nch, lin2db
from gnpy.topology.request import deduplicate_disjunctions, requests_aggregation
from gnpy.topology.request_feasible import find_feasible_paths
from gnpy.topology.spectrum_assignment import build_oms_list
from gnpy.tools.json_io import disjunctions_from_json, load_equipment, load_network, requests_from_json 
from gnpy.tools.service_sheet import read_service_sheet


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'


@pytest.mark.parametrize('xls_input', (DATA_DIR / 'topology_all_feasible_paths.xls', ))
def test_compute_route_constraint(xls_input):
    """."""
    equipment = load_equipment(EQPT_FILENAME)
    network = load_network(xls_input, equipment)
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max,
                                             equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    service_sheet = read_service_sheet(xls_input, equipment, network)
    oms_list = build_oms_list(network, equipment)
    service_list = requests_from_json(service_sheet, equipment)
    disjunction_list = deduplicate_disjunctions(disjunctions_from_json(service_sheet))
    service_list, disjunction_list = requests_aggregation(service_list, disjunction_list)
    paths = find_feasible_paths(disjunction_list, equipment, network, service_list)
    assert len(paths['0']['mode 1']) == 11
    assert len(paths['1']['mode 1']) == 11
    assert len(paths['2']['mode 1']) == 15
    assert len(paths['3']['mode 1']) == 7
    assert len(paths['4']['mode 1|mode 1']) == 11
    assert len(paths['5']['mode 1|mode 1']) == 23
    assert len(paths['6']['mode 1|mode 1']) == 0
    assert len(paths['7']['mode 1|mode 1']) == 0
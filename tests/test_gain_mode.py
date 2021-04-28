#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2019-05-22
"""
@author: esther.lerouzic
checks behaviour of gain mode
- if all amps have their gains set, check that these gains are used, even if power_dbm or req_power change
- check that saturation is correct in gain mode

"""

from pathlib import Path
from copy import deepcopy
import pytest
from gnpy.core.utils import lin2db, automatic_nch, db2lin, db2lin
from gnpy.core.parameters import FiberParams
from gnpy.core.network import build_network
from gnpy.tools.json_io import network_from_json, load_equipment, load_json, load_network
from gnpy.core.equipment import trx_mode_params
from gnpy.topology.request import PathRequest, compute_constrained_path, propagate


TEST_DIR = Path(__file__).parent
EQPT_FILENAME = TEST_DIR / 'data/eqpt_config.json'
NETWORK_FILENAME = TEST_DIR / 'data/perdegreemeshTopologyExampleV2_auto_design_expected.json'


def net_setup(equipment):
    """ common setup for tests: builds network, equipment
    """
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    return network


def create_rq(equipment, srce, dest, bdir, nd_list, ls_list, mode, power_dbm):
    """ create the usual request list according to parameters
    """
    params = {}
    params['request_id'] = 'test_request'
    params['source'] = srce
    params['bidir'] = bdir
    params['destination'] = dest
    params['trx_type'] = 'Voyager'
    params['trx_mode'] = mode
    params['format'] = params['trx_mode']
    params['spacing'] = 50e9 if mode == 'mode 1' else 75e9
    params['nodes_list'] = nd_list
    params['loose_list'] = ls_list
    trx_params = trx_mode_params(equipment, params['trx_type'], params['trx_mode'], True)
    params.update(trx_params)
    params['power'] = dbm2watt(power_dbm) if power_dbm else db2lin(equipment['SI']['default'].power_dbm) * 1e-3
    f_min = params['f_min']
    f_max_from_si = params['f_max']
    params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
    params['path_bandwidth'] = 100000000000.0
    return PathRequest(**params)

@pytest.mark.parametrize("power_dbm", [0, -2, 3])
@pytest.mark.parametrize("req_power", [1e-3, 0.5e-3, 2e-3])
def test_no_amp_feature(req_power, power_dbm):
    """
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req = create_rq(equipment, 'trx Brest_KLA', 'trx Rennes_STA', False,
                    ['Edfa0_roadm Brest_KLA', 'roadm Lannion_CAS', 'trx Rennes_STA'], 
                    ['STRICT', 'STRICT', 'STRICT'],
                    'mode 1', 0)
    path = compute_constrained_path(network, req)
    infos_expected = propagate(path, req, equipment)

    setattr(equipment['Span']['default'], 'power_mode', False)
    setattr(equipment['SI']['default'], 'power_dbm', power_dbm)
    req.power = req_power
    network2 = net_setup(equipment)
    path2 = compute_constrained_path(network2, req)
    infos_actual = propagate(path2, req, equipment)
    for expected, actual in zip(infos_expected.carriers, infos_actual.carriers):
        assert expected.power.signal == actual.power.signal
        assert expected.power.ase == actual.power.ase
        assert expected.power.nli == actual.power.nli

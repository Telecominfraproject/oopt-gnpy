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
import pytest
from gnpy.core.utils import lin2db, automatic_nch, db2lin
from gnpy.core.network import build_network
from gnpy.tools.json_io import load_equipment, load_network
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
    p_total_db = p_db + nb_channels_db(equipment)
    build_network(network, equipment, p_db, p_total_db)
    return network


def nb_channels_db(equipment):
    """ computes the nb of channels in dB using SI
    """
    spectrum = equipment['SI']['default']
    return lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))


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
    params['power'] = db2lin(power_dbm) * 1e-3
    f_min = params['f_min']
    f_max_from_si = params['f_max']
    params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
    params['path_bandwidth'] = 100000000000.0
    return PathRequest(**params)


@pytest.mark.parametrize("power_dbm", [0, -2, 3])
@pytest.mark.parametrize("req_power", [1e-3, 0.5e-3, 2e-3])
def test_gain_mode(req_power, power_dbm):
    """ tests that power settings have no effect in gain mode. propagates for a setting, and thene checks that
    other settings give the same result.
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req = create_rq(equipment, 'trx Brest_KLA', 'trx Rennes_STA', False,
                    ['Edfa0_roadm Brest_KLA', 'roadm Lannion_CAS', 'trx Rennes_STA'],
                    ['STRICT', 'STRICT', 'STRICT'], 'mode 1', 0)
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


def test_gain_mode_saturation():
    """ In stdard network, and gain_mode saturation can not occur with poser settings
    because target power out of amp fixes the power for the OMS, and gain are used,
    create an artificial case where saturation can occur: eg  a higher target for a given
    roadm. then booster must saturate, and resulting gain is changed.  This saturation only
    occurs because of the nb of channels
    """
    equipment = load_equipment(EQPT_FILENAME)
    equipment['Span']['default'].power_mode = False
    network = net_setup(equipment)
    req = create_rq(equipment, 'trx Brest_KLA', 'trx Rennes_STA', False,
                    ['Edfa0_roadm Brest_KLA', 'roadm Lannion_CAS', 'trx Rennes_STA'],
                    ['STRICT', 'STRICT', 'STRICT'], 'mode 1', 0)
    path = compute_constrained_path(network, req)
    _ = propagate(path, req, equipment)
    assert path[2].effective_gain == 20

    # change roadm target power on the degree
    roadm = next(n for n in network.nodes() if n.uid == 'roadm Brest_KLA')
    roadm.per_degree_pch_out_db['Edfa0_roadm Brest_KLA'] = -12
    path2 = compute_constrained_path(network, req)
    _ = propagate(path2, req, equipment)

    # expected output power = lin2db(nb_channels) + Gain + input power
    # case no saturation:
    #   target power = -20dBm, gain set from network file: 20dB, total_power ~ 19.83 dBm < 21 (amp of booster)
    # case saturation:
    #   target power = -12, gain set = 20, total power ~ 19.83 + 8 > 21. saturated to 21
    #      hence gain is set to 21 - (19.83 -12) ~ 13.18
    #

    assert path2[2].effective_gain == 21 - (-12 + lin2db(req.nb_channel))

    # by the way check that req.nb_channel if not the same as nb_channels_db(equipment),
    # because f_min of Voyager is different!
    assert nb_channels_db(equipment) == lin2db(req.nb_channel + 1)

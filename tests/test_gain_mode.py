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
from numpy.testing import assert_array_equal, assert_allclose

import pytest
from gnpy.core.utils import lin2db, automatic_nch, dbm2watt
from gnpy.core.network import build_network
from gnpy.tools.json_io import network_from_json, load_equipment, load_network
from gnpy.core.equipment import trx_mode_params
from gnpy.topology.request import PathRequest, compute_constrained_path, propagate


TEST_DIR = Path(__file__).parent
EQPT_FILENAME = TEST_DIR / 'data/eqpt_config.json'
NETWORK_FILENAME = TEST_DIR / 'data/perdegreemeshTopologyExampleV2_auto_design_expected.json'


def net_setup(equipment):
    """Common setup for tests: builds network, equipment
    """
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    return network


def create_rq(equipment, srce, dest, bdir, nd_list, ls_list, mode, power_dbm):
    """Create the usual request list according to parameters
    """
    params = {
        'request_id': 'test_request',
        'source': srce,
        'bidir': bdir,
        'destination': dest,
        'trx_type': 'Voyager',
        'trx_mode': mode,
        'format': mode,
        'nodes_list': nd_list,
        'loose_list': ls_list,
        'effective_freq_slot': None,
        'path_bandwidth': 100000000000.0,
        'spacing': 50e9 if mode == 'mode 1' else 75e9,
        'power': dbm2watt(power_dbm)
    }
    trx_params = trx_mode_params(equipment, params['trx_type'], params['trx_mode'], True)
    params.update(trx_params)
    f_min = params['f_min']
    f_max_from_si = params['f_max']
    params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
    return PathRequest(**params)


@pytest.mark.parametrize("power_dbm", [0, -2, 3])
@pytest.mark.parametrize("req_power", [1e-3, 0.5e-3, 2e-3])
def test_gain_mode(req_power, power_dbm):
    """ Gains are all set on the selected path, so that since the design is made for 0dBm,
    in gain mode, whatever the value of equipment power_dbm or request power, the network is unchanged
    and the propagation remains the same as for power mode and 0dBm
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = net_setup(equipment)
    req = create_rq(equipment, 'trx Brest_KLA', 'trx Rennes_STA', False,
                    ['Edfa0_roadm Brest_KLA', 'roadm Lannion_CAS', 'trx Rennes_STA'],
                    ['STRICT', 'STRICT', 'STRICT'], 'mode 1', 0)
    path = compute_constrained_path(network, req)
    # Propagation in power_mode
    infos_expected = propagate(path, req, equipment)
    # Now set to gain mode
    setattr(equipment['Span']['default'], 'power_mode', False)
    setattr(equipment['SI']['default'], 'power_dbm', power_dbm)
    req.power = req_power
    network2 = net_setup(equipment)
    path2 = compute_constrained_path(network2, req)
    infos_actual = propagate(path2, req, equipment)

    assert_array_equal(infos_expected.baud_rate, infos_actual.baud_rate)
    assert_allclose(infos_expected.signal, infos_actual.signal, rtol=1e-14)
    assert_allclose(infos_expected.nli, infos_actual.nli, rtol=1e-14)
    assert_allclose(infos_expected.ase, infos_actual.ase, rtol=1e-14)
    assert_array_equal(infos_expected.roll_off, infos_actual.roll_off)
    assert_array_equal(infos_expected.chromatic_dispersion, infos_actual.chromatic_dispersion)
    assert_array_equal(infos_expected.pmd, infos_actual.pmd)
    assert_array_equal(infos_expected.channel_number, infos_actual.channel_number)
    assert_array_equal(infos_expected.number_of_channels, infos_actual.number_of_channels)

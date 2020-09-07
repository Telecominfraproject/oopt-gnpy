#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2019-05-22
"""
@author: esther.lerouzic
checks that fused placed in amp type is correctly converted to a fused element instead of an edfa
and that no additional amp is added.
checks that restrictions in roadms are correctly applied during autodesign

"""

from pathlib import Path
import pytest
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.core.elements import Fused, Roadm, Edfa
from gnpy.core.network import build_network
from gnpy.tools.json_io import network_from_json, load_equipment, load_json, Amp
from gnpy.core.equipment import trx_mode_params
from gnpy.topology.request import PathRequest, compute_constrained_path
from gnpy.core.info import create_input_spectral_information
from gnpy.core.utils import db2lin

TEST_DIR = Path(__file__).parent
EQPT_LIBRARY_NAME = TEST_DIR / 'data/eqpt_config.json'
NETWORK_FILE_NAME = TEST_DIR / 'data/testTopology_expected.json'
# adding tests to check the roadm restrictions

# mark node_uid amps as fused for testing purpose
@pytest.mark.parametrize("node_uid", ['east edfa in Lannion_CAS to Stbrieuc'])
def test_no_amp_feature(node_uid):
    ''' Check that booster is not placed on a roadm if fused is specified
        test_parser covers partly this behaviour. This test should guaranty that the
        feature is preserved even if convert is changed
    '''
    equipment = load_equipment(EQPT_LIBRARY_NAME)
    json_network = load_json(NETWORK_FILE_NAME)

    for elem in json_network['elements']:
        if elem['uid'] == node_uid:
            # replace edfa node by a fused node in the topology
            elem['type'] = 'Fused'
            elem.pop('type_variety')
            elem.pop('operational')
            elem['params'] = {'loss': 0}

            next_node_uid = next(conn['to_node'] for conn in json_network['connections']
                                 if conn['from_node'] == node_uid)
            previous_node_uid = next(conn['from_node'] for conn in json_network['connections']
                                     if conn['to_node'] == node_uid)

    network = network_from_json(json_network, equipment)
    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))

    build_network(network, equipment, p_db, p_total_db)

    node = next(nd for nd in network.nodes() if nd.uid == node_uid)
    next_node = next(network.successors(node))
    previous_node = next(network.predecessors(node))

    if not isinstance(node, Fused):
        raise AssertionError()
    if not node.params.loss == 0.0:
        raise AssertionError()
    if not next_node_uid == next_node.uid:
        raise AssertionError()
    if not previous_node_uid == previous_node.uid:
        raise AssertionError()


@pytest.fixture()
def equipment():
    """init transceiver class to access snr and osnr calculations"""
    equipment = load_equipment(EQPT_LIBRARY_NAME)
    # define some booster and preamps
    restrictions_list = [
        {
            'type_variety': 'booster_medium_gain',
            'type_def': 'variable_gain',
            'gain_flatmax': 25,
            'gain_min': 15,
            'p_max': 21,
            'nf_min': 5.8,
            'nf_max': 10,
            'out_voa_auto': False,
            'allowed_for_design': False
        },
        {
            'type_variety': 'preamp_medium_gain',
            'type_def': 'variable_gain',
            'gain_flatmax': 26,
            'gain_min': 15,
            'p_max': 23,
            'nf_min': 6,
            'nf_max': 10,
            'out_voa_auto': False,
            'allowed_for_design': False
        },
        {
            'type_variety': 'preamp_high_gain',
            'type_def': 'variable_gain',
            'gain_flatmax': 35,
            'gain_min': 25,
            'p_max': 21,
            'nf_min': 5.5,
            'nf_max': 7,
            'out_voa_auto': False,
            'allowed_for_design': False
        },
        {
            'type_variety': 'preamp_low_gain',
            'type_def': 'variable_gain',
            'gain_flatmax': 16,
            'gain_min': 8,
            'p_max': 23,
            'nf_min': 6.5,
            'nf_max': 11,
            'out_voa_auto': False,
            'allowed_for_design': False
        }]
    # add them to the library
    for entry in restrictions_list:
        equipment['Edfa'][entry['type_variety']] = Amp.from_json(EQPT_LIBRARY_NAME, **entry)
    return equipment


@pytest.mark.parametrize("restrictions", [
    {
        'preamp_variety_list': [],
        'booster_variety_list':[]
    },
    {
        'preamp_variety_list': [],
        'booster_variety_list':['booster_medium_gain']
    },
    {
        'preamp_variety_list': ['preamp_medium_gain', 'preamp_high_gain', 'preamp_low_gain'],
        'booster_variety_list':[]
    }])
def test_restrictions(restrictions, equipment):
    ''' test that restriction is correctly applied if provided in eqpt_config and if no Edfa type
    were provided in the network json
    '''
    # add restrictions
    equipment['Roadm']['default'].restrictions = restrictions
    # build network
    json_network = load_json(NETWORK_FILE_NAME)
    network = network_from_json(json_network, equipment)

    amp_nodes_nobuild_uid = [nd.uid for nd in network.nodes()
                             if isinstance(nd, Edfa) and isinstance(next(network.predecessors(nd)), Roadm)]
    preamp_nodes_nobuild_uid = [nd.uid for nd in network.nodes()
                                if isinstance(nd, Edfa) and isinstance(next(network.successors(nd)), Roadm)]
    amp_nodes_nobuild = {nd.uid: nd for nd in network.nodes()
                         if isinstance(nd, Edfa) and isinstance(next(network.predecessors(nd)), Roadm)}
    preamp_nodes_nobuild = {nd.uid: nd for nd in network.nodes()
                            if isinstance(nd, Edfa) and isinstance(next(network.successors(nd)), Roadm)}
    # roadm dict with restrictions before build
    roadms = {nd.uid: nd for nd in network.nodes() if isinstance(nd, Roadm)}
    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))

    build_network(network, equipment, p_db, p_total_db)

    amp_nodes = [nd for nd in network.nodes()
                 if isinstance(nd, Edfa) and isinstance(next(network.predecessors(nd)), Roadm)
                 and next(network.predecessors(nd)).restrictions['booster_variety_list']]

    preamp_nodes = [nd for nd in network.nodes()
                    if isinstance(nd, Edfa) and isinstance(next(network.successors(nd)), Roadm)
                    and next(network.successors(nd)).restrictions['preamp_variety_list']]

    # check that previously existing amp are not changed
    for amp in amp_nodes:
        if amp.uid in amp_nodes_nobuild_uid:
            print(amp.uid, amp.params.type_variety)
            if not amp.params.type_variety == amp_nodes_nobuild[amp.uid].params.type_variety:
                raise AssertionError()
    for amp in preamp_nodes:
        if amp.uid in preamp_nodes_nobuild_uid:
            if not amp.params.type_variety == preamp_nodes_nobuild[amp.uid].params.type_variety:
                raise AssertionError()
    # check that restrictions are correctly applied
    for amp in amp_nodes:
        if amp.uid not in amp_nodes_nobuild_uid:
            # and if roadm had no restrictions before build:
            if restrictions['booster_variety_list'] and \
               not roadms[next(network.predecessors(amp)).uid]\
               .restrictions['booster_variety_list']:
                if amp.params.type_variety not in restrictions['booster_variety_list']:

                    raise AssertionError()
    for amp in preamp_nodes:
        if amp.uid not in preamp_nodes_nobuild_uid:
            if restrictions['preamp_variety_list'] and\
                    not roadms[next(network.successors(amp)).uid].restrictions['preamp_variety_list']:
                if amp.params.type_variety not in restrictions['preamp_variety_list']:
                    raise AssertionError()


@pytest.mark.parametrize('prev_node_type, effective_pch_out_db', [('edfa', -20.0), ('fused', -22.0)])
def test_roadm_target_power(prev_node_type, effective_pch_out_db):
    ''' Check that egress power of roadm is equal to target power if input power is greater
    than target power else, that it is equal to input power. Use a simple two hops A-B-C topology
    for the test where the prev_node in ROADM B is either an amplifier or a fused, so that the target
    power can not be met in this last case.
    '''
    equipment = load_equipment(EQPT_LIBRARY_NAME)
    json_network = load_json(TEST_DIR / 'data/twohops_roadm_power_test.json')
    prev_node = next(n for n in json_network['elements'] if n['uid'] == 'west edfa in node B to ila2')
    json_network['elements'].remove(prev_node)
    if prev_node_type == 'edfa':
        prev_node = {'uid': 'west edfa in node B to ila2', 'type': 'Edfa'}
    elif prev_node_type == 'fused':
        prev_node = {'uid': 'west edfa in node B to ila2', 'type': 'Fused'}
        prev_node['params'] = {'loss': 0}
    json_network['elements'].append(prev_node)
    network = network_from_json(json_network, equipment)
    # Build the network once using the default power defined in SI in eqpt config
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, 
                                             equipment['SI']['default'].spacing))

    build_network(network, equipment, p_db, p_total_db)

    params = {}
    params['request_id'] = 0
    params['trx_type'] = ''
    params['trx_mode'] = ''
    params['source'] = 'trx node A'
    params['destination'] = 'trx node C'
    params['bidir'] = False
    params['nodes_list'] = ['trx node C']
    params['loose_list'] = ['strict']
    params['format'] = ''
    params['path_bandwidth'] = 100e9
    trx_params = trx_mode_params(equipment)
    params.update(trx_params)
    req = PathRequest(**params)
    path = compute_constrained_path(network, req)
    si = create_input_spectral_information(
        req.f_min, req.f_max, req.roll_off, req.baud_rate,
        req.power, req.spacing)
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            carriers_power_in_roadm = min([c.power.signal + c.power.nli + c.power.ase for c in si.carriers])
            si = el(si, degree=path[i+1].uid)
            if el.uid == 'roadm node B':
                print('input', carriers_power_in_roadm)
                assert el.effective_pch_out_db == effective_pch_out_db
                for carrier in si.carriers:
                    print(carrier.power.signal + carrier.power.nli + carrier.power.ase)
                    power = carrier.power.signal + carrier.power.nli + carrier.power.ase
                    if prev_node_type == 'edfa':
                        # edfa prev_node sets input power to roadm to a high enough value:
                        # Check that egress power of roadm is equal to target power
                        assert power == pytest.approx(db2lin(effective_pch_out_db - 30), rel=1e-3)
                    elif prev_node_type == 'fused':
                        # fused prev_node does reamplfy power after fiber propagation, so input power
                        # to roadm is low.
                        # Check that egress power of roadm is equalized to the min carrier input power.
                        assert power == pytest.approx(carriers_power_in_roadm, rel=1e-3)
        else:
            si = el(si)


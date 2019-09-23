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
from gnpy.core.utils import lin2db, load_json
from gnpy.core.elements import Fused, Roadm, Edfa
from gnpy.core.equipment import load_equipment, Amp, automatic_nch
from gnpy.core.network import network_from_json, build_network


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
            #replace edfa node by a fused node in the topology
            elem['type'] = 'Fused'
            elem.pop('type_variety')
            elem.pop('operational')
            elem['params'] = {'loss': 0}

            next_node_uid = next(conn['to_node'] for conn in json_network['connections'] \
                                 if conn['from_node'] == node_uid)
            previous_node_uid = next(conn['from_node'] for conn in json_network['connections'] \
                                 if conn['to_node'] == node_uid)

    network = network_from_json(json_network, equipment)
    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
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
        'preamp_variety_list':[],
        'booster_variety_list':[]
    },
    {
        'preamp_variety_list':[],
        'booster_variety_list':['booster_medium_gain']
    },
    {
        'preamp_variety_list':['preamp_medium_gain', 'preamp_high_gain', 'preamp_low_gain'],
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

    amp_nodes_nobuild_uid = [nd.uid for nd in network.nodes() \
        if isinstance(nd, Edfa) and isinstance(next(network.predecessors(nd)), Roadm)]
    preamp_nodes_nobuild_uid = [nd.uid for nd in network.nodes() \
        if isinstance(nd, Edfa) and isinstance(next(network.successors(nd)), Roadm)]
    amp_nodes_nobuild = {nd.uid : nd for nd in network.nodes() \
        if isinstance(nd, Edfa) and isinstance(next(network.predecessors(nd)), Roadm)}
    preamp_nodes_nobuild = {nd.uid : nd for nd in network.nodes() \
        if isinstance(nd, Edfa) and isinstance(next(network.successors(nd)), Roadm)}
    # roadm dict with restrictions before build
    roadms = {nd.uid: nd for nd in network.nodes() if isinstance(nd, Roadm)}
    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))

    build_network(network, equipment, p_db, p_total_db)

    amp_nodes = [nd for nd in network.nodes() \
        if isinstance(nd, Edfa) and isinstance(next(network.predecessors(nd)), Roadm)\
           and next(network.predecessors(nd)).restrictions['booster_variety_list']]

    preamp_nodes = [nd for nd in network.nodes() \
        if isinstance(nd, Edfa) and isinstance(next(network.successors(nd)), Roadm)\
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
                if not amp.params.type_variety in restrictions['booster_variety_list']:

                    raise AssertionError()
    for amp in preamp_nodes:
        if amp.uid not in preamp_nodes_nobuild_uid:
            if restrictions['preamp_variety_list'] and\
            not roadms[next(network.successors(amp)).uid].restrictions['preamp_variety_list']:
                if not amp.params.type_variety in restrictions['preamp_variety_list']:
                    raise AssertionError()

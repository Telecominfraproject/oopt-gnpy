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
from copy import deepcopy
import pytest
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.core.elements import Fused, Roadm, Edfa, Transceiver, EdfaOperational, EdfaParams, Fiber
from gnpy.core.parameters import FiberParams
from gnpy.core.network import build_network
from gnpy.tools.json_io import network_from_json, load_equipment, load_json, Amp
from gnpy.core.equipment import trx_mode_params
from gnpy.topology.request import PathRequest, compute_constrained_path, propagate
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


@pytest.mark.parametrize('power_dbm', [0, +1, -2])
@pytest.mark.parametrize('prev_node_type, effective_pch_out_db', [('edfa', -20.0), ('fused', -22.0)])
def test_roadm_target_power(prev_node_type, effective_pch_out_db, power_dbm):
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
    p_db = power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max,
                                             equipment['SI']['default'].spacing))

    build_network(network, equipment, p_db, p_total_db)

    params = {'request_id': 0,
              'trx_type': '',
              'trx_mode': '',
              'source': 'trx node A',
              'destination': 'trx node C',
              'bidir': False,
              'nodes_list': ['trx node C'],
              'loose_list': ['strict'],
              'format': '',
              'path_bandwidth': 100e9}
    trx_params = trx_mode_params(equipment)
    params.update(trx_params)
    req = PathRequest(**params)
    req.power = db2lin(power_dbm - 30)
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
                # if previous was an EDFA, power level at ROADM input is enough for the ROADM to apply its
                # target power (as specified in equipment ie -20 dBm)
                # if it is a Fused, the input power to the ROADM is smaller than the target power, and the
                # ROADM cannot apply this target. In this case, it is assumed that the ROADM has 0 dB loss
                # so the output power will be the same as the input power, which for this particular case
                # corresponds to -22dBm + power_dbm
                # next step (for ROADM modelling) will be to apply a minimum loss for ROADMs !
                if prev_node_type == 'edfa':
                    assert el.effective_pch_out_db == effective_pch_out_db
                if prev_node_type == 'fused':
                    # then output power == input_power == effective_pch_out_db + power_dbm
                    assert effective_pch_out_db + power_dbm ==\
                        pytest.approx(lin2db(carriers_power_in_roadm * 1e3), rel=1e-3)
                    assert el.effective_pch_out_db == effective_pch_out_db + power_dbm
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


def create_per_oms_request(network, eqpt, req_power):
    """ create request template
    """
    params = {'trx_type': '',
              'trx_mode': '',
              'bidir': False,
              'loose_list': ['strict', 'strict'],
              'format': '',
              'path_bandwidth': 100e9}
    trx_params = trx_mode_params(eqpt)
    params.update(trx_params)
    trxs = [e for e in network if isinstance(e, Transceiver)]
    req_list = []
    req_id = 0
    for trx in trxs:
        source = trx.uid
        roadm = next(n for n in network.successors(trx) if isinstance(n, Roadm))
        for degree in roadm.per_degree_pch_out_db.keys():
            node = next(n for n in network.nodes() if n.uid == degree)
            # find next roadm
            while not isinstance(node, Roadm):
                node = next(n for n in network.successors(node))
            next_roadm = node
            destination = next(n.uid for n in network.successors(next_roadm) if isinstance(n, Transceiver))
            params['request_id'] = req_id
            req_id += 1
            params['source'] = source
            params['destination'] = destination
            params['nodes_list'] = [degree, destination]
            req = PathRequest(**params)
            req.power = db2lin(req_power - 30)
            carrier = {key: getattr(req, key) for key in ['power', 'baud_rate', 'roll_off']}
            req.initial_spectrum = {(req.f_min + req.spacing * f):
                                    deepcopy(carrier) for f in range(1, req.nb_channel + 1)}
            req_list.append(req)
    # add one additional request crossing several roadms to have a complete view
    params['source'] = 'trx Rennes_STA'
    params['destination'] = 'trx Vannes_KBE'
    params['nodes_list'] = ['roadm Lannion_CAS', 'trx Vannes_KBE']
    params['bidir'] = True
    req = PathRequest(**params)
    req.power = db2lin(req_power - 30)
    carrier = {key: getattr(req, key) for key in ['power', 'baud_rate', 'roll_off']}
    req.initial_spectrum = {(req.f_min + req.spacing * f): deepcopy(carrier) for f in range(1, req.nb_channel + 1)}
    req_list.append(req)
    return req_list


def list_element_attr(element):
    """ give the list of keys to be checked depending on element type. List only the keys that are not
    created upon element effective propagation
    TODO: many parameters (location ones) look redondant: simplify ?
    """
    if isinstance(element, Roadm):
        return ['effective_pch_out_db', 'lat', 'latitude', 'lng', 'loc',
                'location', 'longitude', 'loss', 'metadata', 'name', 'operational', 'params', 'passive',
                'per_degree_pch_out_db', 'restrictions', 'type_variety']
        # dynamically created: 'effective_loss',
        # TODO: loss is not updated at all : dead param ?
    if isinstance(element, Edfa):
        return ['delta_p', 'effective_gain',
                'lat', 'latitude', 'lng', 'loc', 'location', 'longitude', 'metadata', 'name',
                'operational', 'out_voa', 'params', 'passive',
                'tilt_target']
        # TODO this exhaustive test highlighted that type_variety is not correctly updated from EdfaParams to
        # attributes in preamps
        # dynamically created only with channel propagation: 'att_in', 'channel_freq', 'effective_pch_out_db'
        # 'gprofile', 'interpol_dgt', 'interpol_gain_ripple', 'interpol_nf_ripple', 'nch',  'nf', 'pin_db', 'pout_db',
        # 'target_pch_out_db',
    if isinstance(element, EdfaOperational):
        return ['delta_p', 'gain_target', 'out_voa', 'tilt_target']
    if isinstance(element, EdfaParams):
        return ['allowed_for_design', 'dgt', 'dual_stage_model', 'f_max', 'f_min', 'gain_flatmax', 'gain_min',
                'gain_ripple', 'nf_fit_coeff', 'nf_model', 'nf_ripple', 'out_voa_auto', 'p_max', 'raman',
                'type_def', 'type_variety']
    if isinstance(element, Fiber):
        return ['lat', 'latitude', 'lng', 'loc', 'location',
                'longitude', 'loss', 'metadata', 'name', 'operational',
                'params', 'passive', 'pmd', 'type_variety']
        # dynamically created 'output_total_power', 'pch_out_db'
    if isinstance(element, FiberParams):
        return ['att_in', 'beta2', 'beta3', 'con_in', 'con_out', 'dispersion', 'dispersion_slope', 'f_loss_ref',
                'gamma', 'length', 'loss_coef', 'pmd_coef', 'pumps_loss_coef', 'raman_efficiency', 'ref_frequency',
                'ref_wavelength']
    if isinstance(element, Fused):
        return ['lat', 'latitude', 'lng', 'loc', 'location', 'longitude', 'loss', 'metadata', 'name',
                'operational', 'params', 'passive']

    raise ValueError('This type is not a supported element')


def verify_first_amp(base_amps, element_after_propagation, power_dbm, req_power, dp, pch_max):
    """ returns power difference after-before propagation according to the different cases:
    if first amp has a type_variety attribute, design does not apply a saturation verification on amplifiers
    and computed power out before propagation is power_dBm + dp
    else element was saturated (power = min(pch_max, power_dbm + dp)).
    in any case, saturation is applied during propagation, knowing that if
    design was saturated then its delta_p was changed with the required power reduction.
    I used the power to compute the expected gain of each amp wrt each cases.
    """
    if hasattr(base_amps[element_after_propagation.uid], 'type_variety'):
        # no saturation verification during design
        # case CC
        return min(pch_max, req_power + dp) - power_dbm - dp
    # case DD
    # saturation verification during design
    return min(pch_max, req_power + element_after_propagation.delta_p) - min(pch_max, power_dbm + dp)


def verify_other_amps(base_amps, element_after_propagation, previous_amp_was_saturated, power_dbm, req_power,
                      previous_deltap, pch_max):
    """ returns power difference after-before propagation according to the different cases:
    all combinations of saturation for previous amp and this amp are:
                 |                              this amp
    --------------------------------------------------------------------------
                 |               saturated if over pchmax   | not saturated
                 | saturated if |
    previous amp | over pchmax  |          FFF              |     EEE
                 | -----------------------------------------------------------
                 | not saturate |          HHH              |     GGG

    I used the power to compute the expected gain of each amp wrt each cases, knowing that if
    design was saturated then its delta_p was changed with the required power reduction
    """
    if hasattr(base_amps[element_after_propagation.uid], 'type_variety'):  # current amp not saturated
        if previous_amp_was_saturated:
            # case EEE
            # previous amp was possibly saturated during design and this one was not
            return min(pch_max, req_power + element_after_propagation.delta_p) -\
                   min(pch_max, req_power + previous_deltap) -\
                   (power_dbm + element_after_propagation.delta_p) +\
                   min(pch_max, power_dbm + previous_deltap)
        # case GGG
        return min(pch_max, req_power + element_after_propagation.delta_p) -\
               min(pch_max, req_power + previous_deltap) -\
               (power_dbm + element_after_propagation.delta_p) +\
               (power_dbm + previous_deltap)
    else:   # current amp not saturated
        if previous_amp_was_saturated:
            # case FFF
            return min(pch_max, req_power + element_after_propagation.delta_p) -\
                   min(pch_max, req_power + previous_deltap) -\
                   min(pch_max, power_dbm + element_after_propagation.delta_p) +\
                   min(pch_max, power_dbm + previous_deltap)
        # case HHH
        return min(pch_max, req_power + element_after_propagation.delta_p) -\
               min(pch_max, req_power + previous_deltap) -\
               min(pch_max, power_dbm + element_after_propagation.delta_p) +\
               power_dbm + previous_deltap


# all initial delta_p are null in topo file, so add random places to change this value
@pytest.mark.parametrize('amps_with_adjusted_delta_p', [[],
                         ['east edfa in Lorient_KMA to Vannes_KBE',
                          'east edfa in Stbrieuc to Rennes_STA',
                          'west edfa in Lannion_CAS to Morlaix',
                          'east edfa in a to b',
                          'west edfa in b to a']])
@pytest.mark.parametrize('power_dbm, req_power', [(0, 0), (0, -3), (3, 3), (0, 3), (3, 0),
                                                  (3, 1), (3, 5), (3, 2), (3, 4), (2, 4)])
def test_booster_target_power_and_gain(power_dbm, req_power, amps_with_adjusted_delta_p):
    """ Check that network design does not change after propagation
    except for gain in case of power_saturation during design and/or during propagation:
    - in power mode only:
        expected behaviour: target power out of roadm does not change
        so gain of booster should be reduced/augmented by the exact power difference
        the rest of the amplifier have unchanged gain
        except if augmentation leads to total_power above amplifier max power
         ie if amplifier saturates. then first amplifier in OMS is impacted
         eg
                                roadm -----booster (pmax 21dBm, 96 channels= 19.82dB)
        pdesign=0dBm pch= 0dBm,         ^ -20dBm  ^G=20dB, Pch=0dBm, Ptot=19.82dBm
        pdesign=0dBm pch= -3dBm         ^ -20dBm  ^G=17dB, Pch=-3dBm, Ptot=16.82dBm
        pdesign=3dBm pch= 3dBm          ^ -20dBm  ^G=23-1.82dB, Pch=1.18dBm, Ptot=21dBm
            amplifier can not handle 96x3dBm channels, amplifier saturation is considered for the choice
            of amplifier if no type variety has been provided, else the saturation is applied only upon propagation
        pdesign=0dBm pch= 3dBm          ^ -20dBm  ^G=23-1.82dB, Pch=1.18dBm, Ptot=21dBm
            amplifier can not handle 96x3dBm channels, amplifier selection has been done for 0dBm
            saturation is applied for all amps only during propagation
        design applies a saturation verification on amplifiers only if no type_variety is defined for it.
        this saturation leads to a power reduction to the max power in the amp library, which is also applied on
        the amp delta_p.
        This saturation occurs per amplifier and independantly from propagation.
        After design, upon propagation, the amplifier gain may change due to different total power used than
        during design (eg not the same nb of channels, not the same power per channel).
        This test also checks all the possible combinations and expected before/after propagation gain differences.
        it also checks delta_p is applied due to saturation during design
    """
    eqpt = load_equipment(EQPT_LIBRARY_NAME)
    eqpt['SI']['default'].power_dbm = power_dbm
    json_network = load_json(NETWORK_FILE_NAME)
    for element in json_network['elements']:
        for name in amps_with_adjusted_delta_p:
            if element['uid'] == name:
                element['operational']['delta_p'] = 1

    network = network_from_json(json_network, eqpt)
    # Build the network once using the default power defined in SI in eqpt config
    p_db = power_dbm
    p_total_db = p_db + lin2db(automatic_nch(eqpt['SI']['default'].f_min,
                                             eqpt['SI']['default'].f_max,
                                             eqpt['SI']['default'].spacing))
    base_network = deepcopy(network)
    base_amps = {amp.uid: amp for amp in base_network.nodes() if isinstance(amp, Edfa)}
    build_network(network, eqpt, p_db, p_total_db)
    # record network settings before propagating
    network_copy = deepcopy(network)
    # propagate on each oms
    req_list = create_per_oms_request(network, eqpt, req_power)
    paths = [compute_constrained_path(network, r) for r in req_list]
    _ = [propagate(p, r, eqpt) for p, r in zip(paths, req_list)]

    # systematic comparison of elements settings before and after propagation
    # all amps have 21 dBm max power
    pch_max = 21 - lin2db(96)
    for pth in paths:
        # check all elements except source and destination trx
        for el in pth:
            print(el)
        previous_deltap = None
        for i, element in enumerate(pth[1:-1]):
            # index of previous element in path is i
            element_is_first_amp = isinstance(element, Edfa) and isinstance(pth[i], Roadm)

            previous_amp_was_saturated = ((isinstance(element, Edfa) and isinstance(pth[i], Roadm))
                                          and not hasattr(base_amps[element.uid], 'type_variety'))
            if element.uid == 'west edfa in d to c':
                # OMS c to d has no booster but one preamp: the power difference is held there
                element_is_first_amp = True
            element_copy = next(n for n in network_copy.nodes() if n.uid == element.uid)
            for key in list_element_attr(element):
                parameters = getattr(element, key)
                parameters_copy = getattr(element_copy, key)
                print(element.uid, key, parameters, parameters_copy)
                if not isinstance(parameters, (EdfaOperational, EdfaParams, FiberParams)):
                    if not key == 'effective_gain':
                        assert parameters == parameters_copy
                    else:
                        dp = 0 if element.uid not in amps_with_adjusted_delta_p else 1
                        if element_is_first_amp:
                            # if first amp has a type_variety attribute, design does not apply a saturation
                            assert element.effective_gain - element_copy.effective_gain ==\
                                pytest.approx(verify_first_amp(base_amps, element, power_dbm,
                                                               req_power, dp, pch_max), abs=1e-2)
                        else:
                            # then all combinations of saturation for previous amp and this amp:
                            #              |                              this amp
                            # --------------------------------------------------------------------------
                            #              |               saturated if over pchmax   | not saturated
                            #              | saturated if |
                            # previous amp | over pchmax  |          FFF              |     EEE
                            #              | -----------------------------------------------------------
                            #              | not saturate |          HHH              |     GGG
                            #
                            # I used the power to compute the expected gain of each amp wrt each cases, knowing that if
                            # design was saturated then its delta_p was changed with the required power reduction

                            assert element.effective_gain - element_copy.effective_gain ==\
                                        pytest.approx(verify_other_amps(base_amps, element, previous_amp_was_saturated,
                                            power_dbm, req_power, previous_deltap, pch_max), abs=1e-2)
                        # if amp has no type_variety then its output total power is computed to choose a type_variety
                        # in the library, and if all amps have max_power below this vaue, a reduction of the target
                        # power is applied on delta_p. This test checks that the reduction is correctly computed
                        previous_deltap = element.delta_p
                        if hasattr(base_amps[element.uid], 'type_variety'):
                            # case AA
                            previous_amp_was_saturated = False
                            assert element.delta_p == dp
                        else:
                            # case BB
                            previous_amp_was_saturated = True
                            assert element.delta_p == pytest.approx(min(power_dbm + dp, pch_max) - power_dbm, abs=1e-2)
                else:
                    for one_param in list_element_attr(parameters):
                        assert getattr(parameters, one_param) == getattr(parameters_copy, one_param)

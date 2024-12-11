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
from numpy.testing import assert_allclose
from numpy import ndarray, mean
from copy import deepcopy
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.core.elements import Fused, Roadm, Edfa, Transceiver, EdfaOperational, EdfaParams, Fiber
from gnpy.core.parameters import FiberParams, RoadmParams, FusedParams
from gnpy.core.network import build_network, design_network
from gnpy.tools.json_io import network_from_json, load_equipment, load_json, Amp, _equipment_from_json
from gnpy.core.equipment import trx_mode_params
from gnpy.topology.request import PathRequest, compute_constrained_path, propagate
from gnpy.core.info import create_input_spectral_information, Carrier
from gnpy.core.utils import db2lin, dbm2watt, merge_amplifier_restrictions
from gnpy.core.exceptions import ConfigurationError, NetworkTopologyError


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILE_NAME = DATA_DIR / 'testTopology_expected.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}
# adding tests to check the roadm restrictions

# mark node_uid amps as fused for testing purpose
@pytest.mark.parametrize("node_uid", ['east edfa in Lannion_CAS to Stbrieuc'])
def test_no_amp_feature(node_uid):
    """Check that booster is not placed on a roadm if fused is specified
    test_parser covers partly this behaviour. This test should guaranty that the
    feature is preserved even if convert is changed
    """
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
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
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
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
        equipment['Edfa'][entry['type_variety']] = Amp.from_json(EXTRA_CONFIGS, **entry)
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
    """test that restriction is correctly applied if provided in eqpt_config and if no Edfa type
    were provided in the network json
    """
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


@pytest.mark.parametrize('roadm_type_variety, roadm_b_maxloss', [('default', 0),
                                                                 ('example_detailed_impairments', 16.5)])
@pytest.mark.parametrize('power_dbm', [0, +1, -2])
@pytest.mark.parametrize('prev_node_type, effective_pch_out_db', [('edfa', -20.0), ('fused', -22.0)])
def test_roadm_target_power(prev_node_type, effective_pch_out_db, power_dbm, roadm_type_variety, roadm_b_maxloss):
    """Check that egress power of roadm is equal to target power if input power is greater
    than target power else, that it is equal to input power. Use a simple two hops A-B-C topology
    for the test where the prev_node in ROADM B is either an amplifier or a fused, so that the target
    power can not be met in this last case.
    """
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    equipment['SI']['default'].power_dbm = power_dbm
    json_network = load_json(TEST_DIR / 'data/twohops_roadm_power_test.json')
    prev_node = next(n for n in json_network['elements'] if n['uid'] == 'west edfa in node B to ila2')
    json_network['elements'].remove(prev_node)
    roadm_b = next(element for element in json_network['elements'] if element['uid'] == 'roadm node B')
    roadm_b['type_variety'] = roadm_type_variety
    if prev_node_type == 'edfa':
        prev_node = {'uid': 'west edfa in node B to ila2', 'type': 'Edfa'}
    elif prev_node_type == 'fused':
        prev_node = {'uid': 'west edfa in node B to ila2', 'type': 'Fused'}
        prev_node['params'] = {'loss': 0}
    json_network['elements'].append(prev_node)
    network = network_from_json(json_network, equipment)
    nb_channel = automatic_nch(equipment['SI']['default'].f_min, equipment['SI']['default'].f_max,
                               equipment['SI']['default'].spacing)
    p_total_db = power_dbm + lin2db(nb_channel)

    build_network(network, equipment, power_dbm, p_total_db)

    params = {'request_id': 0,
              'trx_type': '',
              'trx_mode': '',
              'source': 'trx node A',
              'destination': 'trx node C',
              'bidir': False,
              'nodes_list': ['trx node C'],
              'loose_list': ['strict'],
              'format': '',
              'path_bandwidth': 100e9,
              'effective_freq_slot': None,
              'nb_channel': nb_channel,
              'power': dbm2watt(power_dbm),
              'tx_power': dbm2watt(power_dbm)
              }
    trx_params = trx_mode_params(equipment)
    params.update(trx_params)
    req = PathRequest(**params)
    req.power = db2lin(power_dbm - 30)
    path = compute_constrained_path(network, req)
    si = create_input_spectral_information(
        f_min=req.f_min, f_max=req.f_max, roll_off=req.roll_off, baud_rate=req.baud_rate,
        spacing=req.spacing, tx_osnr=req.tx_osnr, tx_power=req.tx_power)
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            power_in_roadm = si.signal + si.ase + si.nli
            si = el(si, degree=path[i + 1].uid, from_degree=path[i - 1].uid)
            power_out_roadm = si.signal + si.ase + si.nli
            if el.uid == 'roadm node B':
                # if previous was an EDFA, power level at ROADM input is enough for the ROADM to apply its
                # target power (as specified in equipment ie -20 dBm)
                # if it is a Fused, the input power to the ROADM is smaller than the target power, and the
                # ROADM cannot apply this target. If the ROADM has 0 dB loss the output power will be the same
                # as the input power, which for this particular case corresponds to -22dBm + power_dbm.
                # If ROADM has a minimum losss, then output power will be -22dBm + power_dbm - ROADM loss. !
                if prev_node_type == 'edfa':
                    # edfa prev_node sets input power to roadm to a high enough value:
                    # check that target power is correctly set in the ROADM
                    assert_allclose(el.ref_pch_out_dbm, effective_pch_out_db, rtol=1e-3)
                    # Check that egress power of roadm is equal to target power
                    assert_allclose(power_out_roadm, db2lin(effective_pch_out_db - 30), rtol=1e-3)
                if prev_node_type == 'fused':
                    # fused prev_node does not reamplify power after fiber propagation, so input power
                    # to roadm is low.
                    # check that target power correctly reports power_dbm from previous propagation
                    assert_allclose(el.ref_pch_out_dbm, effective_pch_out_db + power_dbm - roadm_b_maxloss, rtol=1e-3)
                    # Check that egress power of roadm is not equalized:
                    # power out is the same as power in minus the ROADM loss.
                    assert_allclose(power_out_roadm, power_in_roadm / db2lin(roadm_b_maxloss), rtol=1e-3)
                    assert effective_pch_out_db + power_dbm ==\
                        pytest.approx(lin2db(min(power_in_roadm) * 1e3), rel=1e-3)
        else:
            si = el(si)


def create_per_oms_request(network, eqpt, req_power):
    """Create requests between every adjacent ROADMs + one additional request crossing several ROADMs
    """
    nb_channel = automatic_nch(eqpt['SI']['default'].f_min, eqpt['SI']['default'].f_max,
                               eqpt['SI']['default'].spacing)
    params = {
        'trx_type': '',
        'trx_mode': '',
        'bidir': False,
        'loose_list': ['strict', 'strict'],
        'format': '',
        'path_bandwidth': 100e9,
        'effective_freq_slot': None,
        'nb_channel': nb_channel,
        'power': dbm2watt(req_power),
        'tx_power': dbm2watt(req_power)
    }
    trx_params = trx_mode_params(eqpt)
    params.update(trx_params)
    trxs = [e for e in network if isinstance(e, Transceiver)]
    req_list = []
    req_id = 0
    for trx in trxs:
        source = trx.uid
        roadm = next(n for n in network.successors(trx) if isinstance(n, Roadm))
        for degree in roadm.per_degree_pch_out_dbm.keys():
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
            req.power = dbm2watt(req_power)
            carrier = {key: getattr(req, key) for key in ['baud_rate', 'roll_off', 'tx_osnr']}
            carrier['label'] = ""
            carrier['slot_width'] = req.spacing
            carrier['delta_pdb'] = 0
            carrier['tx_power'] = 1e-3
            req.initial_spectrum = {(req.f_min + req.spacing * f): Carrier(**carrier)
                                    for f in range(1, req.nb_channel + 1)}
            req_list.append(req)
    # add one additional request crossing several roadms to have a complete view
    params['source'] = 'trx Rennes_STA'
    params['destination'] = 'trx Vannes_KBE'
    params['nodes_list'] = ['roadm Lannion_CAS', 'trx Vannes_KBE']
    params['bidir'] = True
    req = PathRequest(**params)
    req.power = dbm2watt(req_power)
    carrier = {key: getattr(req, key) for key in ['baud_rate', 'roll_off', 'tx_osnr']}
    carrier['label'] = ""
    carrier['slot_width'] = req.spacing
    carrier['delta_pdb'] = 0
    carrier['tx_power'] = 1e-3
    req.initial_spectrum = {(req.f_min + req.spacing * f): Carrier(**carrier) for f in range(1, req.nb_channel + 1)}
    req_list.append(req)
    return req_list


def list_element_attr(element):
    """Return the list of keys to be checked depending on element type. List only the keys that are not
    created upon element effective propagation
    """

    if isinstance(element, Roadm):
        return ['uid', 'name', 'metadata', 'operational', 'type_variety', 'target_pch_out_dbm',
                'passive', 'restrictions', 'per_degree_pch_out_dbm',
                'target_psd_out_mWperGHz', 'per_degree_pch_psd']
        # Dynamically created: 'effective_loss',
    if isinstance(element, RoadmParams):
        return ['target_pch_out_dbm', 'target_psd_out_mWperGHz', 'per_degree_pch_out_db', 'per_degree_pch_psd',
                'add_drop_osnr', 'pmd', 'restrictions']
    if isinstance(element, Edfa):
        return ['variety_list', 'uid', 'name', 'params', 'metadata', 'operational',
                'passive', 'effective_gain', 'delta_p', 'tilt_target', 'out_voa']
        # TODO this exhaustive test highlighted that type_variety is not correctly updated from EdfaParams to
        # attributes in preamps
        # Dynamically created only with channel propagation: 'att_in', 'channel_freq', 'effective_pch_out_db'
        # 'gprofile', 'interpol_dgt', 'interpol_gain_ripple', 'interpol_nf_ripple', 'nch',  'nf', 'pin_db', 'pout_db',
        # 'target_pch_out_db',
    if isinstance(element, FusedParams):
        return ['loss']
    if isinstance(element, EdfaOperational):
        return ['delta_p', 'gain_target', 'out_voa', 'tilt_target']
    if isinstance(element, EdfaParams):
        return ['f_min', 'f_max', 'type_variety', 'type_def', 'gain_flatmax', 'gain_min', 'p_max', 'nf_model',
                'dual_stage_model', 'nf_fit_coeff', 'nf_ripple', 'dgt', 'gain_ripple', 'out_voa_auto',
                'allowed_for_design', 'raman']
    if isinstance(element, Fiber):

        return ['uid', 'name', 'params', 'metadata', 'operational', 'type_variety', 'passive',
                'lumped_losses', 'z_lumped_losses']
        # Dynamically created 'output_total_power', 'pch_out_db'
    if isinstance(element, FiberParams):
        return ['_length', '_att_in', '_con_in', '_con_out', '_ref_frequency', '_ref_wavelength',
                '_dispersion', '_dispersion_slope', '_dispersion', '_f_dispersion_ref',
                '_gamma', '_pmd_coef', '_loss_coef',
                '_f_loss_ref', '_lumped_losses']
    if isinstance(element, Fused):
        return ['uid', 'name', 'params', 'metadata', 'operational', 'loss', 'passive']
    if isinstance(element, FusedParams):
        return ['loss']
    return ['should never come here']


# all initial delta_p are null in topo file, so add random places to change this value
@pytest.mark.parametrize('amp_with_deltap_one', [[],
                                                 ['east edfa in Lorient_KMA to Vannes_KBE',
                                                  'east edfa in Stbrieuc to Rennes_STA',
                                                  'west edfa in Lannion_CAS to Morlaix',
                                                  'east edfa in a to b',
                                                  'west edfa in b to a']])
@pytest.mark.parametrize('power_dbm, req_power', [(0, 0), (0, -3), (3, 3), (0, 3), (3, 0),
                                                  (3, 1), (3, 5), (3, 2), (3, 4), (2, 4)])
def test_compare_design_propagation_settings(power_dbm, req_power, amp_with_deltap_one):
    """Check that network design does not change after propagation except for gain in
    case of power_saturation during design and/or during propagation:
    - in power mode only:
        expected behaviour: target power out of roadm does not change
        so gain of booster should be reduced/augmented by the exact power difference;
        the following amplifiers on the OMS have unchanged gain except if augmentation
        of channel power on booster leads to total_power above amplifier max power,
        ie if amplifier saturates.

                          roadm -----booster (pmax 21dBm, 96 channels= 19.82dB)
        pdesign=0dBm pch= 0dBm,         ^ -20dBm  ^G=20dB, Pch=0dBm, Ptot=19.82dBm
        pdesign=0dBm pch= -3dBm         ^ -20dBm  ^G=17dB, Pch=-3dBm, Ptot=16.82dBm
        pdesign=3dBm pch= 3dBm          ^ -20dBm  ^G=23-1.82dB, Pch=1.18dBm, Ptot=21dBm
            amplifier can not handle 96x3dBm channels, amplifier saturation is considered
            for the choice of amplifier during design
        pdesign=0dBm pch= 3dBm          ^ -20dBm  ^G=23-1.82dB, Pch=1.18dBm, Ptot=21dBm
            amplifier can not handle 96x3dBm channels during propagation, amplifier selection
            has been done for 0dBm. Saturation is applied for all amps only during propagation

        Design applies a saturation verification on amplifiers.
        This saturation leads to a power reduction to the max power in the amp library, which
        is also applied on the amp delta_p and independantly from propagation.

        After design, upon propagation, the amplifier gain and applied delta_p may also change
        if total power exceeds max power (eg not the same nb of channels, not the same power per channel
        compared to design).

        This test also checks all the possible combinations and expected before/after propagation
        gain differences. It also checks delta_p applied due to saturation during design.
    """
    eqpt = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    eqpt['SI']['default'].power_dbm = power_dbm
    json_network = load_json(NETWORK_FILE_NAME)
    for element in json_network['elements']:
        # Initialize a value for delta_p
        if element['type'] == 'Edfa':
            element['operational']['delta_p'] = 0 + element['operational']['out_voa'] \
                if element['operational']['out_voa'] is not None else 0
            # apply a 1 dB delta_p on the set of amps
            if element['uid'] in amp_with_deltap_one:
                element['operational']['delta_p'] = 1

    network = network_from_json(json_network, eqpt)
    # Build the network once using the default power defined in SI in eqpt config
    p_db = power_dbm
    p_total_db = p_db + lin2db(automatic_nch(eqpt['SI']['default'].f_min,
                                             eqpt['SI']['default'].f_max,
                                             eqpt['SI']['default'].spacing))
    build_network(network, eqpt, p_db, p_total_db, verbose=False)
    # record network settings before propagating
    # propagate on each oms
    req_list = create_per_oms_request(network, eqpt, req_power)
    paths = [compute_constrained_path(network, r) for r in req_list]

    # systematic comparison of elements settings before and after propagation
    # all amps have 21 dBm max power
    pch_max = 21 - lin2db(96)
    for path, req in zip(paths, req_list):
        # check all elements except source and destination trx
        # in order to have clean initialization, use deecopy of paths
        design_network(req, network, eqpt, verbose=False)
        network_copy = deepcopy(network)
        pth = deepcopy(path)
        _ = propagate(pth, req, eqpt)
        for i, element in enumerate(pth[1:-1]):
            element_is_first_amp = False
            # index of previous element in path is i
            if (isinstance(element, Edfa) and isinstance(pth[i], Roadm)) or element.uid == 'west edfa in d to c':
                # oms c to d has no booster but one preamp: the power difference is hold there
                element_is_first_amp = True
            # find the element with the same id in the network_copy
            element_copy = next(n for n in network_copy.nodes() if n.uid == element.uid)
            for key in list_element_attr(element):
                if not isinstance(getattr(element, key),
                                  (EdfaOperational, EdfaParams, FiberParams, RoadmParams, FusedParams)):
                    if not key == 'effective_gain':
                        # for all keys, before and after design should be the same except for gain (in power mode)
                        if isinstance(getattr(element, key), ndarray):
                            if len(getattr(element, key)) > 0:
                                assert getattr(element, key) == getattr(element_copy, key)
                            else:
                                assert len(getattr(element_copy, key)) == 0
                        else:
                            assert getattr(element, key) == getattr(element_copy, key)
                    else:
                        dp = element.out_voa if element.uid not in amp_with_deltap_one else element.out_voa + 1
                        # check that target power is correctly set
                        assert element.target_pch_out_dbm == req_power + dp
                        # check that designed gain is exactly applied except if target power exceeds max power, then
                        # gain is slightly less than the one computed during design for the noiseless reference,
                        # because during propagation, noise has accumulated, additing to signal.
                        # check that delta_p is unchanged unless for saturation
                        if element.target_pch_out_dbm > pch_max:
                            assert element.effective_gain == pytest.approx(element_copy.effective_gain, abs=2e-2)
                        else:
                            assert element.effective_gain == element_copy.effective_gain
                        # check that delta_p is unchanged unless for saturation
                        assert element.delta_p == element_copy.delta_p
                        if element_is_first_amp:
                            # if element is first amp on path, then it is the one that will saturate if req_power is
                            # too high
                            assert mean(element.pch_out_dbm) ==\
                                pytest.approx(min(pch_max, req_power + element.delta_p - element.out_voa), abs=2e-2)
                        # check that delta_p is unchanged unless due to saturation
                        assert element.delta_p == pytest.approx(min(req_power + dp, pch_max) - req_power, abs=1e-2)
                        # check that delta_p is unchanged unless for saturation
                else:
                    # for all subkeys, before and after design should be the same
                    for subkey in list_element_attr(getattr(element, key)):
                        if isinstance(getattr(getattr(element, key), subkey), list):
                            assert getattr(getattr(element, key), subkey) == getattr(getattr(element_copy, key), subkey)
                        elif isinstance(getattr(getattr(element, key), subkey), dict):
                            for value1, value2 in zip(getattr(getattr(element, key), subkey).values(),
                                                      getattr(getattr(element_copy, key), subkey).values()):
                                assert all(value1 == value2)
                        elif isinstance(getattr(getattr(element, key), subkey), ndarray):
                            assert_allclose(getattr(getattr(element, key), subkey),
                                            getattr(getattr(element_copy, key), subkey), rtol=1e-12)
                        else:
                            assert getattr(getattr(element, key), subkey) == getattr(getattr(element_copy, key), subkey)


@pytest.mark.parametrize("restrictions, fail", [
    ({'preamp_variety_list': [], 'booster_variety_list':[]}, False),
    ({'preamp_variety_list': ['std_medium_gain', 'std_low_gain'], 'booster_variety_list':['std_medium_gain']}, False),
    # the two next amp type_variety do not exist
    ({'preamp_variety_list': [], 'booster_variety_list':['booster_medium_gain']}, True),
    ({'preamp_variety_list': ['std_medium_gain', 'preamp_high_gain'], 'booster_variety_list':[]}, True)])
def test_wrong_restrictions(restrictions, fail):
    """Check that sanity_check correctly raises an error when restriction is incorrect and that library
    correctly includes restrictions.
    """
    json_data = load_json(EQPT_FILENAME)
    # define wrong restriction
    json_data['Roadm'][0]['restrictions'] = restrictions
    if fail:
        with pytest.raises(ConfigurationError):
            _ = _equipment_from_json(json_data, EXTRA_CONFIGS)
    else:
        equipment = _equipment_from_json(json_data, EXTRA_CONFIGS)
        assert equipment['Roadm']['example_test'].restrictions == restrictions


@pytest.mark.parametrize('roadm, from_degree, to_degree, expected_impairment_id, expected_type', [
    ('roadm Lannion_CAS', 'trx Lannion_CAS', 'east edfa in Lannion_CAS to Corlay', 1, 'add'),
    ('roadm Lannion_CAS', 'west edfa in Lannion_CAS to Stbrieuc', 'east edfa in Lannion_CAS to Corlay', 0, 'express'),
    ('roadm Lannion_CAS', 'west edfa in Lannion_CAS to Stbrieuc', 'trx Lannion_CAS', 2, 'drop'),
    ('roadm h', 'west edfa in h to g', 'trx h', None, 'drop')
])
def test_roadm_impairments(roadm, from_degree, to_degree, expected_impairment_id, expected_type):
    """Check that impairment id and types are correct
    """
    json_data = load_json(NETWORK_FILE_NAME)
    for el in json_data['elements']:
        if el['uid'] == 'roadm Lannion_CAS':
            el['type_variety'] = 'example_detailed_impairments'
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    build_network(network, equipment, 0.0, 20.0)
    roadm = next(n for n in network.nodes() if n.uid == roadm)
    assert roadm.get_roadm_path(from_degree, to_degree).path_type == expected_type
    assert roadm.get_roadm_path(from_degree, to_degree).impairment_id == expected_impairment_id


@pytest.mark.parametrize('type_variety, from_degree, to_degree, impairment_id, expected_type', [
    (None, 'trx Lannion_CAS', 'east edfa in Lannion_CAS to Corlay', 1, 'add'),
    ('default', 'trx Lannion_CAS', 'east edfa in Lannion_CAS to Corlay', 3, 'add'),
    (None, 'west edfa in Lannion_CAS to Stbrieuc', 'east edfa in Lannion_CAS to Corlay', None, 'express')
])
def test_roadm_per_degree_impairments(type_variety, from_degree, to_degree, impairment_id, expected_type):
    """Check that impairment type is correct also if per degree impairment is defined
    """
    json_data = load_json(EQPT_FILENAME)
    assert 'type_variety' not in json_data['Roadm'][2]
    json_data['Roadm'][2]['roadm-path-impairments'] = [
        {
            "roadm-path-impairments-id": 1,
            "roadm-add-path": [{
                "frequency-range": {
                    "lower-frequency": 191.3e12,
                    "upper-frequency": 196.1e12
                },
                "roadm-osnr": 41,
            }]
        }, {
            "roadm-path-impairments-id": 3,
            "roadm-add-path": [{
                "frequency-range": {
                    "lower-frequency": 191.3e12,
                    "upper-frequency": 196.1e12
                },
                "roadm-inband-crosstalk": 0,
                "roadm-osnr": 20,
                "roadm-noise-figure": 23
            }]
        }]
    equipment = _equipment_from_json(json_data, EXTRA_CONFIGS)
    assert equipment['Roadm']['default'].type_variety == 'default'

    json_data = load_json(NETWORK_FILE_NAME)
    for el in json_data['elements']:
        if el['uid'] == 'roadm Lannion_CAS' and type_variety is not None:
            el['type_variety'] = type_variety
            el['params'] = {
                "per_degree_impairments": [
                    {
                        "from_degree": from_degree,
                        "to_degree": to_degree,
                        "impairment_id": impairment_id
                    }]
            }
    network = network_from_json(json_data, equipment)
    build_network(network, equipment, 0.0, 20.0)
    roadm = next(n for n in network.nodes() if n.uid == 'roadm Lannion_CAS')
    assert roadm.get_roadm_path(from_degree, to_degree).path_type == expected_type
    assert roadm.get_roadm_path(from_degree, to_degree).impairment_id == impairment_id


@pytest.mark.parametrize('from_degree, to_degree, impairment_id, error, message', [
    ('trx Lannion_CAS', 'east edfa in Lannion_CAS to Corlay', 2, NetworkTopologyError,
     'Roadm roadm Lannion_CAS path_type is defined as drop but it should be add'),  # wrong path_type
    ('trx Lannion_CAS', 'east edfa toto', 1, ConfigurationError,
     'Roadm roadm Lannion_CAS has wrong from-to degree uid trx Lannion_CAS - east edfa toto'),  # wrong degree
    ('trx Lannion_CAS', 'east edfa in Lannion_CAS to Corlay', 11, NetworkTopologyError,
     'ROADM roadm Lannion_CAS: impairment profile id 11 is not defined in library')  # wrong impairment_id
])
def test_wrong_roadm_per_degree_impairments(from_degree, to_degree, impairment_id, error, message):
    """Check that wrong per degree definitions are correctly catched
    """
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    json_data = load_json(NETWORK_FILE_NAME)
    for el in json_data['elements']:
        if el['uid'] == 'roadm Lannion_CAS':
            el['type_variety'] = 'example_detailed_impairments'
            el['params'] = {
                "per_degree_impairments": [
                    {
                        "from_degree": from_degree,
                        "to_degree": to_degree,
                        "impairment_id": impairment_id
                    }]
            }
    network = network_from_json(json_data, equipment)
    with pytest.raises(error, match=message):
        build_network(network, equipment, 0.0, 20.0)


@pytest.mark.parametrize('path_type, type_variety, expected_pmd, expected_pdl, expected_osnr, freq', [
    ('express', 'default', 5.0e-12, 0.5, None, [191.3e12]),  # roadm instance parameters pre-empts library
    ('express', 'example_test', 5.0e-12, 0.5, None, [191.3e12]),
    ('express', 'example_detailed_impairments', 0, 0, None, [191.3e12]),  # detailed parameters pre-empts global ones
    ('add', 'default', 5.0e-12, 0.5, None, [191.3e12]),
    ('add', 'example_test', 5.0e-12, 0.5, None, [191.3e12]),
    ('add', 'example_detailed_impairments', 0, 0, 41, [191.3e12]),
    ('add', 'example_detailed_impairments', [0, 0], [0.5, 0], [35, 41], [188.5e12, 191.3e12])])
def test_impairment_initialization(path_type, type_variety, expected_pmd, expected_pdl, expected_osnr, freq):
    """Check that impairments are correctly initialized, with this order:
    - use equipment roadm impairments if no impairment are set in the ROADM instance
    - use roadm global impairment if roadm global impairment are set
    - use roadm detailed impairment for the corresponding path_type if roadm type_variety has detailed impairments
    - use roadm per degree impairment if they are defined
    """
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    extra_params = equipment['Roadm'][type_variety].__dict__
    roadm_config = {
        "uid": "roadm Lannion_CAS",
        "params": {
            "add_drop_osnr": 38,
            "pmd": 5.0e-12,
            "pdl": 0.5
        }
    }
    if type_variety != 'default':
        roadm_config["type_variety"] = type_variety
    roadm_config['params'] = merge_amplifier_restrictions(roadm_config['params'], extra_params)
    roadm = Roadm(**roadm_config)
    roadm.set_roadm_paths(from_degree='tata', to_degree='toto', path_type=path_type)
    assert roadm.get_roadm_path(from_degree='tata', to_degree='toto').path_type == path_type
    assert_allclose(roadm.get_impairment('roadm-pmd', freq, from_degree='tata', degree='toto'),
                    expected_pmd, rtol=1e-12)
    assert_allclose(roadm.get_impairment('roadm-pdl', freq, from_degree='tata', degree='toto'),
                    expected_pdl, rtol=1e-12)
    if path_type == 'add':
        # we assume for simplicity that add contribution is the same as drop contribution
        # add_drop_osnr_db = 10log10(1/add_osnr + 1/drop_osnr)
        if type_variety in ['default', 'example_test']:
            assert_allclose(roadm.get_impairment('roadm-osnr', freq, from_degree='tata', degree='toto'),
                            roadm.params.add_drop_osnr + lin2db(2), rtol=1e-12)
        else:
            assert_allclose(roadm.get_impairment('roadm-osnr', freq, from_degree='tata', degree='toto'),
                            expected_osnr, rtol=1e-12)

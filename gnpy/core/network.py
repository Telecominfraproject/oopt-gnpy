#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.network
=================

This module contains functions for constructing networks of network elements.
'''

from gnpy.core.convert import convert_file
from networkx import DiGraph
from numpy import arange
from logging import getLogger
from os import path
from operator import itemgetter
from gnpy.core import elements
from gnpy.core.elements import Fiber, Edfa, Transceiver, Roadm, Fused
from gnpy.core.equipment import edfa_nf
from gnpy.core.units import UNITS
from gnpy.core.utils import load_json, save_json, round2float, db2lin, lin2db
from sys import exit
from collections import namedtuple

logger = getLogger(__name__)

def load_network(filename, equipment, name_matching = False):
    json_filename = ''
    if filename.suffix.lower() == '.xls':
        logger.info('Automatically generating topology JSON file')
        json_filename = convert_file(filename, name_matching)
    elif filename.suffix.lower() == '.json':
        json_filename = filename
    else:
        raise ValueError(f'unsuported topology filename extension {filename.suffix.lower()}')
    json_data = load_json(json_filename)
    return network_from_json(json_data, equipment)

def save_network(filename, network):
    filename_output = path.splitext(filename)[0] + '_auto_design.json'
    json_data = network_to_json(network)
    save_json(json_data, filename_output)

def network_from_json(json_data, equipment):
    # NOTE|dutc: we could use the following, but it would tie our data format
    #            too closely to the graph library
    # from networkx import node_link_graph
    g = DiGraph()
    for el_config in json_data['elements']:
        typ = el_config.pop('type')
        variety = el_config.pop('type_variety', 'default')
        if typ in equipment and variety in equipment[typ]:
            extra_params = equipment[typ][variety]._asdict()            
            if extra_params.get('type_def','') == 'hybrid':
                raman_model = extra_params['nf_model']
                type_variety = extra_params['type_variety']
                extra_params_edfa = equipment['Edfa'][raman_model.edfa_variety]._asdict()
                extra_params.update(extra_params_edfa)
                extra_params['type_def'] = 'hybrid'
                extra_params['raman_model'] = raman_model
                extra_params['type_variety'] = type_variety
            el_config.setdefault('params', {}).update(extra_params)
        elif typ in ['Edfa', 'Fiber']: #catch it now because the code will crash later!
            print( f'The {typ} of variety type {variety} was not recognized:'
                    '\nplease check it is properly defined in the eqpt_config json file')
            exit()
        cls = getattr(elements, typ)
        el = cls(**el_config)
        g.add_node(el)

    nodes = {k.uid: k for k in g.nodes()}

    for cx in json_data['connections']:
        from_node, to_node = cx['from_node'], cx['to_node']
        try:
            g.add_edge(nodes[from_node], nodes[to_node])
        except KeyError:
            msg = f'In {__name__} network_from_json function:\n\tcan not find {from_node} or {to_node} defined in {cx}'
            print(msg)
            exit(1)

    return g

def network_to_json(network):
    data = {
        'elements': [n.to_json for n in network]
        }
    connections = {
        'connections': [{"from_node": n.uid,
                         "to_node": next_n.uid}
                        for n in network
                        for next_n in network.successors(n) if next_n is not None]
        }
    data.update(connections)
    return data

def select_edfa(gain_target, power_target, equipment):
    """amplifer selection algorithm
    @Orange Jean-Luc Augé
    """
    Edfa_list = namedtuple('Edfa_list', 'variety power gain nf')
    TARGET_EXTENDED_GAIN = 2.1
    #MAX_EXTENDED_GAIN = 5
    edfa_dict = equipment['Edfa']
    pin = power_target - gain_target

    edfa_list = [Edfa_list(
                variety=edfa_variety,
                power=min(
                    pin
                    +edfa.gain_flatmax
                    +TARGET_EXTENDED_GAIN,
                    edfa.p_max
                    )
                    -power_target,
                gain=edfa.gain_flatmax-gain_target,
                nf=edfa_nf(gain_target, edfa_variety, equipment)) \
                for edfa_variety, edfa in edfa_dict.items()
                if edfa.allowed_for_design]

    acceptable_gain_list = \
    list(filter(lambda x : x.gain>-TARGET_EXTENDED_GAIN, edfa_list))
    if len(acceptable_gain_list) < 1:
        #no amplifier satisfies the required gain, so pick the highest gain:
        gain_max = max(edfa_list, key=itemgetter(2)).gain
        #pick up all amplifiers that share this max gain:
        acceptable_gain_list = \
        list(filter(lambda x : x.gain-gain_max>-0.1, edfa_list))
    acceptable_power_list = \
    list(filter(lambda x : x.power>=0, acceptable_gain_list))
    if len(acceptable_power_list) < 1:
        #no amplifier satisfies the required power, so pick the highest power:
        power_max = \
        max(acceptable_gain_list, key=itemgetter(1)).power
        #pick up all amplifiers that share this max gain:
        acceptable_power_list = \
        list(filter(lambda x : x.power-power_max>-0.1, acceptable_gain_list))
    # gain and power requirements are resolved,
    #       =>chose the amp with the best NF among the acceptable ones:
    return min(acceptable_power_list, key=itemgetter(3)).variety #filter on NF


def set_roadm_loss(network, equipment, pref_ch_db):
    roadms = [roadm for roadm in network if isinstance(roadm, Roadm)]
    power_mode = equipment['Spans']['default'].power_mode
    default_roadm_loss = equipment['Roadms']['default'].gain_mode_default_loss
    pout_target = equipment['Roadms']['default'].power_mode_pout_target
    roadm_loss = pref_ch_db - pout_target

    for roadm in roadms:
        if power_mode:
            roadm.loss = roadm_loss
            roadm.target_pch_out_db = pout_target
        elif roadm.loss == None:
            roadm.loss = default_roadm_loss

def target_power(dp_from_gain, network, node, equipment): #get_fiber_dp
    SPAN_LOSS_REF = 20
    POWER_SLOPE = 0.3
    power_mode = equipment['Spans']['default'].power_mode
    dp_range = list(equipment['Spans']['default'].delta_power_range_db)
    node_loss = span_loss(network, node)

    dp_gain_mode = 0
    try:
        dp_power_mode = round2float((node_loss - SPAN_LOSS_REF) * POWER_SLOPE, dp_range[2])
        dp_power_mode = max(dp_range[0], dp_power_mode)
        dp_power_mode = min(dp_range[1], dp_power_mode)
    except KeyError:
        print(f'invalid delta_power_range_db definition in eqpt_config[Spans]'
              f'delta_power_range_db: [lower_bound, upper_bound, step]')
        exit()

    if dp_from_gain:
        dp_power_mode = dp_from_gain
        dp_gain_mode = dp_from_gain
    if isinstance(node, Roadm):
        dp_power_mode = 0

    dp = dp_power_mode if power_mode else dp_gain_mode
    #print(f'{repr(node)} delta power in:\n{dp}dB')

    return dp


def prev_node_generator(network, node):
    """fused spans interest:
    iterate over all predecessors while they are Fused or Fiber type"""
    try:
        prev_node = next(n for n in network.predecessors(node))
    except StopIteration:
        msg = f'In {__name__} prev_node_generator function:\n\t{node.uid} is not properly connected, please check network topology'
        print(msg)
        logger.critical(msg)
        exit(1)
    # yield and re-iterate
    if isinstance(prev_node, Fused) or isinstance(node, Fused):
        yield prev_node
        yield from prev_node_generator(network, prev_node)
    else:
        StopIteration

def next_node_generator(network, node):
    """fused spans interest:
    iterate over all successors while they are Fused or Fiber type"""
    try:
        next_node = next(n for n in network.successors(node))
    except StopIteration:
        print(f'In {__name__} next_node_generator function:\n\t{node.uid}  is not properly connected, please check network topology')
        exit(1)        
    # yield and re-iterate
    if isinstance(next_node, Fused) or isinstance(node, Fused):
        yield next_node
        yield from next_node_generator(network, next_node)
    else:
        StopIteration

def span_loss(network, node):
    """Fused span interest:
    return the total span loss of all the fibers spliced by a Fused node"""
    loss = node.loss if node.passive else 0
    try:
        prev_node = next(n for n in network.predecessors(node))
        if isinstance(prev_node, Fused):
            loss += sum(n.loss for n in prev_node_generator(network, node))
    except StopIteration:
        pass
    try:
        next_node = next(n for n in network.successors(node))
        if isinstance(next_node, Fused):
            loss += sum(n.loss for n in next_node_generator(network, node))
    except StopIteration:
        pass
    return loss

def find_first_node(network, node):
    """Fused node interest:
    returns the 1st node at the origin of a succession of fused nodes
    (aka no amp in between)"""
    this_node = node
    for this_node in prev_node_generator(network, node):
        pass
    return this_node

def find_last_node(network, node):
    """Fused node interest:
    returns the last node in a succession of fused nodes
    (aka no amp in between)"""
    this_node = node
    for this_node in next_node_generator(network, node):
        pass
    return this_node

def set_amplifier_voa(amp, pref_total_db, power_mode):
    VOA_MARGIN = 0
    if amp.operational.out_voa is None:
        if power_mode:
            gain_target = amp.operational.gain_target
            pout = pref_total_db + amp.dp_db
            voa = min(amp.params.p_max-pout,
                      amp.params.gain_flatmax-amp.operational.gain_target)
            voa = round2float(max(voa, 0), 0.5) - VOA_MARGIN if amp.params.out_voa_auto else 0
            amp.dp_db = amp.dp_db + voa
            amp.operational.gain_target = amp.operational.gain_target + voa
        else:
            voa = 0 # no output voa optimization in gain mode
        amp.operational.out_voa = voa

def set_egress_amplifier(network, roadm, equipment, pref_total_db):
    power_mode = equipment['Spans']['default'].power_mode
    next_oms = (n for n in network.successors(roadm) if not isinstance(n, Transceiver))
    for oms in next_oms:
        #go through all the OMS departing from the Roadm
        node = roadm
        prev_node = roadm
        next_node = oms
        # if isinstance(next_node, Fused): #support ROADM wo egress amp for metro applications
        #     node = find_last_node(next_node)
        #     next_node = next(n for n in network.successors(node))
        #     next_node = find_last_node(next_node)
        prev_dp = 0
        dp = 0
        while True:
        #go through all nodes in the OMS (loop until next Roadm instance)
            if isinstance(node, Edfa):
                node_loss = span_loss(network, prev_node)
                if isinstance(prev_node, Roadm):
                    # gain readings shouldnot set the power level after roadm if exact roadm loss are not known
                    # => use gain readings only to set span power differences 
                    # but leave auto_design to set the average powe level after roadm
                    dp_from_gain = None
                else:
                    dp_from_gain = prev_dp + node.operational.gain_target - node_loss \
                        if node.operational.gain_target > 0 else None
                dp = target_power(dp_from_gain, network, next_node, equipment)
                gain_target = node_loss + dp - prev_dp

                if power_mode:
                    node.dp_db = dp
                node.operational.gain_target = gain_target

                if node.params.type_variety == '':
                    power_target = pref_total_db + dp
                    edfa_variety = select_edfa(gain_target, power_target, equipment)
                    extra_params = equipment['Edfa'][edfa_variety]
                    node.params.update_params(extra_params._asdict())
                set_amplifier_voa(node, pref_total_db, power_mode)
            if isinstance(next_node, Roadm) or isinstance(next_node, Transceiver):
                break
            prev_dp = dp
            prev_node = node
            node = next_node
            # print(f'{node.uid}')
            next_node = next(n for n in network.successors(node))


def add_egress_amplifier(network, node):
    next_nodes = [n for n in network.successors(node)
        if not (isinstance(n, Transceiver) or isinstance(n, Fused) or isinstance(n, Edfa))]
        #no amplification for fused spans or TRX
    for i, next_node in enumerate(next_nodes):
        network.remove_edge(node, next_node)
        amp = Edfa(
                    uid = f'Edfa{i}_{node.uid}',
                    params = {},
                    operational = {
                        'gain_target': 0,
                        'tilt_target': 0,
                    })
        network.add_node(amp)
        network.add_edge(node, amp)
        network.add_edge(amp, next_node)


def calculate_new_length(fiber_length, bounds, target_length):
    if fiber_length < bounds.stop:
        return fiber_length, 1

    n_spans = int(fiber_length // target_length)

    length1 = fiber_length / (n_spans+1)
    delta1 = target_length-length1
    result1 = (length1, n_spans+1)

    length2 = fiber_length / n_spans
    delta2 = length2-target_length
    result2 = (length2, n_spans)

    if (bounds.start<=length1<=bounds.stop) and not(bounds.start<=length2<=bounds.stop):
        result = result1
    elif (bounds.start<=length2<=bounds.stop) and not(bounds.start<=length1<=bounds.stop):
        result = result2
    else:
        result = result1 if delta1 < delta2 else result2

    return result


def split_fiber(network, fiber, bounds, target_length, equipment):
    new_length, n_spans = calculate_new_length(fiber.length, bounds, target_length)
    if n_spans == 1:
        return

    try:
        next_node = next(network.successors(fiber))
        prev_node = next(network.predecessors(fiber))
    except StopIteration:

        print(f'In {__name__} split_fiber function:\n\t{fiber.uid}   is not properly connected, please check network topology')
        exit()

    network.remove_node(fiber)

    fiber_params = fiber.params._asdict()
    fiber_params['length'] = new_length / UNITS[fiber.params.length_units]
    fiber_params['con_in'] = fiber.con_in
    fiber_params['con_out'] = fiber.con_out
    
    for span in range(n_spans):
        new_span = Fiber(uid =      f'{fiber.uid}_({span+1}/{n_spans})',
                          metadata = fiber.metadata,
                          params = fiber_params)
        network.add_edge(prev_node, new_span)
        prev_node = new_span
    network.add_edge(prev_node, next_node)

def add_connector_loss(fibers, con_in, con_out, EOL):
    for fiber in fibers:
        if fiber.con_in is None: fiber.con_in = con_in
        if fiber.con_out is None:
            fiber.con_out = con_out #con_out includes EOL
        else:
            fiber.con_out = fiber.con_out+EOL

def add_fiber_padding(network, fibers, padding):
    """last_fibers = (fiber for n in network.nodes()
                         if not (isinstance(n, Fiber) or isinstance(n, Fused))
                         for fiber in network.predecessors(n)
                         if isinstance(fiber, Fiber))"""
    for fiber in fibers:
        this_span_loss = span_loss(network, fiber)
        try:
            next_node = next(network.successors(fiber))
        except StopIteration:
            msg = f'In {__name__} add_fiber_padding function:\n\t{fiber.uid}   is not properly connected, please check network topology'
            print(msg)
            logger.critical(msg)
            exit(1)            
        if this_span_loss < padding and not (isinstance(next_node, Fused)):
            #add a padding att_in at the input of the 1st fiber:
            #address the case when several fibers are spliced together
            first_fiber = find_first_node(network, fiber)
            if first_fiber.att_in is None:
                first_fiber.att_in = padding - this_span_loss
            else :
                first_fiber.att_in = first_fiber.att_in + padding - this_span_loss

def build_network(network, equipment, pref_ch_db, pref_total_db):
    default_span_data = equipment['Spans']['default']
    max_length = int(default_span_data.max_length * UNITS[default_span_data.length_units])
    min_length = max(int(default_span_data.padding/0.2*1e3),50_000)
    bounds = range(min_length, max_length)
    target_length = max(min_length, 90_000)
    con_in = default_span_data.con_in
    con_out = default_span_data.con_out + default_span_data.EOL
    padding = default_span_data.padding

    #set raodm loss for gain_mode before to build network
    set_roadm_loss(network, equipment, pref_ch_db)
    fibers = [f for f in network.nodes() if isinstance(f, Fiber)]
    add_connector_loss(fibers, con_in, con_out, default_span_data.EOL)
    add_fiber_padding(network, fibers, padding)
    # don't group split fiber and add amp in the same loop
    # =>for code clarity (at the expense of speed):
    for fiber in fibers:
        split_fiber(network, fiber, bounds, target_length, equipment)

    amplified_nodes = [n for n in network.nodes()
                        if isinstance(n, Fiber) or isinstance(n, Roadm)]
    for node in amplified_nodes:
        add_egress_amplifier(network, node)

    roadms = [r for r in network.nodes() if isinstance(r, Roadm)]
    for roadm in roadms:
        set_egress_amplifier(network, roadm, equipment, pref_total_db)

    #support older json input topology wo Roadms:
    if len(roadms) == 0:
        trx = [t for t in network.nodes() if isinstance(t, Transceiver)]
        for t in trx:
            set_egress_amplifier(network, t, equipment, pref_total_db)


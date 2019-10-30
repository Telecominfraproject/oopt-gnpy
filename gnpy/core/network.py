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
from scipy.interpolate import interp1d
from logging import getLogger
from os import path
from operator import itemgetter, attrgetter
from gnpy.core import elements
from gnpy.core.elements import Fiber, Edfa, Transceiver, Roadm, Fused, RamanFiber
from gnpy.core.equipment import edfa_nf
from gnpy.core.exceptions import ConfigurationError, NetworkTopologyError
from gnpy.core.units import UNITS
from gnpy.core.utils import (load_json, save_json, round2float, db2lin,
                            merge_amplifier_restrictions)
from gnpy.core.science_utils import SimParams
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
            extra_params = equipment[typ][variety]
            temp = el_config.setdefault('params', {})
            temp = merge_amplifier_restrictions(temp, extra_params.__dict__)
            el_config['params'] = temp
        elif typ in ['Edfa', 'Fiber']: # catch it now because the code will crash later!
            raise ConfigurationError(f'The {typ} of variety type {variety} was not recognized:'
                    '\nplease check it is properly defined in the eqpt_config json file')
        cls = getattr(elements, typ)
        el = cls(**el_config)
        g.add_node(el)

    nodes = {k.uid: k for k in g.nodes()}

    for cx in json_data['connections']:
        from_node, to_node = cx['from_node'], cx['to_node']
        try:
            if isinstance(nodes[from_node], Fiber):
                edge_length = nodes[from_node].params.length
            else:
                edge_length = 0.01
            g.add_edge(nodes[from_node], nodes[to_node], weight = edge_length)
        except KeyError:
            raise NetworkTopologyError(f'can not find {from_node} or {to_node} defined in {cx}')

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

def select_edfa(raman_allowed, gain_target, power_target, equipment, uid, restrictions=None):
    """amplifer selection algorithm
    @Orange Jean-Luc Augé
    """
    Edfa_list = namedtuple('Edfa_list', 'variety power gain_min nf')
    TARGET_EXTENDED_GAIN = equipment['Span']['default'].target_extended_gain

    # for roadm restriction only: create a dict including not allowed for design amps
    # because main use case is to have specific radm amp which are not allowed for ILA
    # with the auto design
    edfa_dict = {name: amp for (name, amp) in equipment['Edfa'].items()
        if restrictions is None or name in restrictions}

    pin = power_target - gain_target

    # create 2 list of available amplifiers with relevant attributes for their selection

    # edfa list with:
    # extended gain min allowance of 3dB: could be parametrized, but a bit complex
    # extended gain max allowance TARGET_EXTENDED_GAIN is coming from eqpt_config.json
    # power attribut include power AND gain limitations
    edfa_list = [Edfa_list(
                variety=edfa_variety,
                power=min(
                    pin
                    +edfa.gain_flatmax
                    +TARGET_EXTENDED_GAIN,
                    edfa.p_max
                    )
                    -power_target,
                gain_min=
                    gain_target+3
                    -edfa.gain_min,
                nf=edfa_nf(gain_target, edfa_variety, equipment)) \
                for edfa_variety, edfa in edfa_dict.items()
                if ((edfa.allowed_for_design or restrictions is not None) and not edfa.raman)]

    #consider a Raman list because of different gain_min requirement: 
    #do not allow extended gain min for Raman
    raman_list = [Edfa_list(
                variety=edfa_variety,
                power=min(
                    pin
                    +edfa.gain_flatmax
                    +TARGET_EXTENDED_GAIN,
                    edfa.p_max
                    )
                    -power_target,
                gain_min=
                    gain_target
                    -edfa.gain_min,
                nf=edfa_nf(gain_target, edfa_variety, equipment))
                for edfa_variety, edfa in edfa_dict.items()
                if (edfa.allowed_for_design and edfa.raman)] \
                if raman_allowed else []

    #merge raman and edfa lists
    amp_list = edfa_list + raman_list

    #filter on min gain limitation: 
    acceptable_gain_min_list = [x for x in amp_list if x.gain_min>0]

    if len(acceptable_gain_min_list) < 1:
        #do not take this empty list into account for the rest of the code
        #but issue a warning to the user and do not consider Raman
        #Raman below min gain should not be allowed because i is meant to be a design requirement
        #and raman padding at the amplifier input is impossible!

        if len(edfa_list) < 1:
            raise ConfigurationError(f'auto_design could not find any amplifier \
                    to satisfy min gain requirement in node {uid} \
                    please increase span fiber padding')
        else:
            # TODO: convert to logging
            print(
                f'\x1b[1;31;40m'\
                + f'WARNING: target gain in node {uid} is below all available amplifiers min gain: \
                    amplifier input padding will be assumed, consider increase span fiber padding instead'\
                + '\x1b[0m'
                )
            acceptable_gain_min_list = edfa_list

    #filter on gain+power limitation:
    #this list checks both the gain and the power requirement
    #because of the way .power is calculated in the list
    acceptable_power_list = [x for x in acceptable_gain_min_list if x.power>0]
    if len(acceptable_power_list) < 1:
        #no amplifier satisfies the required power, so pick the highest power(s):
        power_max = max(acceptable_gain_min_list, key=attrgetter('power')).power
        #check and pick if other amplifiers may have a similar gain/power
        #allow a 0.3dB power range 
        #this allows to chose an amplifier with a better NF subsequentely
        acceptable_power_list = [x for x in acceptable_gain_min_list
                                 if x.power-power_max>-0.3]

    
    # gain and power requirements are resolved,
    #       =>chose the amp with the best NF among the acceptable ones:
    selected_edfa = min(acceptable_power_list, key=attrgetter('nf')) #filter on NF
    #check what are the gain and power limitations of this amp
    power_reduction = round(min(selected_edfa.power, 0),2)
    if power_reduction < -0.5:
        print(
            f'\x1b[1;31;40m'\
            + f'WARNING: target gain and power in node {uid}\n \
    is beyond all available amplifiers capabilities and/or extended_gain_range:\n\
    a power reduction of {power_reduction} is applied\n'\
            + '\x1b[0m'
            )


    return selected_edfa.variety, power_reduction

def target_power(network, node, equipment): #get_fiber_dp
    SPAN_LOSS_REF = 20
    POWER_SLOPE = 0.3
    power_mode = equipment['Span']['default'].power_mode
    dp_range = list(equipment['Span']['default'].delta_power_range_db)
    node_loss = span_loss(network, node)

    try:
        dp = round2float((node_loss - SPAN_LOSS_REF) * POWER_SLOPE, dp_range[2])
        dp = max(dp_range[0], dp)
        dp = min(dp_range[1], dp)
    except KeyError:
        raise ConfigurationError(f'invalid delta_power_range_db definition in eqpt_config[Span]'
              f'delta_power_range_db: [lower_bound, upper_bound, step]')

    if isinstance(node, Roadm):
        dp = 0

    return dp

def prev_node_generator(network, node):
    """fused spans interest:
    iterate over all predecessors while they are Fused or Fiber type"""
    try:
        prev_node = next(n for n in network.predecessors(node))
    except StopIteration:
        raise NetworkTopologyError(f'Node {node.uid} is not properly connected, please check network topology')
    # yield and re-iterate
    if isinstance(prev_node, Fused) or isinstance(node, Fused) and not isinstance(prev_node, Roadm):
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
        raise NetworkTopologyError('Node {node.uid} is not properly connected, please check network topology')
    # yield and re-iterate
    if isinstance(next_node, Fused) or isinstance(node, Fused) and not isinstance(next_node, Roadm):
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

def set_amplifier_voa(amp, power_target, power_mode):
    VOA_MARGIN = 1 #do not maximize the VOA optimization
    if amp.out_voa is None:
        if power_mode:
            gain_target = amp.effective_gain
            voa = min(amp.params.p_max-power_target,
                      amp.params.gain_flatmax-amp.effective_gain)
            voa = max(round2float(max(voa, 0), 0.5) - VOA_MARGIN, 0) if amp.params.out_voa_auto else 0
            amp.delta_p = amp.delta_p + voa
            amp.effective_gain = amp.effective_gain + voa
        else:
            voa = 0 # no output voa optimization in gain mode
        amp.out_voa = voa

def set_egress_amplifier(network, roadm, equipment, pref_total_db):
    power_mode = equipment['Span']['default'].power_mode
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
        
        if node.per_degree_target_pch_out_db:
            # find the target power on this degree
            try:
                prev_dp = next(el["target_pch_out_db"] for el in \
                    node.per_degree_target_pch_out_db if el["to_node"]==next_node.uid)
            except StopIteration:
                # if no target power is defined on this degree use the global one
                prev_dp = getattr(node.params, 'target_pch_out_db', 0)
        else:
            # if no per degree target power is given use the global one
            prev_dp = getattr(node.params, 'target_pch_out_db', 0)
        dp = prev_dp
        prev_voa = 0
        voa = 0
        while True:
        #go through all nodes in the OMS (loop until next Roadm instance)
            if isinstance(node, Edfa):
                node_loss = span_loss(network, prev_node)
                voa = node.out_voa if node.out_voa else 0
                if node.delta_p is None:
                    dp = target_power(network, next_node, equipment)
                else:
                    dp = node.delta_p
                gain_from_dp = node_loss + dp - prev_dp + prev_voa
                if node.effective_gain is None or power_mode:
                    gain_target = gain_from_dp
                else: #gain mode with effective_gain 
                    gain_target = node.effective_gain
                    dp = prev_dp - node_loss + gain_target

                power_target = pref_total_db + dp         

                raman_allowed = False
                if isinstance(prev_node, Fiber):
                    max_fiber_lineic_loss_for_raman = \
                            equipment['Span']['default'].max_fiber_lineic_loss_for_raman
                    raman_allowed = prev_node.params.loss_coef < max_fiber_lineic_loss_for_raman

                # implementation of restrictions on roadm boosters
                if isinstance(prev_node,Roadm):
                    if prev_node.restrictions['booster_variety_list']:
                        restrictions = prev_node.restrictions['booster_variety_list']
                    else:
                        restrictions = None
                elif isinstance(next_node,Roadm):
                    # implementation of restrictions on roadm preamp
                    if next_node.restrictions['preamp_variety_list']:
                        restrictions = next_node.restrictions['preamp_variety_list']
                    else:
                        restrictions = None
                else:
                    restrictions = None

                if node.params.type_variety == '':                   
                    edfa_variety, power_reduction = select_edfa(raman_allowed, 
                                   gain_target, power_target, equipment, node.uid, restrictions)
                    extra_params = equipment['Edfa'][edfa_variety]
                    node.params.update_params(extra_params.__dict__)
                    dp += power_reduction
                    gain_target += power_reduction
                elif node.params.raman and not raman_allowed:
                    print(
                        f'\x1b[1;31;40m'\
                        + f'WARNING: raman is used in node {node.uid}\n \
                but fiber lineic loss is above threshold\n'\
                        + '\x1b[0m'
                        )                    
                                
                node.delta_p = dp if power_mode else None
                node.effective_gain = gain_target
                set_amplifier_voa(node, power_target, power_mode)
            if isinstance(next_node, Roadm) or isinstance(next_node, Transceiver):
                break
            prev_dp = dp
            prev_voa = voa
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
                    metadata = {
                        'location': {
                            'latitude':  (node.lat * 2 + next_node.lat * 2) / 4,
                            'longitude': (node.lng * 2 + next_node.lng * 2) / 4,
                            'city':      node.loc.city,
                            'region':    node.loc.region,
                        }
                    },
                    operational = {
                        'gain_target': None,
                        'tilt_target': 0,
                    })
        network.add_node(amp)
        if isinstance(node,Fiber):
            edgeweight = node.params.length
        else:
            edgeweight = 0.01
        network.add_edge(node, amp, weight = edgeweight)
        network.add_edge(amp, next_node, weight = 0.01)


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
        raise NetworkTopologyError(f'Fiber {fiber.uid} is not properly connected, please check network topology')

    network.remove_node(fiber)

    fiber_params = fiber.params._asdict()
    fiber_params['length'] = new_length / UNITS[fiber.params.length_units]
    fiber_params['con_in'] = fiber.con_in
    fiber_params['con_out'] = fiber.con_out

    f = interp1d([prev_node.lng, next_node.lng], [prev_node.lat, next_node.lat])
    xpos = [prev_node.lng + (next_node.lng - prev_node.lng) * (n+1)/(n_spans+1) for n in range(n_spans)]
    ypos = f(xpos)
    for span, lng, lat in zip(range(n_spans), xpos, ypos):
        new_span = Fiber(uid = f'{fiber.uid}_({span+1}/{n_spans})',
                          metadata = {
                            'location': {
                                'latitude':  lat,
                                'longitude': lng,
                                'city':      fiber.loc.city,
                                'region':    fiber.loc.region,
                            }
                          },
                          params = fiber_params)
        if isinstance(prev_node,Fiber):
            edgeweight = prev_node.params.length
        else:
            edgeweight = 0.01
        network.add_edge(prev_node, new_span, weight = edgeweight)
        prev_node = new_span
    if isinstance(prev_node,Fiber):
        edgeweight = prev_node.params.length
    else:
        edgeweight = 0.01    
    network.add_edge(prev_node, next_node, weight = edgeweight)

def add_connector_loss(network, fibers, default_con_in, default_con_out, EOL):
    for fiber in fibers:
        if fiber.con_in is None: fiber.con_in = default_con_in
        if fiber.con_out is None: fiber.con_out = default_con_out
        next_node = next(n for n in network.successors(fiber))
        if not isinstance(next_node, Fused):
            fiber.con_out += EOL

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
            raise NetworkTopologyError(f'Fiber {fiber.uid} is not properly connected, please check network topology')
        if this_span_loss < padding and not (isinstance(next_node, Fused)):
            #add a padding att_in at the input of the 1st fiber:
            #address the case when several fibers are spliced together
            first_fiber = find_first_node(network, fiber)
            # in order to support no booster , fused might be placed
            # just after a roadm: need to check that first_fiber is really a fiber
            if isinstance(first_fiber,Fiber):
                if first_fiber.att_in is None:
                    first_fiber.att_in = padding - this_span_loss
                else:
                    first_fiber.att_in = first_fiber.att_in + padding - this_span_loss

def build_network(network, equipment, pref_ch_db, pref_total_db):
    default_span_data = equipment['Span']['default']
    max_length = int(default_span_data.max_length * UNITS[default_span_data.length_units])
    min_length = max(int(default_span_data.padding/0.2*1e3),50_000)
    bounds = range(min_length, max_length)
    target_length = max(min_length, 90_000)
    default_con_in = default_span_data.con_in
    default_con_out = default_span_data.con_out
    padding = default_span_data.padding

    #set roadm loss for gain_mode before to build network
    fibers = [f for f in network.nodes() if isinstance(f, Fiber)]
    add_connector_loss(network, fibers, default_con_in, default_con_out, default_span_data.EOL)
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

def load_sim_params(filename):
    sim_params = load_json(filename)
    return SimParams(params=sim_params)

def configure_network(network, sim_params):
    for node in network.nodes:
        if isinstance(node, RamanFiber):
            node.sim_params = sim_params

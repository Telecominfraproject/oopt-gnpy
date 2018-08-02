#!/usr/bin/env python3

'''
gnpy.core.network
=================

This module contains functions for constructing networks of network elements.
'''

from gnpy.core.convert import convert_file
from networkx import DiGraph
from numpy import arange
from logging import getLogger
from operator import itemgetter
from gnpy.core import elements
from gnpy.core.elements import Fiber, Edfa, Transceiver, Roadm, Fused
from gnpy.core.equipment import edfa_nf
from gnpy.core.units import UNITS
from gnpy.core.utils import load_json
from gnpy.core.utils import round2float
from sys import exit

logger = getLogger(__name__)

def load_network(filename, equipment):
    json_filename = ''
    if filename.suffix.lower() == '.xls':
        logger.info('Automatically generating topology JSON file')        
        json_filename = convert_file(filename)
    elif filename.suffix.lower() == '.json':
        json_filename = filename
    else:
        raise ValueError(f'unsuported topology filename extension {filename.suffix.lower()}')
    json_data = load_json(json_filename)
    return network_from_json(json_data, equipment)

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
            el_config.setdefault('params', {}).update(extra_params._asdict())
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
        g.add_edge(nodes[from_node], nodes[to_node])

    return g

def select_edfa(ingress_span_loss, equipment):
    """amplifer selection algorithm
    @Orange Jean-Luc Augé
    """
    #TODO |jla add power requirement in the selection criteria
    TARGET_EXTENDED_GAIN = 2.1
    #MAX_EXTENDED_GAIN = 5
    edfa_dict = equipment['Edfa']
    edfa_list = [(edfa_variety, 
                edfa_dict[edfa_variety].gain_flatmax-ingress_span_loss,
                edfa_nf(ingress_span_loss, edfa_variety, equipment)) \
                for edfa_variety in edfa_dict]
    acceptable_edfa_list = list(filter(lambda x : x[1]>-TARGET_EXTENDED_GAIN, edfa_list))
    if len(acceptable_edfa_list) < 1: 
        #no amplifier satisfies the required gain, so pick the highest gain one:
        return max(edfa_list, key=itemgetter(1))[0]
    else:
        #chose the amp with the best NF among the acceptable ones:
        return min(acceptable_edfa_list, key=itemgetter(2))[0]

def set_roadm_loss(network, equipment, power_mode, roadm_loss):
    roadms = (roadm for roadm in network if isinstance(roadm, Roadm))
    default_roadm_loss = equipment['Roadms']['default'].gain_mode_default_loss
    for roadm in roadms:
        if power_mode:
            roadm.loss = roadm_loss
        elif roadm.loss == None:
            roadm.loss = default_roadm_loss

def set_fiber_dp(network, fiber, equipment):
    SPAN_LOSS_REF = 20
    POWER_SLOPE = 0.3
    dp_range = list(equipment['Spans']['default'].delta_power_range_db)
    fiber_loss = span_loss(network, fiber)
    try:
        dp = round2float((fiber_loss - SPAN_LOSS_REF) * POWER_SLOPE, dp_range[2])
        dp = max(dp_range[0], dp)
        dp = min(dp_range[1], dp)
    except KeyError:
        print(f'invalid delta_power_range_db definition in eqpt_config[Spans]'
              f'delta_power_range_db: [lower_bound, upper_bound, step]')
        exit()
    print(f'{repr(fiber)} launched delta power:        {dp}dB')
    return dp

def set_edfa_dp(network, path, equipment):
    """only called in power_mode
    set the amplifier power target as a function of the gain target
    or as a function of the span loss and fiber type in automatic design"""
    path_amps = (amp for amp in path if isinstance(amp, Edfa))
    prev_dp = 0
    for amp in path_amps:
        next_node = next(network.successors(amp))
        prev_node = next(network.predecessors(amp))
        prev_node_loss = span_loss(network, prev_node)
        if isinstance(next_node, Roadm): #ingress amp: set dp = 0
            dp = 0
        elif amp.operational.gain_target > 0:
            dp = prev_dp + amp.operational.gain_target - prev_node_loss
        else:
            #automatic design
            dp = set_fiber_dp(network, next_node, equipment)
        amp.dp_db = dp
        prev_dp = dp

def prev_fiber_node_generator(network, node):
    """fused spans interest:
    iterate over all predecessors while they are Fused or Fiber type"""
    prev_node = next(n for n in network.predecessors(node))
    # yield and re-iterate
    if isinstance(prev_node, Fused) or isinstance(node, Fused):
        yield prev_node
        yield from prev_fiber_node_generator(network, prev_node)
    else:
        StopIteration

def next_fiber_node_generator(network, node):
    """fused spans interest:
    iterate over all successors while they are Fused or Fiber type"""
    next_node = next(n for n in network.successors(node))
    # yield and re-iterate
    if isinstance(next_node, Fused) or isinstance(node, Fused):
        yield next_node
        yield from next_fiber_node_generator(network, next_node)
    else:
        StopIteration 

def span_loss(network, node):
    """Fused span interest:
    return the total span loss of all the fibers spliced by a Fused node"""
    loss = node.loss if node.passive else 0
    prev_node = next(n for n in network.predecessors(node))
    next_node = next(n for n in network.successors(node))
    if isinstance(prev_node, Fused):
        return loss + sum(n.loss for n in prev_fiber_node_generator(network, node))
    elif isinstance(next_node, Fused):
        return loss + sum(n.loss for n in next_fiber_node_generator(network, node))
    else: 
        return loss

def find_first_fiber(network, node):
    """Fused span interest:
    returns the 1st fiber at the origin of a succession of spliced fibers"""
    fiber = node
    for fiber in prev_fiber_node_generator(network, node):
        pass
    return fiber

def add_egress_amplifier(network, node, equipment):
    if isinstance(node, Edfa):
        return

    next_nodes = (n for n in network.successors(node)
        if not (isinstance(n, Transceiver) or isinstance(n, Fused)))
        #no amplification for fused spans or TRX

    #do not set the gain in power mode: will be done later: so set the gain to 0
    power_mode = equipment['Spans']['default'].power_mode
    total_loss = span_loss(network, node)
    gain_target = 0 if power_mode else total_loss
    for i, next_node in enumerate(next_nodes):
        if isinstance(next_node, Edfa):
            if next_node.operational.gain_target == 0:
                next_node.operational.gain_target = gain_target
        else:
            network.remove_edge(node, next_node)
            edfa_variety = select_edfa(total_loss, equipment)
            extra_params = equipment['Edfa'][edfa_variety]
            amp = Edfa(
                        uid = f'Edfa{i}_{node.uid}',
                        params = extra_params._asdict(),
                        operational = {
                            'gain_target': gain_target,
                            'tilt_target': 0,
                        })            
            network.add_node(amp)
            network.add_edge(node,amp)
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
        add_egress_amplifier(network, fiber, equipment)
        return

    next_node = next(network.successors(fiber))
    prev_node = next(network.predecessors(fiber))
    network.remove_edge(fiber, next_node)
    network.remove_edge(prev_node, fiber)
    network.remove_node(fiber)
    # update connector loss parameter with default values
    fiber_params = fiber.params._asdict()
    fiber_params['con_in'] = fiber.con_in
    fiber_params['con_out'] = fiber.con_out
    new_spans = [
        Fiber(
            uid =      f'{fiber.uid}_({span}/{n_spans})',
            metadata = fiber.metadata,
            params = fiber_params
        ) for span in range(n_spans)
    ]
    
    new_spans[0].length = new_length
    network.add_node(new_spans[0])
    network.add_edge(prev_node, new_spans[0])
    prev_node = new_spans[0]
    for new_span in new_spans[1:]:
        new_span.length = new_length
        network.add_node(new_span)
        network.add_edge(prev_node, new_span)
        add_egress_amplifier(network, prev_node, equipment)
        prev_node = new_span
    network.add_edge(prev_node, next_node)
    add_egress_amplifier(network, prev_node, equipment)

def add_connector_loss(fibers, con_in, con_out):
    for fiber in fibers:
        if fiber.con_in is None: fiber.con_in = con_in
        if fiber.con_out is None: fiber.con_out = con_out

def add_fiber_padding(network, fibers, padding):
    """last_fibers = (fiber for n in network.nodes()
                         if not (isinstance(n, Fiber) or isinstance(n, Fused))
                         for fiber in network.predecessors(n)
                         if isinstance(fiber, Fiber))"""
    for fiber in fibers:
        fiber_loss = span_loss(network, fiber)
        next_node = next(network.successors(fiber))
        if fiber_loss < padding and not (isinstance(next_node, Fused)):
            #add a padding att_in at the input of the 1st fiber:
            #address the case when several fibers are spliced together
            first_fiber = find_first_fiber(network, fiber)
            first_fiber.att_in = padding - fiber_loss

def build_network(network, equipment):
    default_span_data = equipment['Spans']['default']
    max_length = int(default_span_data.max_length * UNITS[default_span_data.length_units])
    min_length = max(int(default_span_data.padding/0.2*1e3),50_000)
    bounds = range(min_length, max_length)
    target_length = max(min_length, 90_000)
    con_in = default_span_data.con_in
    con_out = default_span_data.con_out + default_span_data.EOL
    padding = default_span_data.padding

    fibers = [f for f in network.nodes() if isinstance(f, Fiber)]
    add_connector_loss(fibers, con_in, con_out)
    add_fiber_padding(network, fibers, padding)
    for fiber in fibers:
        split_fiber(network, fiber, bounds, target_length, equipment)        

    roadms = [r for r in network.nodes() if isinstance(r, Roadm)]
    for roadm in roadms:
        add_egress_amplifier(network, roadm, equipment)

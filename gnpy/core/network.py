#!/usr/bin/env python3

'''
gnpy.core.network
=================

This module contains functions for constructing networks of network elements.
'''


from networkx import DiGraph

from gnpy.core import elements
from gnpy.core.elements import Fiber, Edfa, Transceiver, Roadm, Fused
from gnpy.core.units import UNITS
from gnpy.core.equipment import get_eqpt_params


MAX_SPAN_LENGTH = 150_000
TARGET_SPAN_LENGTH = 100_000
MIN_SPAN_LENGTH = 30_000


def network_from_json(json_data):
    # NOTE|dutc: we could use the following, but it would tie our data format
    #            too closely to the graph library
    # from networkx import node_link_graph
    g = DiGraph()
    for el_config in json_data['elements']:
        g.add_node(getattr(elements, el_config['type'])(el_config))

    nodes = {k.uid: k for k in g.nodes()}

    for cx in json_data['connections']:
        from_node, to_node = cx['from_node'], cx['to_node']
        g.add_edge(nodes[from_node], nodes[to_node])

    return g

def calculate_new_length(fiber_length):
    result = (fiber_length, 1)
    if fiber_length > MAX_SPAN_LENGTH:
        n_spans = int(fiber_length // TARGET_SPAN_LENGTH)

        length1 = fiber_length / (n_spans+1)
        result1 = (length1, n_spans+1)
        delta1 = TARGET_SPAN_LENGTH-length1

        length2 = fiber_length / n_spans
        delta2 = length2-TARGET_SPAN_LENGTH
        result2 = (length2, n_spans)

        if length1<MIN_SPAN_LENGTH and length2<MAX_SPAN_LENGTH:
            result = result2
        elif length2>MAX_SPAN_LENGTH and length1>MIN_SPAN_LENGTH:
            result = result1
        else:
            result = result1 if delta1 < delta2 else result2

    return result

def split_fiber(network, fiber):
    new_length, n_spans = calculate_new_length(fiber.length)
    prev_node = fiber
    if n_spans > 1:
        next_nodes = [_ for _ in network.successors(fiber)]
        for next_node in next_nodes:
            network.remove_edge(fiber, next_node)

        new_params_length = new_length / UNITS[fiber.params.length_units]
        config = {'uid':fiber.uid, 'type': 'Fiber', 'metadata': fiber.__dict__['metadata'], \
            'params': fiber.__dict__['params']}
        fiber.uid = config['uid'] + '_1'
        fiber.length = new_length
        fiber.loss = fiber.loss_coef * fiber.length

        for i in range(2, n_spans+1):
            new_config = dict(config)
            new_config['uid'] = new_config['uid'] + '_' + str(i)
            new_config['params'].length = new_params_length
            new_node = Fiber(new_config)
            network.add_node(new_node)
            network.add_edge(prev_node, new_node)
            network = add_egress_amplifier(network, prev_node)
            prev_node = new_node

        for next_node in next_nodes:
            network.add_edge(prev_node, next_node)

    network = add_egress_amplifier(network, prev_node)
    return network

def select_edfa(ingress_span_loss):
    #TODO select amplifier in eqpt_library based on gain, NF and power requirement
    return "std_medium_gain"

def prev_fiber_node_generator(network, node):
    """fused spans interest:
    iterate over all predecessors while they are Fiber type"""
    prev_node = [n for n in network.predecessors(node)]
    if len(prev_node) == 1:
        #fibers or fused spans so there is only 1 predecessor
        if isinstance(prev_node[0], Fused) or isinstance(node, Fused):
            # yield and re-iterate
            yield prev_node[0]
            yield from prev_fiber_node_generator(network, prev_node[0])
        else:
            StopIteration

def span_loss(network, node):
    loss = node.loss if node.passive else 0
    return loss + sum(n.loss for n in prev_fiber_node_generator(network, node))

def add_egress_amplifier(network, node):
    next_nodes = [n for n in network.successors(node)
        if not (isinstance(n, Transceiver) or isinstance(n, Fused))]
        #no amplification for fused spans or TRX

    i = 1
    for next_node in next_nodes:
        if isinstance(next_node, Edfa):
            if next_node.operational.gain_target == 0:
                total_loss = span_loss(network, node)
                next_node.operational.gain_target = total_loss
        else:
            network.remove_edge(node, next_node)
            total_loss = span_loss(network, node)
            uid = f'Edfa{i}_{node.uid}'
            metadata = next_node.metadata
            operational = {'gain_target': total_loss, 'tilt_target': 0}
            edfa_variety_type = select_edfa(total_loss)
            config = {'uid': uid, 'type': 'Edfa', 'metadata': metadata,
                      'type_variety': edfa_variety_type, 'operational': operational}
            new_edfa = Edfa(config)
            network.add_node(new_edfa)
            network.add_edge(node,new_edfa)
            network.add_edge(new_edfa, next_node)
            i += 1

    return network

def build_network(network):
    fibers = [f for f in network.nodes() if isinstance(f, Fiber)]
    for fiber in fibers:
        network = split_fiber(network, fiber)

    roadms = [r for r in network.nodes() if isinstance(r, Roadm)]
    for roadm in roadms:
        add_egress_amplifier(network, roadm)


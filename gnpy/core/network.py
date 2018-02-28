#!/usr/bin/env python3

from networkx import DiGraph

from gnpy.core import elements
from gnpy.core.elements import Fiber, Edfa
from gnpy.core.units import UNITS

MAX_SPAN_LENGTH = 125000
TARGET_SPAN_LENGTH = 100000
MIN_SPAN_LENGTH = 75000

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
            if delta1 < delta2: 
                result = result1
            else:
                result = result2

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

def add_egress_amplifier(network, node):
    next_nodes = [n for n in network.successors(node) if not isinstance(n, Edfa)]
    i = 1
    for next_node in next_nodes:
        network.remove_edge(node, next_node)
        
        uid = 'Edfa' + str(i)+ '_' + str(node.uid)
        metadata = next_node.metadata
        operational = {'gain_target': node.loss, 'tilt_target': 0}
        edfa_config_json = 'edfa_config.json'
        config = {'uid':uid, 'type': 'Edfa', 'metadata': metadata, \
                    'config_from_json': edfa_config_json, 'operational': operational}
        new_edfa = Edfa(config)
        network.add_node(new_edfa)
        network.add_edge(node,new_edfa)
        network.add_edge(new_edfa, next_node)
        i +=1

    return network

def build_network(network):
    fibers = [f for f in network.nodes() if isinstance(f, Fiber)]
    for fiber in fibers:
        network = split_fiber(network, fiber)


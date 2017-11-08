#!/usr/bin/env python3
from networkx import DiGraph
from logging import getLogger

logger = getLogger('gnpy.core')

from . import elements

def network_from_json(json_data):
    # NOTE|dutc: we could use the following, but it would tie our data format
    #            too closely to the graph library
    # from networkx import node_link_graph
    g = DiGraph()

    nodes = {}
    for el in json_data['elements']:
        el = getattr(elements, el['type'])(el['id'], **el['metadata'])
        g.add_node(el)
        nodes[el.id] = el

    for cx in json_data['connections']:
        from_node, to_node = nodes[cx['from_node']], nodes[cx['to_node']]
        g.add_edge(from_node, to_node)

    return g

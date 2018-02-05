#!/usr/bin/env python3

from networkx import DiGraph

from gnpy.core import elements


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

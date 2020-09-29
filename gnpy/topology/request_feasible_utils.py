import networkx as nx
import numpy as np

from gnpy.core.elements import Roadm, Transceiver
from gnpy.core.info import create_input_spectral_information
from gnpy.core.utils import lin2db
from gnpy.topology.request import propagate


def are_paths_disjointed(path0, path1):
    """."""
    nodes_list0 = [node.name for node in path0 if isinstance(node, Roadm)][1:-1]
    nodes_list1 = [node.name for node in path1 if isinstance(node, Roadm)][1:-1]
    if set(nodes_list0) == set(nodes_list1):
        return False
    for node0 in nodes_list0:
        if node0 in nodes_list1:
            return False
    return True


def build_service_no_mode(mode, service):
    """."""
    service.OSNR = mode['OSNR']
    service.baud_rate = mode['baud_rate']
    service.bit_rate = mode['bit_rate']
    service.cost = mode['cost']
    service.format = mode['format']
    service.min_spacing = mode['min_spacing']
    service.roll_off = mode['roll_off']
    service.tsp_mode = mode['format']
    service.tx_osnr = mode['tx_osnr']
    return service


def check_service_modes(service0, service1):
    """."""
    if service0.baud_rate is None:
        return service0, service1
    else:
        return service1, service0


def find_feasible_path(equipment, network, service):
    """."""
    shortest_simple_paths = find_shortest_simple_paths(network, service)
    alternative_paths = []
    try:
        for path in shortest_simple_paths:
            if is_routing_node_in_path(path, service.nodes_list):
                propagated_path = find_propagated_path(equipment, path, service)
                if propagated_path:
                    return propagated_path
            else:
                if 'STRICT' not in service.loose_list:
                    alternative_paths.append(path)
        for path in alternative_paths:
            propagated_path = find_propagated_path(equipment, path, service)
            if propagated_path:
                return propagated_path
        return []
    except nx.NetworkXNoPath:
        return []


def find_feasible_paths_disjunction(equipment, network, service0, service1):
    """."""
    shortest_simple_paths0 = find_shortest_simple_paths(network, service0)
    shortest_simple_paths1 = find_shortest_simple_paths(network, service1)
    alternative_pairs = []
    alternative_path = {}
    try:
        for path0 in shortest_simple_paths0:
            for path1 in shortest_simple_paths1:
                if are_paths_disjointed(path0, path1):
                    if is_routing_node_in_path(path0, service0.nodes_list) and is_routing_node_in_path(path1, service1.nodes_list):
                        propagated_path0 = find_propagated_path(equipment, path0, service0)
                        propagated_path1 = find_propagated_path(equipment, path1, service1)
                        if propagated_path0 and propagated_path1:
                            return propagated_path0, propagated_path1
                    else:
                        if 'STRICT' not in service0.loose_list and 'STRICT' not in service1.loose_list:
                            alternative_path['0'] = path0
                            alternative_path['1'] = path1
                            alternative_pairs.append(alternative_path)
            shortest_simple_paths1 = find_shortest_simple_paths(network, service1)
        if alternative_pairs:
            for pair in alternative_pairs:
                propagated_path0 = find_propagated_path(equipment, pair['0'], service0)
                propagated_path1 = find_propagated_path(equipment, pair['1'], service1)
                if propagated_path0 and propagated_path1:
                    return propagated_path0, propagated_path1
        else:
            return [], []
    except nx.NetworkXNoPath:
        return [], []


def find_propagated_path(equipment, path, service):
    """."""
    propagate(path, service, equipment)
    snr01nm = round(np.mean(path[-1].snr + lin2db(service.baud_rate / 12.5e9)), 2)
    if snr01nm < service.OSNR:
        return []
    else:
        return path


def find_service_not_in_disjunction(disjunction_list, service_list):
    """."""
    service_in_disjunction = []
    for disjunction in disjunction_list:
        for service in disjunction.disjunctions_req:
            if service not in service_in_disjunction:
                service_in_disjunction.append(service)
    service_not_in_disjunction = []
    for service in service_list:
        if service.request_id not in service_in_disjunction:
            service_not_in_disjunction.append(service)
    return service_not_in_disjunction


def find_shortest_simple_paths(network, service):
    """."""
    source = next(node for node in network.nodes() if node.uid == service.source)
    target = next(node for node in network.nodes() if node.uid == service.destination)
    return nx.shortest_simple_paths(network, source, target)


def is_routing_node_in_path(simple_path, nodes_list):
    """."""
    if nodes_list:
        path_nodes_list = [node.name for node in simple_path if isinstance(node, Roadm) or isinstance(node, Transceiver)]
        i = 0
        for node in nodes_list:
            if node in path_nodes_list:
                if path_nodes_list.index(node) > i:
                    i = path_nodes_list.index(node)
                else:
                    return False
            else:
                return False
        return True
    else:
        return True
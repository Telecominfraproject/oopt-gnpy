from gnpy.core.elements import Roadm, Transceiver
from networkx.utils import pairwise
import networkx as nx


def find_all_paths(network, service):
    """This function receives a service request and returns a dictionary with the following attributes:
    - obg_paths: list of obligatory simple paths to follow (in case of no routing paths or STRICT constraint);
    - opt_paths: list of optional simple paths to follow (in case of LOOSE constraint).
    """
    all_paths = {}
    routing_paths = []
    source = next(node for node in network.nodes() if node.uid == service.source)
    destination = next(node for node in network.nodes() if node.uid == service.destination)
    #nx.shortest_simple_paths
    simple_paths_list = list(nx.all_simple_paths(network, source, destination, cutoff=80))
    if len(service.nodes_list) > 1:
        for path in simple_paths_list:
            path_nodes = [node.name for node in path if isinstance(node, Roadm) or isinstance(node, Transceiver)]
            if is_routing_node_in_path(path_nodes, service.nodes_list):
                routing_paths.append(path)
        all_paths['obg_paths'] = routing_paths
        if 'LOOSE' not in service.loose_list:
            all_paths['opt_paths'] = []
        else:
            all_paths['opt_paths'] = [path for path in simple_paths_list if path not in routing_paths]
    else:
        all_paths['obg_paths'] = simple_paths_list
        all_paths['opt_paths'] = []
    return all_paths


def is_routing_node_in_path(path_nodes, routing_nodes_list):
    """."""
    i = 0
    for node in routing_nodes_list:
        if node in path_nodes:
            if path_nodes.index(node) > i:
                i = path_nodes.index(node)
            else:
                return False
        else:
            return False
    return True


def find_disjointed_paths(service0, paths0_list, service1, paths1_list):
    """."""
    disjointed_paths = {}
    obg_paths = {}
    pair = {}
    i = 0
    for path0 in paths0_list['obg_paths']:
        for path1 in paths1_list['obg_paths']:
            path0_names = [node.name for node in path0 if isinstance(node, Roadm)]
            path1_names = [node.name for node in path1 if isinstance(node, Roadm)]
            if are_paths_disjointed(path0_names, path1_names):
                pair[service0] = path0
                pair[service1] = path1
                obg_paths[str(i)] = pair
                pair = {}
                i = i + 1
    disjointed_paths['obg_paths'] = obg_paths
    combinations_list = [['opt_paths', 'obg_paths'], ['obg_paths', 'opt_paths'], ['opt_paths', 'opt_paths']]
    opt_paths = {}
    pair = {}
    i = 0
    for combination in combinations_list:
        for path0 in paths0_list[combination[0]]:
            for path1 in paths1_list[combination[1]]:
                path0_names = [node.name for node in path0 if isinstance(node, Roadm)]
                path1_names = [node.name for node in path1 if isinstance(node, Roadm)]
                if are_paths_disjointed(path0_names, path1_names):
                    pair[service0] = path0
                    pair[service1] = path1
                    opt_paths[str(i)] = pair
                    pair = {}
                    i = i + 1
    disjointed_paths['opt_paths'] = opt_paths
    return disjointed_paths


def are_paths_disjointed(path0, path1):
    """."""
    pairs_path0_list = list(pairwise(path0))
    pairs_path1_list = list(pairwise(path1))
    for pair0 in pairs_path0_list:
        if pair0 in pairs_path1_list:
            return False
    pairs_path0_list_reversed = list(pairwise(path0[::-1]))
    for pair0 in pairs_path0_list_reversed:
        if pair0 in pairs_path1_list:
            return False
    return True


def find_paths_not_in_disjunction(disjunction_list, paths_per_service, service_list):
    """."""
    service_in_disjunction = find_service_not_in_disjunction(disjunction_list, service_list)
    for service in service_in_disjunction:
        del paths_per_service[service]
    paths_not_in_disjunction = paths_per_service
    return paths_not_in_disjunction


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
            service_not_in_disjunction.append(service.request_id)
    return service_not_in_disjunction
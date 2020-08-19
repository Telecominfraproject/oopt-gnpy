from gnpy.topology.request_all_utils import (find_all_paths,
                                             find_disjointed_paths,
                                             find_service_not_in_disjunction)


def find_all_feasible_paths(disjunction_list, equipment, network, service_list):
    """This function receives a list of service requests and returns a list of combinations of feasible paths."""
    # find all simple paths per service request (accounting for possible routing nodes and constraints)
    paths_per_service = find_paths_per_service(network, service_list)
    # deal with disjunction requests
    paths_per_disjunction = find_paths_per_disjunction(disjunction_list, paths_per_service, service_list)
    # find all feasible paths
    all_feasible_paths = {}
    service_not_in_disjunction = find_service_not_in_disjunction(disjunction_list, service_list)
    for service in service_not_in_disjunction:
        all_feasible_paths[service] = paths_per_service[service]
    for disjunction in disjunction_list:
        index = disjunction.disjunctions_req[0] + '-' + disjunction.disjunctions_req[1]
        all_feasible_paths[index] = paths_per_disjunction[disjunction.disjunction_id]
    return all_feasible_paths


def find_paths_per_service(network, service_list):
    """."""
    paths_per_service = {}
    for service in service_list:
        all_paths = find_all_paths(network, service)
        paths_per_service[service.request_id] = all_paths
    return paths_per_service


def find_paths_per_disjunction(disjunction_list, paths_per_service, service_list):
    """."""
    paths_per_disjunction = {}
    for disjunction in  disjunction_list:
        service0 = disjunction.disjunctions_req[0]
        paths0_list = paths_per_service[service0]
        service1 = disjunction.disjunctions_req[1]
        paths1_list = paths_per_service[service1]
        disjointed_paths = find_disjointed_paths(service0, paths0_list, service1, paths1_list)
        paths_per_disjunction[disjunction.disjunction_id] = disjointed_paths
    return paths_per_disjunction
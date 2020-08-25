from gnpy.topology.request_feasible_utils import (build_service_no_mode,
                                                  check_service_modes,
                                                  find_feasible_path,
                                                  find_feasible_paths_disjunction,
                                                  find_service_not_in_disjunction)

def find_feasible_paths(disjunction_list, equipment, network, service_list):
    """."""
    # find a feasible path for services not associated with a disjunction
    service_not_in_disjunction = find_service_not_in_disjunction(disjunction_list, service_list)
    feasible_paths = {}
    for service in service_not_in_disjunction:
        path_per_mode = {}
        if service.baud_rate is not None:
            path_per_mode[service.tsp_mode] = find_feasible_path(equipment, network, service)
            feasible_paths[service.request_id] = path_per_mode
        else:
            modes_list = equipment['Transceiver'][service.tsp].mode
            for mode in modes_list:
                service = build_service_no_mode(mode, service)
                path_per_mode[service.tsp_mode] = find_feasible_path(equipment, network, service)
            feasible_paths[service.request_id] = path_per_mode
    # find a feasible path for services associated with a disjunction
    for disjunction in disjunction_list:
        service0 = next(service for service in service_list if service.request_id == disjunction.disjunctions_req[0])
        service1 = next(service for service in service_list if service.request_id == disjunction.disjunctions_req[1])
        path_per_mode0 = {}
        path_per_mode1 = {}
        if service0.baud_rate is not None and service1.baud_rate is not None:
            path0, path1 = find_feasible_paths_disjunction(equipment, network, service0, service1)
            path_per_mode0[service0.tsp_mode] = path0
            path_per_mode1[service1.tsp_mode] = path1
            feasible_paths[service0.request_id] = path_per_mode0
            feasible_paths[service1.request_id] = path_per_mode1
        elif service0.baud_rate is None and service1.baud_rate is None:
            modes_list0 = equipment['Transceiver'][service0.tsp].mode
            modes_list1 = equipment['Transceiver'][service1.tsp].mode
            for mode0 in modes_list0:
                service0 = build_service_no_mode(mode0, service0)
                for mode1 in modes_list1:
                    service1 = build_service_no_mode(mode1, service1)
                    path0, path1 = find_feasible_paths_disjunction(equipment, network, service0, service1)
                    path_per_mode0[service0.tsp_mode + '|' + service1.tsp_mode] = path0
                    path_per_mode1[service0.tsp_mode + '|' + service1.tsp_mode] = path1
                    feasible_paths[service0.request_id] = path_per_mode0
                    feasible_paths[service1.request_id] = path_per_mode1
        else:
            service0, service1 = check_service_modes(service0, service1)
            modes_list0 = equipment['Transceiver'][service0.tsp].mode
            for mode0 in modes_list0:
                service0 = build_service_no_mode(mode0, service0)
                path0, path1 = find_feasible_paths_disjunction(equipment, network, service0, service1)
                path_per_mode0[service0.tsp_mode + '|' + service1.tsp_mode] = path0
                path_per_mode1[service0.tsp_mode + '|' + service1.tsp_mode] = path1
                feasible_paths[service0.request_id] = path_per_mode0
                feasible_paths[service1.request_id] = path_per_mode1
    #return final dictionary
    return feasible_paths
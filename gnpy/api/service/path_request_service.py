# -*- coding: utf-8 -*-
import logging

import gnpy.core.ansi_escapes as ansi_escapes
from gnpy.core.network import build_network
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.tools.json_io import requests_from_json, disjunctions_from_json
from gnpy.topology.request import (compute_path_dsjctn, requests_aggregation,
                                   correct_json_route_list,
                                   deduplicate_disjunctions, compute_path_with_disjunction)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum

_logger = logging.getLogger(__name__)

def path_requests_run(service, network, equipment):
    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    oms_list = build_oms_list(network, equipment)
    rqs = requests_from_json(service, equipment)

    # check that request ids are unique. Non unique ids, may
    # mess the computation: better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for item in list(set(all_ids)):
            all_ids.remove(item)
        msg = f'Requests id {all_ids} are not unique'
        _logger.critical(msg)
        raise ValueError('Requests id ' + all_ids + ' are not unique')
    rqs = correct_json_route_list(network, rqs)

    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(service)

    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = deduplicate_disjunctions(dsjn)

    rqs, dsjn = requests_aggregation(rqs, dsjn)
    # TODO export novel set of aggregated demands in a json file

    _logger.info(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}' + str(rqs))

    _logger.info(f'{ansi_escapes.blue}Computing all paths with constraints{ansi_escapes.reset}')
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)

    _logger.info(f'{ansi_escapes.blue}Propagating on selected path{ansi_escapes.reset}')
    propagatedpths, reversed_pths, reversed_propagatedpths = compute_path_with_disjunction(network, equipment, rqs,
                                                                                           pths)
    # Note that deepcopy used in compute_path_with_disjunction returns
    # a list of nodes which are not belonging to network (they are copies of the node objects).
    # so there can not be propagation on these nodes.

    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)
    return propagatedpths, reversed_propagatedpths, rqs

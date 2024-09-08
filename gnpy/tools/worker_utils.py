#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.worker_utils
=======================

Common code for CLI examples and API
'''
import logging
from copy import deepcopy
from typing import Union, List, Tuple
from numpy import linspace
from networkx import DiGraph

from gnpy.core.utils import automatic_nch, watt2dbm, dbm2watt, pretty_summary_print, per_label_average
from gnpy.core.equipment import trx_mode_params
from gnpy.core.network import add_missing_elements_in_network, design_network
from gnpy.core import exceptions
from gnpy.core.info import SpectralInformation
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum, OMS
from gnpy.topology.request import correct_json_route_list, deduplicate_disjunctions, requests_aggregation, \
    compute_path_dsjctn, compute_path_with_disjunction, ResultElement, PathRequest, Disjunction, \
    compute_constrained_path, propagate
from gnpy.tools.json_io import requests_from_json, disjunctions_from_json


logger = logging.getLogger(__name__)


def designed_network(equipment: dict, network: DiGraph, source: str = None, destination: str = None,
                     nodes_list: List[str] = None, loose_list: List[str] = None,
                     initial_spectrum: dict = None, no_insert_edfas: bool = False,
                     args_power: Union[str, float, int] = None,
                     service_req: PathRequest = None) -> Tuple[DiGraph, PathRequest, PathRequest]:
    """Build the reference channels based on inputs and design the network for this reference channel, and build the
    channel to be propagated for the single transmission script.

    Reference channel (target input power in spans, nb of channels, transceiver output power) is built using
    equipment['SI'] information. If indicated,  with target input power in spans is updated with args_power.
    Channel to be propagated is using the same channel reference, except if different settings are provided
    with service_req and initial_spectrum. The service to be propagated uses specified source, destination
    and list nodes_list of include nodes constraint except if the service_req is specified.

    Args:
    - equipment: a dictionary containing equipment information.
    - network: a directed graph representing the initial network.
    - no_insert_edfas: a boolean indicating whether to insert EDFAs in the network.
    - args_power: the power to be used for the network design.
    - service_req: the service request the user wants to propagate.
    - source: the source node for the channel to be propagated if no service_req is specified.
    - destination: the destination node for the channel to be propagated if no service_req is specified.
    - nodes_list: a list of nodes to be included ifor the channel to be propagated if no service_req is specified.
    - loose_list: a list of loose nodes to be included in the network design.
    - initial_spectrum: a dictionary representing the initial spectrum to propagate.

    Returns:
    - The designed network.
    - The channel to propagate.
    - The reference channel used for the design.
    """
    if loose_list is None:
        loose_list = []
    if nodes_list is None:
        nodes_list = []
    if not no_insert_edfas:
        add_missing_elements_in_network(network, equipment)

    if not nodes_list:
        if destination:
            nodes_list = [destination]
            loose_list = ['STRICT']
        else:
            nodes_list = []
            loose_list = []
    params = {
        'request_id': 'reference',
        'trx_type': '',
        'trx_mode': '',
        'source': source,
        'destination': destination,
        'bidir': False,
        'nodes_list': nodes_list,
        'loose_list': loose_list,
        'format': '',
        'path_bandwidth': 0,
        'effective_freq_slot': None,
        'nb_channel': automatic_nch(equipment['SI']['default'].f_min, equipment['SI']['default'].f_max,
                                    equipment['SI']['default'].spacing),
        'power': dbm2watt(equipment['SI']['default'].power_dbm),
        'tx_power': None
    }
    params['tx_power'] = dbm2watt(equipment['SI']['default'].power_dbm)
    if equipment['SI']['default'].tx_power_dbm is not None:
        # use SI tx_power if present
        params['tx_power'] = dbm2watt(equipment['SI']['default'].tx_power_dbm)
    trx_params = trx_mode_params(equipment)
    params.update(trx_params)

    # use args_power instead of si
    if args_power:
        params['power'] = dbm2watt(float(args_power))
        if equipment['SI']['default'].tx_power_dbm is None:
            params['tx_power'] = params['power']

    # use si as reference channel
    reference_channel = PathRequest(**params)
    # temporary till multiband design feat is available: do not design for L band
    reference_channel.nb_channel = min(params['nb_channel'], automatic_nch(191.2e12, 196.0e12, params['spacing']))

    if service_req:
        # use service_req as reference channel with si tx_power if service_req tx_power is None
        if service_req.tx_power is None:
            service_req.tx_power = params['tx_power']
        reference_channel = service_req

    design_network(reference_channel, network, equipment, set_connector_losses=True, verbose=True)

    if initial_spectrum:
        params['nb_channel'] = len(initial_spectrum)

    req = PathRequest(**params)
    if service_req:
        req = service_req

    req.initial_spectrum = initial_spectrum
    return network, req, reference_channel


def check_request_path_ids(rqs: List[PathRequest]):
    """check that request ids are unique. Non unique ids, may
    mess the computation: better to stop the computation
    """
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for item in list(set(all_ids)):
            all_ids.remove(item)
        msg = f'Requests id {all_ids} are not unique'
        logger.error(msg)
        raise ValueError(msg)


def planning(network: DiGraph, equipment: dict, data: dict, redesign: bool = False) \
        -> Tuple[List[OMS], list, list, List[PathRequest], List[Disjunction], List[ResultElement]]:
    """Run planning
    data contain the service dict from json
    redesign True means that network is redesign using each request as reference channel
    when False it means that the design is made once and successive propagation use the settings
    computed with this design.
    """
    oms_list = build_oms_list(network, equipment)
    rqs = requests_from_json(data, equipment)
    # check that request ids are unique.
    check_request_path_ids(rqs)
    rqs = correct_json_route_list(network, rqs)
    dsjn = disjunctions_from_json(data)
    logger.info('List of disjunctions:\n%s', dsjn)
    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = deduplicate_disjunctions(dsjn)
    logger.info('Aggregating similar requests')
    rqs, dsjn = requests_aggregation(rqs, dsjn)
    logger.info('The following services have been requested:\n%s', rqs)
    # logger.info('Computing all paths with constraints for request %s', optical_path_result_id)

    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    logger.info('Propagating on selected path')
    propagatedpths, reversed_pths, reversed_propagatedpths = \
        compute_path_with_disjunction(network, equipment, rqs, pths, redesign=redesign)
    # Note that deepcopy used in compute_path_with_disjunction returns
    # a list of nodes which are not belonging to network (they are copies of the node objects).
    # so there can not be propagation on these nodes.

    # Allowed user_policy are first_fit and 2partition
    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)
    for i, rq in enumerate(rqs):
        if hasattr(rq, 'OSNR') and rq.OSNR:
            rq.osnr_with_sys_margin = rq.OSNR + equipment["SI"]["default"].sys_margins

    # assumes that list of rqs and list of propgatedpths have same order
    result = [ResultElement(rq, pth, rpth) for rq, pth, rpth in zip(rqs, propagatedpths, reversed_propagatedpths)]
    return oms_list, propagatedpths, reversed_propagatedpths, rqs, dsjn, result


def transmission_simulation(equipment: dict, network: DiGraph, req: PathRequest, ref_req: PathRequest) \
        -> Tuple[list, List[list], List[Union[float, int]], SpectralInformation]:
    """Run simulation and returms the propagation result for each power sweep iteration.
    Args:
    - equipment: a dictionary containing equipment information.
    - network: network after being designed using ref_req. Any missing information (amp gain or delta_p) must have
    been filled using ref_req as reference channel previuos to this function.
    - req: channel to be propagated.
    - ref_req: the reference channel used for filling missing information in the network.
    In case of power sweep, network is redesigned using ref_req whose target input power in span is
    updated with the power step.

    Returns a tuple containing:
    - path: last propagated path. Power sweep is not possible with gain mode (as gain targets are used)
    - propagations: list of propagated path for each power iteration
    - powers_dbm: list of power used for the power sweep
    - infos: last propagated spectral information
    """
    power_mode = equipment['Span']['default'].power_mode
    logger.info('Power mode is set to %s=> it can be modified in eqpt_config.json - Span', power_mode)
    # initial network is designed using ref_req. that is that any missing information (amp gain or delta_p) is filled
    # using this ref_req.power, previous to any sweep requested later on.

    pref_ch_db = watt2dbm(ref_req.power)
    p_ch_db = watt2dbm(req.power)
    path = compute_constrained_path(network, req)
    power_range = [0]
    if power_mode:
        # power cannot be changed in gain mode
        try:
            p_start, p_stop, p_step = equipment['SI']['default'].power_range_db
            p_num = abs(int(round((p_stop - p_start) / p_step))) + 1 if p_step != 0 else 1
            power_range = list(linspace(p_start, p_stop, p_num))
        except TypeError as e:
            msg = 'invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]'
            logger.error(msg)
            raise exceptions.EquipmentConfigError(msg) from e

    logger.info('Now propagating between %s and %s', req.source, req.destination)

    propagations = []
    powers_dbm = []
    for dp_db in power_range:
        ref_req.power = dbm2watt(pref_ch_db + dp_db)
        req.power = dbm2watt(p_ch_db + dp_db)

        # Power sweep is made to evaluate different span input powers, so redesign is mandatory for each power,
        #  but no need to redesign if there are no power sweep
        if len(power_range) > 1:
            design_network(ref_req, network.subgraph(path), equipment, set_connector_losses=False, verbose=False)

        infos = propagate(path, req, equipment)
        propagations.append(deepcopy(path))
        powers_dbm.append(pref_ch_db + dp_db)
        logger.info('\nChannels propagating: (Input optical power deviation in span = '
                    + f'{pretty_summary_print(per_label_average(infos.delta_pdb_per_channel, infos.label))}dB,\n'
                    + '                       spacing = '
                    + f'{pretty_summary_print(per_label_average(infos.slot_width * 1e-9, infos.label))}GHz,\n'
                    + '                       transceiver output power = '
                    + f'{pretty_summary_print(per_label_average(watt2dbm(infos.tx_power), infos.label))}dBm,\n'
                    + f'                       nb_channels = {infos.number_of_channels})')
        if not power_mode:
            logger.info('\n\tPropagating using gain targets: Input optical power deviation in span ignored')
    return path, propagations, powers_dbm, infos

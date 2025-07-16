#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.service_sheet: XLS parser that can be called to create a JSON request file
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.tools.service_sheet
========================

XLS parser that can be called to create a JSON request file in accordance with
Yang model for requesting path computation.

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from logging import getLogger
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Generator
from networkx import DiGraph

from gnpy.core.utils import db2lin
from gnpy.core.exceptions import ServiceError
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fiber
from gnpy.tools.convert import corresp_names, corresp_next_node, all_rows, generic_open_workbook, get_sheet, \
    parse_row, parse_headers
from gnpy.tools.xls_utils import correct_cell_int_to_str, get_sheet_name, is_type_cell_empty


SERVICES_COLUMN = 12
SERVICE_LINE = 4


logger = getLogger(__name__)


class Request:
    """DATA class for a request.

    :params request_id (int): The unique identifier for the request.
    :params source (str): The source node for the communication.
    :params destination (str): The destination node for the communication.
    :params trx_type (str): The type of transmission for the communication.
    :params mode (str, optional): The mode of transmission. Defaults to None.
    :params spacing (float, optional): The spacing between channels. Defaults to None.
    :params power (float, optional): The power level for the communication. Defaults to None.
    :params nb_channel (int, optional): The number of channels required for the communication. Defaults to None.
    :params disjoint_from (str, optional): The node to be disjoint from. Defaults to ''.
    :params nodes_list (list, optional): The list of nodes involved in the communication. Defaults to None.
    :params is_loose (str, optional): Indicates if the communication is loose. Defaults to ''.
    :params path_bandwidth (float, optional): The bandwidth required for the communication. Defaults to None.
    """
    def __init__(self, **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
        """Updates the attributes of the node based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != '' and v is not None}
        for k, v in self.default_values.items():
            v = clean_kwargs.get(k, v)
            if k != 'is_loose':
                if k in ['request_id', 'trx_type', 'mode', 'disjoint_from']:
                    v = correct_cell_int_to_str(v)
                setattr(self, k, v)
            else:
                self.is_loose = v in ['', None, 'yes', 'Yes', 'YES']

    default_values = {
        'request_id': None,
        'source': None,
        'destination': None,
        'trx_type': None,
        'mode': None,
        'spacing': None,
        'power': None,
        'nb_channel': None,
        'disjoint_from': '',
        'nodes_list': '',
        'is_loose': None,
        'path_bandwidth': None
    }


class Element:
    """
    """
    def __init__(self, uid):
        self.uid = uid

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.uid == other.ui

    def __hash__(self):
        return hash((type(self), self.uid))


class Request_element(Element):
    """Class that generate the request in the json format

    :params request_param (Request): The request object containing the information for the element.
    :params equipment (dict): The equipment configuration for the communication.
    :params bidir (bool): Indicates if the communication is bidirectional.

    Attributes:
        request_id (str): The unique identifier for the request.
        source (str): The source node for the communication.
        destination (str): The destination node for the communication.
        srctpid (str): The source TP ID for the communication.
        dsttpid (str): The destination TP ID for the communication.
        bidir (bool): Indicates if the communication is bidirectional.
        trx_type (str): The type of transmission for the communication.
        mode (str): The mode of transmission for the communication.
        spacing (float): The spacing between channels for the communication.
        power (float): The power level for the communication.
        nb_channel (int): The number of channels required for the communication.
        disjoint_from (list): The list of nodes to be disjoint from.
        nodes_list (list): The list of nodes involved in the communication.
        loose (str): Indicates if the communication is loose or strict.
        path_bandwidth (float): The bandwidth required for the communication.
    """
    def __init__(self, request_param: Request, equipment: Dict, bidir: bool):
        """
        """
        super().__init__(uid=request_param.request_id)
        # request_id is str
        # excel has automatic number formatting that adds .0 on integer values
        # the next lines recover the pure int value, assuming this .0 is unwanted
        self.request_id = request_param.request_id
        self.source = f'trx {request_param.source}'
        self.destination = f'trx {request_param.destination}'
        # The automatic naming generated by excel parser requires that source and dest name
        # be a string starting with 'trx' : this is manually added here.
        self.srctpid = f'trx {request_param.source}'
        self.dsttpid = f'trx {request_param.destination}'
        self.bidir = bidir
        # test that trx_type belongs to eqpt_config.json
        # if not replace it with a default
        self.mode = None
        try:
            available_modes = [mode['format'] for mode in equipment['Transceiver'][request_param.trx_type].mode]
            self.trx_type = request_param.trx_type
            if request_param.mode not in [None, '']:
                if request_param.mode in available_modes:
                    self.mode = request_param.mode
                else:
                    msg = f'Request Id: {self.request_id} - could not find tsp : \'{request_param.trx_type}\' ' \
                          + f'with mode: \'{request_param.mode}\' in eqpt library \nComputation stopped.'
                    raise ServiceError(msg)
        except KeyError as e:
            msg = f'Request Id: {self.request_id} - could not find tsp : \'{request_param.trx_type}\' with mode: ' \
                  + f'\'{request_param.mode}\' in eqpt library \nComputation stopped.'
            raise ServiceError(msg) from e
        # excel input are in GHz and dBm
        if request_param.spacing:
            self.spacing = request_param.spacing * 1e9
        else:
            msg = f'Request {self.request_id} missing spacing: spacing is mandatory.\ncomputation stopped'
            raise ServiceError(msg)

        self.power = None
        if request_param.power is not None:
            self.power = db2lin(request_param.power) * 1e-3

        self.nb_channel = None
        if request_param.nb_channel is not None:
            self.nb_channel = int(request_param.nb_channel)

        self.disjoint_from = [n for n in request_param.disjoint_from.split(' | ') if request_param.disjoint_from]
        self.nodes_list = []
        if request_param.nodes_list:
            self.nodes_list = request_param.nodes_list.split(' | ')

        self.loose = 'LOOSE'
        if not request_param.is_loose:
            self.loose = 'STRICT'

        self.path_bandwidth = 0
        if request_param.path_bandwidth is not None:
            self.path_bandwidth = request_param.path_bandwidth * 1e9

    @property
    def pathrequest(self):
        """Creates json dictionnary for the request
        """
        # Default assumption for bidir is False
        req_dictionnary = {
            'request-id': self.request_id,
            'source': self.source,
            'destination': self.destination,
            'src-tp-id': self.srctpid,
            'dst-tp-id': self.dsttpid,
            'bidirectional': self.bidir,
            'path-constraints': {
                'te-bandwidth': {
                    'technology': 'flexi-grid',
                    'trx_type': self.trx_type,
                    'trx_mode': self.mode,
                    'effective-freq-slot': [{'N': None, 'M': None}],
                    'spacing': self.spacing,
                    'max-nb-of-channel': self.nb_channel,
                    'output-power': self.power
                }
            }
        }

        if self.nodes_list:
            req_dictionnary['explicit-route-objects'] = {}
            temp = {'route-object-include-exclude': [
                {
                    'index': self.nodes_list.index(node),
                    'explicit-route-usage': 'route-include-ero',
                    'num-unnum-hop': {
                        'node-id': f'{node}',
                        'link-tp-id': 'link-tp-id is not used',
                        'hop-type': f'{self.loose}',
                    }
                }
                for node in self.nodes_list]
            }
            req_dictionnary['explicit-route-objects'] = temp
        if self.path_bandwidth is not None:
            req_dictionnary['path-constraints']['te-bandwidth']['path_bandwidth'] = self.path_bandwidth

        return req_dictionnary

    @property
    def pathsync(self):
        """Creates json dictionnary for disjunction list (synchronization vector)
        """
        if self.disjoint_from:
            return {'synchronization-id': self.request_id,
                    'svec': {
                        'relaxable': 'false',
                        'disjointness': 'node link',
                        'request-id-number': [self.request_id] + list(self.disjoint_from)
                    }
                    }
        return None
        # TO-DO: avoid multiple entries with same synchronisation vectors

    @property
    def json(self):
        """Returns the json dictionnary for requests and for synchronisation vector
        """
        return self.pathrequest, self.pathsync


def read_service_sheet(
        input_filename: Path,
        eqpt: Dict,
        network: DiGraph,
        network_filename: Path = None,
        bidir: bool = False) -> Dict:
    """ converts a service sheet into a json structure
    """
    if network_filename is None:
        network_filename = input_filename
    service = parse_excel(input_filename)
    req = [Request_element(n, eqpt, bidir) for n in service]
    req = correct_xls_route_list(network_filename, network, req)
    # if there is no sync vector , do not write any synchronization
    synchro = [n.json[1] for n in req if n.json[1] is not None]
    data = {'path-request': [n.json[0] for n in req]}
    if synchro:
        data['synchronization'] = synchro
    return data


def parse_excel(input_filename: Path) -> List[Request]:
    """Open xls_file and reads 'Service' sheet
    Returns the list of services data in Request class
    """
    wb, is_xlsx = generic_open_workbook(input_filename)
    service_sheet = get_sheet(wb, 'Service', is_xlsx)
    services = list(parse_service_sheet(service_sheet, is_xlsx))
    return services


def parse_service_sheet(service_sheet, is_xlsx) -> Generator[Request, None, None]:
    """ reads each column according to authorized fieldnames. order is not important.
    """
    logger.debug('Validating headers on %r', get_sheet_name(service_sheet, is_xlsx))
    # add a test on field to enable the '' field case that arises when columns on the
    # right hand side are used as comments or drawing in the excel sheet
    authorized_fieldnames = {
        'route id': 'request_id', 'Source': 'source', 'Destination': 'destination',
        'TRX type': 'trx_type', 'Mode': 'mode', 'System: spacing': 'spacing',
        'System: input power (dBm)': 'power', 'System: nb of channels': 'nb_channel',
        'routing: disjoint from': 'disjoint_from', 'routing: path': 'nodes_list',
        'routing: is loose?': 'is_loose', 'path bandwidth': 'path_bandwidth'}
    header = parse_headers(service_sheet, is_xlsx, authorized_fieldnames, {}, SERVICE_LINE, (0, SERVICES_COLUMN))

    # create a service_fieldname independant from the excel column order
    # to be compatible with any version of the sheet
    # the following dictionnary records the excel field names and the corresponding parameter's name

    for row in all_rows(service_sheet, is_xlsx, start=5):
        if not is_type_cell_empty(row[0], is_xlsx):
            # Check required because openpyxl in read_only mode can return "ghost" rows at the end of the document
            # (ReadOnlyCell cells with no actual value but formatting information even for empty rows).
            yield Request(**parse_row(row[0:SERVICES_COLUMN], header))


def check_end_points(pathreq: Request_element, network: DiGraph):
    """Raise error if end point is not correct
    """
    transponders = [n.uid for n in network.nodes() if isinstance(n, Transceiver)]
    if pathreq.source not in transponders:
        msg = f'Request: {pathreq.request_id}: could not find' +\
            f' transponder source : {pathreq.source}.'
        logger.critical(msg)
        raise ServiceError(msg)
    if pathreq.destination not in transponders:
        msg = f'Request: {pathreq.request_id}: could not find' +\
            f' transponder destination: {pathreq.destination}.'
        logger.critical(msg)
        raise ServiceError(msg)


def find_node_sugestion(n_id, corresp_roadm, corresp_fused, corresp_ila, network):
    """
    """
    roadmtype = [n.uid for n in network.nodes() if isinstance(n, Roadm)]
    edfatype = [n.uid for n in network.nodes() if isinstance(n, Edfa)]
    # check that n_id is in the node list, if not find a correspondance name
    if n_id in roadmtype + edfatype:
        return [n_id]
    # checks first roadm, fused, and ila in this order, because ila automatic name
    # contains roadm names. If it is a fused node, next ila names might be correct
    # suggestions, especially if following fibers were splitted and ila names
    # created with the name of the fused node
    if n_id in corresp_roadm.keys():
        return corresp_roadm[n_id]
    if n_id in corresp_fused.keys():
        return corresp_fused[n_id] + corresp_ila[n_id]
    if n_id in corresp_ila.keys():
        return corresp_ila[n_id]
    return []


def correct_xls_route_list(network_filename: Path, network: DiGraph,
                           pathreqlist: List[Request_element]) -> List[Request_element]:
    """ prepares the format of route list of nodes to be consistant with nodes names:
        remove wrong names, find correct names for ila, roadm and fused if the entry was
        xls.
        if it was not xls, all names in list should be exact name in the network.
    """

    # first loads the base correspondance dict built with excel naming
    corresp_roadm, corresp_fused, corresp_ila = corresp_names(network_filename, network)
    # then correct dict names with names of the autodisign and find next_node name
    # according to xls naming
    corresp_ila, next_node = corresp_next_node(network, corresp_ila, corresp_roadm)
    # finally correct constraints based on these dict
    trxfibertype = [n.uid for n in network.nodes() if isinstance(n, (Transceiver, Fiber))]
    # TODO there is a problem of identification of fibers in case of parallel
    # fibers between two adjacent roadms so fiber constraint is not supported
    for pathreq in pathreqlist:
        # first check that source and dest are transceivers
        check_end_points(pathreq, network)
        # silently pop source and dest nodes from the list if they were added by the user as first
        # and last elem in the constraints respectively. Other positions must lead to an error
        # caught later on
        if pathreq.nodes_list and pathreq.source == pathreq.nodes_list[0]:
            pathreq.nodes_list.pop(0)
        if pathreq.nodes_list and pathreq.destination == pathreq.nodes_list[-1]:
            pathreq.nodes_list.pop(-1)
        # Then process user defined constraints with respect to automatic namings
        temp = deepcopy(pathreq)
        # This needs a temporary object since we may suppress/correct elements in the list
        # during the process
        for i, n_id in enumerate(temp.nodes_list):
            # n_id must not be a transceiver and must not be a fiber (non supported, user
            # can not enter fiber names in excel)
            if n_id not in trxfibertype:
                nodes_suggestion = find_node_sugestion(n_id, corresp_roadm, corresp_fused, corresp_ila, network)
                try:
                    if len(nodes_suggestion) > 1:
                        # if there is more than one suggestion, we need to choose the direction
                        # we rely on the next node provided by the user for this purpose
                        new_n = next(n for n in nodes_suggestion
                                     if n in next_node
                                     and next_node[n] in temp.nodes_list[i:] + [pathreq.destination]
                                     and next_node[n] not in temp.nodes_list[:i])
                    elif len(nodes_suggestion) == 1:
                        new_n = nodes_suggestion[0]
                    else:
                        if temp.loose == 'LOOSE':
                            # if no matching can be found in the network just ignore this constraint
                            # if it is a loose constraint
                            # warns the user that this node is not part of the topology
                            msg = f'{pathreq.request_id}: Invalid node specified:\n\t\'{n_id}\'' \
                                + ', could not use it as constraint, skipped!'
                            print(msg)
                            logger.info(msg)
                            pathreq.nodes_list.remove(n_id)
                            continue
                        msg = f'{pathreq.request_id}: Could not find node:\n\t\'{n_id}\' in network' \
                            + ' topology. Strict constraint can not be applied.'
                        raise ServiceError(msg)
                    if new_n != n_id:
                        # warns the user when the correct name is used only in verbose mode,
                        # eg 'a' is a roadm and correct name is 'roadm a' or when there was
                        # too much ambiguity, 'b' is an ila, its name can be:
                        # "east edfa in b to c", or "west edfa in b to a" if next node is c or
                        # "west edfa in b to c", or "east edfa in b to a" if next node is a
                        msg = f'{pathreq.request_id}: Invalid route node specified:' \
                            + f'\n\t\'{n_id}\', replaced with \'{new_n}\''
                        logger.info(msg)
                        pathreq.nodes_list[pathreq.nodes_list.index(n_id)] = new_n
                except StopIteration:
                    # shall not come in this case, unless requested direction does not exist
                    msg = f'{pathreq.request_id}: Invalid route specified {n_id}: could' \
                        + ' not decide on direction, skipped!.\nPlease add a valid' \
                        + ' direction in constraints (next neighbour node)'
                    logger.info(msg)
                    pathreq.nodes_list.remove(n_id)
            else:
                if temp.loose == 'LOOSE':
                    msg = f'{pathreq.request_id}: Invalid route node specified:\n\t\'{n_id}\'' \
                        + ' type is not supported as constraint with xls network input, skipped!'
                    logger.warning(msg)
                    pathreq.nodes_list.remove(n_id)
                else:
                    msg = f'{pathreq.request_id}: Invalid route node specified \n\t\'{n_id}\'' \
                        + ' type is not supported as constraint with xls network input,' \
                        + ', Strict constraint can not be applied.'
                    raise ServiceError(msg)
    return pathreqlist

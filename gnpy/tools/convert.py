#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.convert: utilities for converting between XLS and JSON
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.tools.convert
==================

This module contains utilities for converting between XLS and JSON.

The input XLS file must contain sheets named "Nodes" and "Links".
It may optionally contain a sheet named "Eqpt".

In the "Nodes" sheet, only the "City" column is mandatory. The column "Type"
can be determined automatically given the topology (e.g., if degree 2, ILA;
otherwise, ROADM.) Incorrectly specified types (e.g., ILA for node of
degree ≠ 2) will be automatically corrected.

In the "Links" sheet, only the first three columns ("Node A", "Node Z" and
"east Distance (km)") are mandatory.  Missing "west" information is copied from
the "east" information so that it is possible to input undirected data.
"""

from logging import getLogger
from argparse import ArgumentParser
from collections import namedtuple, Counter, defaultdict
from itertools import chain
from json import dumps
from pathlib import Path
from copy import copy
from typing import Generator, Tuple, List, Dict, DefaultDict
from xlrd import open_workbook
from xlrd.sheet import Sheet
from xlrd.biffh import XLRDError
from networkx import DiGraph

from gnpy.core.utils import silent_remove, transform_data, convert_pmd_lineic
from gnpy.core.exceptions import NetworkTopologyError
from gnpy.core.elements import Edfa, Fused, Fiber


_logger = getLogger(__name__)


def all_rows(sh: Sheet, start: int = 0) -> Generator[list, None, None]:
    """Returns all rows of the xls(x) sheet starting from start row.

    :param sh: The sheet object from which to retrieve rows.
    :type sh: xlrd.sheet.Sheet
    :param start: The starting row index (default is 0).
    :type sart: int
    :return: A generator yielding all rows from the specified starting index.
    :rtype: Generator[list, None, None]
    """
    return (sh.row(x) for x in range(start, sh.nrows))


class Node:
    """Node data class representing a network node.

    :ivar city: The city where the node is located.
    :vartype city: str
    :ivar state: The state where the node is located.
    :vartype state: str
    :ivar country: The country where the node is located.
    :vartype country: str
    :ivar region: The region where the node is located.
    :vartype region: str
    :ivar latitude: The latitude of the node's location.
    :vartype latitude: float
    :ivar longitude: The longitude of the node's location.
    :vartype longitude: float
    :ivar node_type: The type of the node (e.g., ILA, ROADM).
    :vartype node_type: str
    :ivar booster_restriction: Restrictions on booster amplifiers.
    :vartype booster_restriction: str
    :ivar preamp_restriction: Restrictions on preamplifiers.
    :vartype preamp_restriction: str
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
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)

    default_values = {
        'city': '',
        'state': '',
        'country': '',
        'region': '',
        'latitude': 0,
        'longitude': 0,
        'node_type': 'ILA',
        'booster_restriction': '',
        'preamp_restriction': ''
    }


class Link:
    """Link data class representing a connection between nodes.

    :ivar from_city: The city where the link starts.
    :vartype from_city: str
    :ivar to_city: The city where the link ends.
    :vartype to_city: str
    :ivar east_distance: The distance of the link in the east direction.
    :vartype east_distance: float
    :ivar east_fiber: The type of fiber used in the east direction.
    :vartype east_fiber: str
    :ivar east_lineic: The linear attenuation in the east direction.
    :vartype east_lineic: float
    :ivar east_con_in: Connection input in the east direction.
    :vartype east_con_in: str
    :ivar east_con_out: Connection output in the east direction.
    :vartype east_con_out: str
    :ivar east_pmd: Polarization mode dispersion in the east direction.
    :vartype east_pmd: float
    :ivar east_cable: The cable identifier in the east direction.
    :vartype east_cable: str
    :ivar distance_units: The units of distance (default is 'km').
    :vartype distance_units: str
    """

    def __init__(self, **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)
        self.distance_units = 'km'

    def update_attr(self, kwargs):
        """Updates the attributes of the link based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)
            k = 'west' + k.rsplit('east', maxsplit=1)[-1]
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)

    def __eq__(self, link):
        """Checks if two links are equivalent (same or reversed).
        Parrallel links are not handled correctly yet.

        :param link: The link to compare with.
        :return: True if the links are equivalent, False otherwise.
        """
        # Disable all the no-member violations in this function
        # pylint: disable=E1101
        return (self.from_city == link.from_city and self.to_city == link.to_city) \
            or (self.from_city == link.to_city and self.to_city == link.from_city)

    default_values = {
        'from_city': '',
        'to_city': '',
        'east_distance': 80,
        'east_fiber': 'SSMF',
        'east_lineic': 0.2,
        'east_con_in': None,
        'east_con_out': None,
        'east_pmd': None,
        'east_cable': ''
    }


class Eqpt:
    """Equipment data class representing amplifiers or other equipment.

    :ivar from_city: The city where the equipment is located.
    :vartype from_city: str
    :ivar to_city: The city where the equipment connects to.
    :vartype to_city: str
    :ivar east_amp_type: The type of amplifier in the east direction.
    :vartype east_amp_type: str
    :ivar east_amp_gain: The gain of the amplifier in the east direction.
    :vartype east_amp_gain: float
    :ivar east_amp_dp: The delta power of the amplifier in the east direction.
    :vartype east_amp_dp: float
    :ivar east_tilt_vs_wavelength: Tilt of the amplifier versus wavelength in the east direction.
    :vartype east_tilt_vs_wavelength: float
    :ivar east_att_out: Output attenuation in the east direction.
    :vartype east_att_out: float
    :ivar east_att_in: Input attenuation in the east direction.
    :vartype east_att_in: float
    """
    def __init__(self, **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
        """Updates the attributes of the equipment based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            v_east = clean_kwargs.get(k, v)
            setattr(self, k, v_east)
            k = 'west' + k.rsplit('east', maxsplit=1)[-1]
            v_west = clean_kwargs.get(k, v)
            setattr(self, k, v_west)

    default_values = {
        'from_city': '',
        'to_city': '',
        'east_amp_type': '',
        'east_amp_gain': None,
        'east_amp_dp': None,
        'east_tilt_vs_wavelength': None,
        'east_att_out': None,
        'east_att_in': 0
    }


class Roadm:
    """ROADM data class representing a reconfigurable optical add-drop multiplexer.

    :ivar from_node: The starting node of the ROADM.
    :vartype from_node: str
    :ivar to_node: The ending node of the ROADM.
    :vartype to_node: str
    :ivar target_pch_out_db: Target output power per channel in dBm.
    :vartype target_pch_out_db: float
    :ivar type_variety: The type variety of the ROADM.
    :vartype type_variety: str
    :ivar from_degrees: Degrees from the starting node.
    :vartype from_degrees: str
    :ivar impairment_ids: Impairment identifiers associated with the ROADM.
    :vartype impairment_ids: str
    """
    def __init__(self, **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
        """Updates the attributes of the ROADM based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        :type kwargs: dict
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)

    default_values = {'from_node': '',
                      'to_node': '',
                      'target_pch_out_db': None,
                      'type_variety': None,
                      'from_degrees': None,
                      'impairment_ids': None
                      }


def read_header(my_sheet: Sheet, line: int, slice_: Tuple[int, int]) -> List[namedtuple]:
    """Return the list of headers in a specified range.

    header_i = [(header, header_column_index), ...]
    in a {line, slice1_x, slice_y} range

    :param my_sheet: The sheet object from which to read headers.
    :type my_sheet: xlrd.sheet.Sheet
    :param line: The row index to read headers from.
    :type line: int
    :param slice_: A tuple specifying the start and end column indices.
    :type slice_: Tuple[int, int]
    :return: A list of namedtuples containing headers and their column indices.
    :rtype: List[namedtuple]
    """
    param_header = namedtuple('param_header', 'header colindex')
    try:
        header = [x.value.strip() for x in my_sheet.row_slice(line, slice_[0], slice_[1])]
        header_i = [param_header(header, i + slice_[0]) for i, header in enumerate(header) if header != '']
    except (AttributeError, IndexError):
        header_i = []
    if header_i != [] and header_i[-1].colindex != slice_[1]:
        header_i.append(param_header('', slice_[1]))
    return header_i


def read_slice(my_sheet: Sheet, line: int, slice_: Tuple[int, int], header: str) -> Tuple[int, int]:
    """return the slice range of a given header
    in a defined range {line, slice_x, slice_y}

    :param my_sheet: The sheet object from which to read the header.
    :type my_sheet: xlrd.sheet.Sheet
    :param line: The row index to read from.
    :type line: int
    :param slice_: A tuple specifying the start and end column indices.
    :type slice_: Tuple[int, int]
    :param header: The header name to search for.
    :return: A tuple representing the start and end indices of the slice.
    :rtype: Tuple[int, int]
    """
    header_i = read_header(my_sheet, line, slice_)
    slice_range = (-1, -1)
    if header_i != []:
        try:
            slice_range = next((h.colindex, header_i[i + 1].colindex)
                               for i, h in enumerate(header_i) if header in h.header)
        except StopIteration:
            pass
    return slice_range


def parse_headers(my_sheet: Sheet, input_headers_dict: Dict, headers: Dict[int, str],
                  start_line: int, slice_in: Tuple[int, int]) -> Dict[int, str]:
    """return a dict of header_slice

    - key = column index
    - value = header name

    :param my_sheet: The sheet object from which to read headers.
    :type my_sheet: xlrd.sheet.Sheet
    :param input_headers_dict: A dictionary mapping expected headers to internal names.
    :type input_headers_dict: dict
    :param headers: A dictionary to store the header slices.
    :type headers: Dict[int, str]
    :param start_line: The starting line to search for headers.
    :type start_line: int
    :param slice_in: A tuple specifying the start and end column indices.
    :type slice_in: Tuple[int, int]
    :return: A dictionary mapping column indices to header names.
    :rtype: Dict[int, str]
    """
    for h0 in input_headers_dict:
        slice_out = read_slice(my_sheet, start_line, slice_in, h0)
        iteration = 1
        while slice_out == (-1, -1) and iteration < 10:
            # try next lines
            slice_out = read_slice(my_sheet, start_line + iteration, slice_in, h0)
            iteration += 1
        if slice_out == (-1, -1):
            msg = f'missing header {h0}'
            if h0 in ('east', 'Node A', 'Node Z', 'City'):
                raise NetworkTopologyError(msg)
            _logger.warning(msg)
        elif not isinstance(input_headers_dict[h0], dict):
            headers[slice_out[0]] = input_headers_dict[h0]
        else:
            headers = parse_headers(my_sheet, input_headers_dict[h0], headers, start_line + 1, slice_out)
    if headers == {}:
        msg = 'CRITICAL ERROR: could not find any header to read _ ABORT'
        raise NetworkTopologyError(msg)
    return headers


def parse_row(row, headers):
    """Parse a row of data into a dictionary based on headers.

    :param row: The row object to parse.
    :param headers: A dictionary mapping header names to column indices.
    :return: A dictionary mapping header names to their corresponding values in the row.
    """
    return {f: r.value for f, r in
            zip(list(headers.values()), [row[i] for i in headers])}


def parse_sheet(my_sheet: Sheet, input_headers_dict: Dict, header_line: int,
                start_line: int, column: int) -> Generator[Dict[str, str], None, None]:
    """Parse a sheet and yield rows as dictionaries.

    :param my_sheet: The sheet object to parse.
    :type my_sheet: xlrd.sheet.Sheet
    :param input_headers_dict: A dictionary mapping expected headers to internal names.
    :type input_headers_dict: dict
    :param header_line: The line number where headers are located.
    :type header_line: int
    :param start_line: The starting line number for data rows.
    :type start_line: int
    :param column: The number of columns to read.
    :type column: int
    :return: A generator yielding parsed rows as dictionaries.
    """
    headers = parse_headers(my_sheet, input_headers_dict, {}, header_line, (0, column))
    for row in all_rows(my_sheet, start=start_line):
        yield parse_row(row[0: column], headers)


def _format_items(items: List[str]):
    """Format a list of items into a string.

    :param items: A list of items to format.
    :type items: List[str]
    :return: A formatted string with each item on a new line.
    :rtype: str
    """
    return '\n'.join(f' - {item}' for item in items)


def sanity_check(nodes: List[Node], links: List[Link],
                 nodes_by_city: Dict[str, Node], links_by_city: DefaultDict[str, List[Link]],
                 eqpts_by_city: DefaultDict[str, List[Eqpt]]) -> Tuple[List[Node], List[Link]]:
    """Perform sanity checks on nodes and links. Raise correct issues if xls(x) is not correct,
    Correct type to ROADM if more tha 2-degrees, checks duplicate links, unreferenced nodes in links,
    in eqpts, unreferenced link in eqpts, duplicate items

    :param nodes: A list of Node objects.
    :type nodes: List[Node]
    :param links: A list of Link objects.
    :type links: List[Link]
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, Node]
    :param links_by_city: A defaultdict mapping city names to lists of Link objects.
    :type links_by_city: DefaultDict[str, List[Link]]
    :param eqpts_by_city: A defaultdict mapping city names to lists of Eqpt objects.
    :type eqpts_by_city: DefaultDict[str, List[Eqpt]]
    :return: A tuple containing the validated lists of nodes and links.
    :rtype: Tuple[List[Node], List[Link]]
    :raises NetworkTopologyError: If any issues are found during validation.
    """
    duplicate_links = []
    for l1 in links:
        for l2 in links:
            if l1 is not l2 and l1 == l2 and l2 not in duplicate_links:
                _logger.warning(f'\nWARNING\n \
                    link {l1.from_city}-{l1.to_city} is duplicate \
                    \nthe 1st duplicate link will be removed but you should check Links sheet input')
                duplicate_links.append(l1)

    if duplicate_links:
        msg = 'XLS error: ' \
              + f'links {_format_items([(d.from_city, d.to_city) for d in duplicate_links])} are duplicate'
        raise NetworkTopologyError(msg)
    unreferenced_nodes = [n for n in nodes_by_city if n not in links_by_city]
    if unreferenced_nodes:
        msg = 'XLS error: The following nodes are not ' \
              + 'referenced from the Links sheet. ' \
              + 'If unused, remove them from the Nodes sheet:\n' \
              + _format_items(unreferenced_nodes)
        raise NetworkTopologyError(msg)
    # no need to check "Links" for invalid nodes because that's already in parse_excel()
    wrong_eqpt_from = [n for n in eqpts_by_city if n not in nodes_by_city]
    wrong_eqpt_to = [n.to_city for destinations in eqpts_by_city.values()
                     for n in destinations if n.to_city not in nodes_by_city]
    wrong_eqpt = wrong_eqpt_from + wrong_eqpt_to
    if wrong_eqpt:
        msg = 'XLS error: ' \
              + 'The Eqpt sheet refers to nodes that ' \
              + 'are not defined in the Nodes sheet:\n'\
              + _format_items(wrong_eqpt)
        raise NetworkTopologyError(msg)
    # Now check links that are not listed in Links sheet, and duplicates
    bad_eqpt = []
    possible_links = [f'{e.from_city}|{e.to_city}' for e in links] + [f'{e.to_city}|{e.from_city}' for e in links]
    possible_eqpt = []
    duplicate_eqpt = []
    duplicate_ila = []
    for city, eqpts in eqpts_by_city.items():
        for eqpt in eqpts:
            # Check that each node_A-node_Z exists in links
            nodea_nodez = f'{eqpt.from_city}|{eqpt.to_city}'
            nodez_nodea = f'{eqpt.to_city}|{eqpt.from_city}'
            if nodea_nodez not in possible_links \
                    or nodez_nodea not in possible_links:
                bad_eqpt.append([eqpt.from_city, eqpt.to_city])
            else:
                # Check that there are no duplicate lines in the Eqpt sheet
                if nodea_nodez in possible_eqpt:
                    duplicate_eqpt.append([eqpt.from_city, eqpt.to_city])
                else:
                    possible_eqpt.append(nodea_nodez)
            # check that there are no two lines defining an ILA with different directions
        if nodes_by_city[city].node_type == 'ILA' and len(eqpts) > 1:
            duplicate_ila.append(city)
    if bad_eqpt:
        msg = 'XLS error: ' \
              + 'The Eqpt sheet references links that ' \
              + 'are not defined in the Links sheet:\n' \
              + _format_items(f'{item[0]} -> {item[1]}' for item in bad_eqpt)
        raise NetworkTopologyError(msg)
    if duplicate_eqpt:
        msg = 'XLS error: Duplicate lines in Eqpt sheet:' \
              + _format_items(f'{item[0]} -> {item[1]}' for item in duplicate_eqpt)
        raise NetworkTopologyError(msg)
    if duplicate_ila:
        msg = 'XLS error: Duplicate ILA eqpt definition in Eqpt sheet:' \
              + _format_items(duplicate_ila)
        raise NetworkTopologyError(msg)

    for city, link in links_by_city.items():
        if nodes_by_city[city].node_type.lower() == 'ila' and len(link) != 2:
            # wrong input: ILA sites can only be Degree 2
            # => correct to make it a ROADM and remove entry in links_by_city
            _logger.warning(f'invalid node type ({nodes_by_city[city].node_type}) '
                            + f'specified in {city}, replaced by ROADM')
            nodes_by_city[city].node_type = 'ROADM'
            for n in nodes:
                if n.city == city:
                    n.node_type = 'ROADM'
    return nodes, links


def create_roadm_element(node: Node, roadms_by_city: DefaultDict[str, List[Roadm]]) -> Dict:
    """Create the json element for a roadm node, including the different cases:

        - if there are restrictions
        - if there are per degree target power defined on a direction

    direction is defined by the booster name, so that booster must also be created in eqpt sheet
    if the direction is defined in roadm.

    :param node: The Node object representing the ROADM.
    :type node: Node
    :param roadms_by_city: A dictionary mapping city names to lists of ROADM objects.
    :type roadms_by_city: DefaultDict[str, List[Roadm]]
    :return: A dictionary representing the ROADM element in JSON format.
    :rtype: Dict
    """
    roadm = {'uid': f'roadm {node.city}'}
    if node.preamp_restriction != '' or node.booster_restriction != '':
        roadm['params'] = {
            'restrictions': {
                'preamp_variety_list': silent_remove(node.preamp_restriction.split(' | '), ''),
                'booster_variety_list': silent_remove(node.booster_restriction.split(' | '), '')}
        }
    if node.city in roadms_by_city.keys():
        if 'params' not in roadm:
            roadm['params'] = {}
        roadm['params']['per_degree_pch_out_db'] = {}
        for elem in roadms_by_city[node.city]:
            to_node = f'east edfa in {node.city} to {elem.to_node}'
            if elem.target_pch_out_db is not None:
                roadm['params']['per_degree_pch_out_db'][to_node] = elem.target_pch_out_db
            if elem.from_degrees is not None and elem.impairment_ids is not None:
                # only set per degree impairment if there is an entry (reduce verbose)
                if roadm['params'].get('per_degree_impairments') is None:
                    roadm['params']['per_degree_impairments'] = []
                fromdegrees = elem.from_degrees.split(' | ')
                impairment_ids = transform_data(elem.impairment_ids)
                if len(fromdegrees) != len(impairment_ids):
                    msg = f'Roadm {node.city} per degree impairment id do not match with from degree definition'
                    raise NetworkTopologyError(msg)
                for from_degree, impairment_id in zip(fromdegrees, impairment_ids):
                    from_node = f'west edfa in {node.city} to {from_degree}'
                    roadm['params']['per_degree_impairments'].append({'from_degree': from_node,
                                                                      'to_degree': to_node,
                                                                      'impairment_id': impairment_id})
            if elem.type_variety is not None:
                roadm['type_variety'] = elem.type_variety
    roadm['metadata'] = {'location': {'city':      node.city,      # noqa: E241
                                      'region':    node.region,    # noqa: E241
                                      'latitude':  node.latitude,  # noqa: E241
                                      'longitude': node.longitude}}
    roadm['type'] = 'Roadm'
    return roadm


def create_east_eqpt_element(node: Node, nodes_by_city: Dict[str, Node]) -> dict:
    """Create amplifiers json elements for the east direction.
    this includes the case where the case of a fused element defined instead of an
    ILA in eqpt sheet.

    :param node: The Node object representing the equipment.
    :type node: Node
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, Node]
    :return: A dictionary representing the east equipment element in JSON format.
    :rtype: dict
    """
    eqpt = {'uid': f'east edfa in {node.from_city} to {node.to_city}',
            'metadata': {'location': {'city':      nodes_by_city[node.from_city].city,      # noqa: E241
                                      'region':    nodes_by_city[node.from_city].region,    # noqa: E241
                                      'latitude':  nodes_by_city[node.from_city].latitude,  # noqa: E241
                                      'longitude': nodes_by_city[node.from_city].longitude}}}
    if node.east_amp_type.lower() != '' and node.east_amp_type.lower() != 'fused':
        eqpt['type'] = 'Edfa'
        eqpt['type_variety'] = f'{node.east_amp_type}'
        eqpt['operational'] = {'gain_target': node.east_amp_gain,
                               'delta_p':     node.east_amp_dp,   # noqa: E241
                               'tilt_target': node.east_tilt_vs_wavelength,
                               'out_voa':     node.east_att_out,  # noqa: E241
                               'in_voa':      node.east_att_in}   # noqa: E241
    elif node.east_amp_type.lower() == '':
        eqpt['type'] = 'Edfa'
        eqpt['operational'] = {'gain_target': node.east_amp_gain,
                               'delta_p':     node.east_amp_dp,   # noqa: E241
                               'tilt_target': node.east_tilt_vs_wavelength,
                               'out_voa':     node.east_att_out,  # noqa: E241
                               'in_voa':      node.east_att_in}   # noqa: E241
    elif node.east_amp_type.lower() == 'fused':
        # fused edfa variety is a hack to indicate that there should not be
        # booster amplifier out the roadm.
        # If user specifies ILA in Nodes sheet and fused in Eqpt sheet, then assumes that
        # this is a fused nodes.
        eqpt['type'] = 'Fused'
        eqpt['params'] = {'loss': 0}
    return eqpt


def create_west_eqpt_element(node: Node, nodes_by_city: Dict[str, Node]) -> dict:
    """Create amplifiers json elements for the west direction.
    this includes the case where the case of a fused element defined instead of an
    ILA in eqpt sheet.

    :param node: The Node object representing the equipment.
    :type node: Node
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, Node]
    :return: A dictionary representing the west equipment element in JSON format.
    :rtype: dict
    """
    eqpt = {'uid': f'west edfa in {node.from_city} to {node.to_city}',
            'metadata': {'location': {'city':      nodes_by_city[node.from_city].city,      # noqa: E241
                                      'region':    nodes_by_city[node.from_city].region,    # noqa: E241
                                      'latitude':  nodes_by_city[node.from_city].latitude,  # noqa: E241
                                      'longitude': nodes_by_city[node.from_city].longitude}},
            'type': 'Edfa'}
    if node.west_amp_type.lower() != '' and node.west_amp_type.lower() != 'fused':
        eqpt['type_variety'] = f'{node.west_amp_type}'
        eqpt['operational'] = {'gain_target': node.west_amp_gain,
                               'delta_p':     node.west_amp_dp,    # noqa: E241
                               'tilt_target': node.west_tilt_vs_wavelength,
                               'out_voa':     node.west_att_out,   # noqa: E241
                               'in_voa':      node.west_att_in}    # noqa: E241
    elif node.west_amp_type.lower() == '':
        eqpt['operational'] = {'gain_target': node.west_amp_gain,
                               'delta_p':     node.west_amp_dp,    # noqa: E241
                               'tilt_target': node.west_tilt_vs_wavelength,
                               'out_voa':     node.west_att_out,   # noqa: E241
                               'in_voa':      node.west_att_in}    # noqa: E241
    elif node.west_amp_type.lower() == 'fused':
        eqpt['type'] = 'Fused'
        eqpt['params'] = {'loss': 0}
    return eqpt


def create_east_fiber_element(fiber: Node, nodes_by_city: Dict[str, Node]) -> Dict:
    """Create fibers json elements for the east direction.

    :param fiber: The Node object representing the equipment.
    :type fiber: Node
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, Node]
    :return: A dictionary representing the west equipment element in JSON format.
    :rtype: Dict
    """
    fiber_dict = {
        'uid': f'fiber ({fiber.from_city} \u2192 {fiber.to_city})-{fiber.east_cable}',
        'metadata': {'location': midpoint(nodes_by_city[fiber.from_city],
                                          nodes_by_city[fiber.to_city])},
        'type': 'Fiber',
        'type_variety': fiber.east_fiber,
        'params': {
            'length': round(fiber.east_distance, 3),
            'length_units': fiber.distance_units,
            'loss_coef': fiber.east_lineic,
            'con_in': fiber.east_con_in,
            'con_out': fiber.east_con_out
        }
    }
    if fiber.east_pmd:
        fiber_dict['params']['pmd_coef'] = convert_pmd_lineic(fiber.east_pmd, fiber.east_distance, fiber.distance_units)
    return fiber_dict


def create_west_fiber_element(fiber: Node, nodes_by_city: Dict[str, Node]) -> Dict:
    """Create fibers json elements for the west direction.

    :param fiber: The Node object representing the equipment.
    :type fiber: Node
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, Node]
    :return: A dictionary representing the west equipment element in JSON format.
    :rtype: Dict
    """
    fiber_dict = {
        'uid': f'fiber ({fiber.to_city} \u2192 {fiber.from_city})-{fiber.west_cable}',
        'metadata': {'location': midpoint(nodes_by_city[fiber.from_city],
                                          nodes_by_city[fiber.to_city])},
        'type': 'Fiber',
        'type_variety': fiber.west_fiber,
        'params': {'length': round(fiber.west_distance, 3),
                   'length_units': fiber.distance_units,
                   'loss_coef': fiber.west_lineic,
                   'con_in': fiber.west_con_in,
                   'con_out': fiber.west_con_out}
    }
    if fiber.west_pmd:
        fiber_dict['params']['pmd_coef'] = convert_pmd_lineic(fiber.west_pmd, fiber.west_distance, fiber.distance_units)
    return fiber_dict


def xls_to_json_data(input_filename: Path, filter_region: List[str] = None) -> dict:
    """Read the Excel sheets and produce the JSON dict in GNPy format (legacy).

    :param input_filename: The path to the input XLS file.
    :type input_filename: Path
    :param filter_region: A list of regions to filter the nodes (default is None).
    :type filter_region: List[str]
    :return: A dictionary representing the JSON data.
    :rtype: dict
    """
    if filter_region is None:
        filter_region = []
    nodes, links, eqpts, roadms = parse_excel(input_filename)
    if filter_region:
        nodes = [n for n in nodes if n.region.lower() in filter_region]
        cities = {n.city for n in nodes}
        links = [lnk for lnk in links if lnk.from_city in cities and lnk.to_city in cities]
        cities = {lnk.from_city for lnk in links} | {lnk.to_city for lnk in links}
        nodes = [n for n in nodes if n.city in cities]

    nodes_by_city = {n.city: n for n in nodes}

    links_by_city = defaultdict(list)
    for link in links:
        links_by_city[link.from_city].append(link)
        links_by_city[link.to_city].append(link)

    eqpts_by_city = defaultdict(list)
    for eqpt in eqpts:
        eqpts_by_city[eqpt.from_city].append(eqpt)

    roadms_by_city = defaultdict(list)
    for roadm in roadms:
        roadms_by_city[roadm.from_node].append(roadm)

    nodes, links = sanity_check(nodes, links, nodes_by_city, links_by_city, eqpts_by_city)

    return {
        'elements':
            [{'uid': f'trx {x.city}',
              'metadata': {'location': {'city': x.city,
                                        'region': x.region,
                                        'latitude': x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Transceiver'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'roadm']
            + [create_roadm_element(x, roadms_by_city)
               for x in nodes_by_city.values() if x.node_type.lower() == 'roadm']
            + [{'uid': f'west fused spans in {x.city}',
                'metadata': {'location': {'city': x.city,
                                          'region': x.region,
                                          'latitude': x.latitude,
                                          'longitude': x.longitude}},
                'type': 'Fused'}
               for x in nodes_by_city.values() if x.node_type.lower() == 'fused']
            + [{'uid': f'east fused spans in {x.city}',
                'metadata': {'location': {'city': x.city,
                                          'region': x.region,
                                          'latitude': x.latitude,
                                          'longitude': x.longitude}},
                'type': 'Fused'}
               for x in nodes_by_city.values() if x.node_type.lower() == 'fused']
            + [create_east_fiber_element(x, nodes_by_city) for x in links]
            + [create_west_fiber_element(x, nodes_by_city) for x in links]
            + [{'uid': f'west edfa in {x.city}',
                'metadata': {'location': {'city': x.city,
                                          'region': x.region,
                                          'latitude': x.latitude,
                                          'longitude': x.longitude}},
                'type': 'Edfa',
                'operational': {'gain_target': None,
                                'tilt_target': None}
                } for x in nodes_by_city.values() if x.node_type.lower() == 'ila' and x.city not in eqpts_by_city]
            + [{'uid': f'east edfa in {x.city}',
                'metadata': {'location': {'city': x.city,
                                          'region': x.region,
                                          'latitude': x.latitude,
                                          'longitude': x.longitude}},
                'type': 'Edfa',
                'operational': {'gain_target': None,
                                'tilt_target': None}
                } for x in nodes_by_city.values() if x.node_type.lower() == 'ila' and x.city not in eqpts_by_city]
            + [create_east_eqpt_element(e, nodes_by_city) for e in eqpts]
            + [create_west_eqpt_element(e, nodes_by_city) for e in eqpts],
        'connections':
            list(chain.from_iterable([eqpt_connection_by_city(n.city, eqpts_by_city, links_by_city, nodes_by_city)
                                      for n in nodes]))
            + list(chain.from_iterable(zip(
                [{'from_node': f'trx {x.city}', 'to_node': f'roadm {x.city}'}
                 for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'],
                [{'from_node': f'roadm {x.city}', 'to_node': f'trx {x.city}'}
                 for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'])))
    }


def convert_file(input_filename: Path, filter_region: List[str] = None, output_json_file_name: Path = None) -> Path:
    """Convert the input XLS file to JSON format and save it.

    :param input_filename: The path to the input XLS file.
    :type input_filename: Path
    :param filter_region: A list of regions to filter the nodes (default is None).
    :type filter_region: List[str]
    :param output_json_file_name: The path to save the output JSON file (default is None).
    :type output_json_file_name: Path
    :return: The path to the saved JSON file.
    :rtype: Path
    """
    if filter_region is None:
        filter_region = []
    data = xls_to_json_data(input_filename, filter_region)
    if output_json_file_name is None:
        output_json_file_name = input_filename.with_suffix('.json')
    with open(output_json_file_name, 'w', encoding='utf-8') as edfa_json_file:
        edfa_json_file.write(dumps(data, indent=2, ensure_ascii=False))
        edfa_json_file.write('\n')   # add end of file newline because json dumps does not.
    return output_json_file_name


def corresp_names(input_filename: Path, network: DiGraph) -> Tuple[dict, dict, dict]:
    """Build the correspondence between names given in the Excel and names used in the JSON.

    :param input_filename: The path to the input XLS file.
    :type input_filename: Path
    :param network: The network graph object.
    :type network: DiGraph
    :return: A tuple containing dictionaries for ROADMs, fused nodes, and ILAs.
    :rtype: Tuple[dict, dict, dict]
    """
    nodes, links, eqpts, _ = parse_excel(input_filename)
    fused = [n.uid for n in network.nodes() if isinstance(n, Fused)]
    ila = [n.uid for n in network.nodes() if isinstance(n, Edfa)]

    corresp_roadm = {x.city: [f'roadm {x.city}'] for x in nodes
                     if x.node_type.lower() == 'roadm'}
    corresp_fused = {x.city: [f'west fused spans in {x.city}', f'east fused spans in {x.city}']
                     for x in nodes if x.node_type.lower() == 'fused'
                     and f'west fused spans in {x.city}' in fused
                     and f'east fused spans in {x.city}' in fused}
    corresp_ila = defaultdict(list)
    # add the special cases when an ila is changed into a fused
    for my_e in eqpts:
        name = f'east edfa in {my_e.from_city} to {my_e.to_city}'
        if my_e.east_amp_type.lower() == 'fused' and name in fused:
            corresp_fused.get(my_e.from_city, []).append(name)
        name = f'west edfa in {my_e.from_city} to {my_e.to_city}'
        if my_e.west_amp_type.lower() == 'fused' and name in fused:
            corresp_fused.get(my_e.from_city, []).append(name)
    # build corresp ila based on eqpt sheet
    # start with east direction
    for my_e in eqpts:
        for name in [f'east edfa in {my_e.from_city} to {my_e.to_city}',
                     f'west edfa in {my_e.from_city} to {my_e.to_city}']:
            if name in ila:
                corresp_ila[my_e.from_city].append(name)
    # complete with potential autodesign names: amplifiers
    for my_l in links:
        # create names whatever the type and filter them out
        # from-to direction
        names = [f'Edfa_preamp_roadm {my_l.from_city}_from_fiber ({my_l.to_city} \u2192 {my_l.from_city})-{my_l.west_cable}',
                 f'Edfa_booster_roadm {my_l.from_city}_to_fiber ({my_l.from_city} \u2192 {my_l.to_city})-{my_l.east_cable}']
        for name in names:
            if name in ila:
                # "east edfa in Stbrieuc to Rennes_STA"  is equivalent name as
                # "Edfa_booster_roadm Stbrieuc_to_fiber (Lannion_CAS → Stbrieuc)-F056"
                # "west edfa in Stbrieuc to Rennes_STA"  is equivalent name as
                # "Edfa_preamp_roadm Stbrieuc_to_fiber (Rennes_STA → Stbrieuc)-F057"
                # in case fibers are splitted the name here is a
                corresp_ila[my_l.from_city].append(name)
        # to-from direction
        names = [f'Edfa_preamp_roadm {my_l.to_city}_from_fiber ({my_l.from_city} \u2192 {my_l.to_city})-{my_l.east_cable}',
                 f'Edfa_booster_roadm {my_l.to_city}_to_fiber ({my_l.to_city} \u2192 {my_l.from_city})-{my_l.west_cable}']
        for name in names:
            if name in ila:
                corresp_ila[my_l.to_city].append(name)
    for node in nodes:
        names = [f'east edfa in {node.city}', f'west edfa in {node.city}']
        for name in names:
            if name in ila:
                # "east edfa in Stbrieuc to Rennes_STA" (created with Eqpt) is equivalent name as
                # "east edfa in Stbrieuc" or "west edfa in Stbrieuc" (created with Links sheet)
                # depending on link node order
                corresp_ila[node.city].append(name)

    # merge fused with ila:
    for key, val in corresp_fused.items():
        corresp_ila[key].extend(val)
        # no need of roadm booster
    return corresp_roadm, corresp_fused, corresp_ila


def parse_excel(input_filename: Path) -> Tuple[List[Node], List[Link], List[Eqpt], List[Roadm]]:
    """Reads XLS(X) sheets among Nodes, Eqpts, Links, Roadms and parses the data.

    :param input_filename: The path to the input XLS file.
    :type input_filename: Path
    :return: A tuple containing lists of Node, Link, Eqpt, and Roadm objects.
    :rtype: Tuple[List[Node], List[Link], List[Eqpt], List[Roadm]]
    :raises NetworkTopologyError: If any issues are found during parsing.
    """
    link_headers = {
        'Node A': 'from_city',
        'Node Z': 'to_city',
        'east': {
            'Distance (km)': 'east_distance',
            'Fiber type': 'east_fiber',
            'lineic att': 'east_lineic',
            'Con_in': 'east_con_in',
            'Con_out': 'east_con_out',
            'PMD': 'east_pmd',
            'Cable id': 'east_cable'
        },
        'west': {
            'Distance (km)': 'west_distance',
            'Fiber type': 'west_fiber',
            'lineic att': 'west_lineic',
            'Con_in': 'west_con_in',
            'Con_out': 'west_con_out',
            'PMD': 'west_pmd',
            'Cable id': 'west_cable'
        }
    }
    node_headers = {
        'City': 'city',
        'State': 'state',
        'Country': 'country',
        'Region': 'region',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Type': 'node_type',
        'Booster_restriction': 'booster_restriction',
        'Preamp_restriction': 'preamp_restriction'
    }
    eqpt_headers = {
        'Node A': 'from_city',
        'Node Z': 'to_city',
        'east': {
            'amp type': 'east_amp_type',
            'amp gain': 'east_amp_gain',
            'delta p': 'east_amp_dp',
            'tilt': 'east_tilt_vs_wavelength',
            'att_out': 'east_att_out',
            'att_in': 'east_att_in'
        },
        'west': {
            'amp type': 'west_amp_type',
            'amp gain': 'west_amp_gain',
            'delta p': 'west_amp_dp',
            'tilt': 'west_tilt_vs_wavelength',
            'att_out': 'west_att_out',
            'att_in': 'west_att_in'
        }
    }
    roadm_headers = {'Node A': 'from_node',
                     'Node Z': 'to_node',
                     'per degree target power (dBm)': 'target_pch_out_db',
                     'type_variety': 'type_variety',
                     'from degrees': 'from_degrees',
                     'from degree to degree impairment id': 'impairment_ids'
                     }

    with open_workbook(input_filename) as wb:
        nodes_sheet = wb.sheet_by_name('Nodes')
        links_sheet = wb.sheet_by_name('Links')
        try:
            eqpt_sheet = wb.sheet_by_name('Eqpt')
        except XLRDError:
            # eqpt_sheet is optional
            eqpt_sheet = None
        try:
            roadm_sheet = wb.sheet_by_name('Roadms')
        except XLRDError:
            # roadm_sheet is optional
            roadm_sheet = None

        nodes = [Node(**node) for node in parse_sheet(nodes_sheet, node_headers,
                                                      NODES_LINE, NODES_LINE + 1, NODES_COLUMN)]
        expected_node_types = {'ROADM', 'ILA', 'FUSED'}
        for n in nodes:
            if n.node_type not in expected_node_types:
                n.node_type = 'ILA'

        links = [Link(**link) for link in parse_sheet(links_sheet, link_headers,
                                                      LINKS_LINE, LINKS_LINE + 2, LINKS_COLUMN)]
        eqpts = []
        if eqpt_sheet is not None:
            eqpts = [Eqpt(**eqpt) for eqpt in parse_sheet(eqpt_sheet, eqpt_headers,
                                                          EQPTS_LINE, EQPTS_LINE + 2, EQPTS_COLUMN)]
        roadms = []
        if roadm_sheet is not None:
            roadms = [Roadm(**roadm) for roadm in parse_sheet(roadm_sheet, roadm_headers,
                                                              ROADMS_LINE, ROADMS_LINE + 2, ROADMS_COLUMN)]

    # sanity check
    all_cities = Counter(n.city for n in nodes)
    if len(all_cities) != len(nodes):
        msg = f'Duplicate city: {all_cities}'
        raise NetworkTopologyError(msg)
    bad_links = []
    for lnk in links:
        if lnk.from_city not in all_cities or lnk.to_city not in all_cities:
            bad_links.append([lnk.from_city, lnk.to_city])

    if bad_links:
        msg = 'XLS error: ' \
              + 'The Links sheet references nodes that ' \
              + 'are not defined in the Nodes sheet:\n' \
              + _format_items(f'{item[0]} -> {item[1]}' for item in bad_links)
        raise NetworkTopologyError(msg)

    return nodes, links, eqpts, roadms


def eqpt_connection_by_city(city_name: str, eqpts_by_city: DefaultDict[str, List[Eqpt]],
                            links_by_city: DefaultDict[str, List[Link]], nodes_by_city: Dict[str, Node]) -> list:
    """Returns the list of equipment installed in the specified city.

    :param city_name: The name of the city to check for equipment.
    :type city_name: str
    :param eqpts_by_city: A defaultdict mapping city names to lists of Eqpt objects.
    :type eqpts_by_city: DefaultDict[str, List[Eqpt]]
    :param links_by_city: A defaultdict mapping city names to lists of Link objects.
    :type links_by_city: DefaultDict[str, List[Link]]
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, Node]
    :return: A list of connection dictionaries for the specified city.
    :rtype: list
    """
    other_cities = fiber_dest_from_source(city_name, links_by_city)
    subdata = []
    if nodes_by_city[city_name].node_type.lower() in {'ila', 'fused'}:
        # Then len(other_cities) == 2
        direction = ['west', 'east']
        for i in range(2):
            from_ = fiber_link(other_cities[i], city_name, links_by_city)
            in_ = eqpt_in_city_to_city(city_name, other_cities[0], eqpts_by_city, nodes_by_city, direction[i])
            to_ = fiber_link(city_name, other_cities[1 - i], links_by_city)
            subdata += connect_eqpt(from_, in_, to_)
    elif nodes_by_city[city_name].node_type.lower() == 'roadm':
        for other_city in other_cities:
            from_ = f'roadm {city_name}'
            in_ = eqpt_in_city_to_city(city_name, other_city, eqpts_by_city, nodes_by_city)
            to_ = fiber_link(city_name, other_city, links_by_city)
            subdata += connect_eqpt(from_, in_, to_)

            from_ = fiber_link(other_city, city_name, links_by_city)
            in_ = eqpt_in_city_to_city(city_name, other_city, eqpts_by_city, nodes_by_city, "west")
            to_ = f'roadm {city_name}'
            subdata += connect_eqpt(from_, in_, to_)
    return subdata


def connect_eqpt(from_: str, in_: str, to_: str) -> List[dict]:
    """Create the topology connection JSON dict between in and to.

    :param from_: The starting node identifier.
    :type from_: str
    :param in_: The intermediate node identifier.
    :type in_: str
    :param to_: The ending node identifier.
    :type to_: str
    :return: A list of connection dictionaries.
    :rtype: List[dict]
    """
    connections = []
    if in_ != '':
        connections = [{'from_node': from_, 'to_node': in_},
                       {'from_node': in_, 'to_node': to_}]
    else:
        connections = [{'from_node': from_, 'to_node': to_}]
    return connections


def eqpt_in_city_to_city(in_city: str, to_city: str,
                         eqpts_by_city: DefaultDict[str, List[Eqpt]], nodes_by_city: Dict[str, Node],
                         direction: str = 'east') -> str:
    """Returns the formatted string corresponding to in_city types and direction.

    :param in_city: The city where the equipment is located.
    :type in_city: str
    :param to_city: The city where the equipment connects to.
    :type to_city: str
    :param eqpts_by_city: A defaultdict mapping city names to lists of Eqpt objects.
    :type eqpts_by_city: DefaultDict[str, List[Eqpt]]
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, Node]
    :param direction: The direction of the equipment (default is 'east').
    :type direction: str
    :return: A formatted string representing the equipment in the specified direction.
    :rtype: str
    """
    rev_direction = 'west' if direction == 'east' else 'east'
    return_eqpt = ''
    if in_city in eqpts_by_city:
        for e in eqpts_by_city[in_city]:
            if nodes_by_city[in_city].node_type.lower() == 'roadm':
                if e.to_city == to_city:
                    return_eqpt = f'{direction} edfa in {e.from_city} to {e.to_city}'
            elif nodes_by_city[in_city].node_type.lower() == 'ila':
                if e.to_city != to_city:
                    direction = rev_direction
                return_eqpt = f'{direction} edfa in {e.from_city} to {e.to_city}'
    elif nodes_by_city[in_city].node_type.lower() == 'ila':
        return_eqpt = f'{direction} edfa in {in_city}'
    if nodes_by_city[in_city].node_type.lower() == 'fused':
        return_eqpt = f'{direction} fused spans in {in_city}'
    return return_eqpt


def corresp_next_node(network: DiGraph, corresp_ila: dict, corresp_roadm: dict) -> Tuple[dict, dict]:
    """Find the next node in the network for each name in the correspondence dictionaries.
    For each name in corresp dictionnaries find the next node in network and its name
    given by user in excel. for meshTopology_exampleV2.xls:
    user ILA name Stbrieuc covers the two direction. convert.py creates 2 different ILA
    with possible names (depending on the direction and if the eqpt was defined in eqpt
    sheet)
    for an ILA and if it is defined in eqpt:

    - east edfa in Stbrieuc to Rennes_STA
    - west edfa in Stbrieuc to Rennes_STA

    for an ILA and if it is notdefined in eqpt:

    - east edfa in Stbrieuc
    - west edfa in Stbrieuc

    for a roadm

    - "Edfa_preamp_roadm node1_from_fiber (siteE → node1)-CABLES#19"
    - "Edfa_booster_roadm node1_to_fiber (node1 → siteE)-CABLES#19"

    next_nodes finds the user defined name of next node to be able to map the path constraints

    - east edfa in Stbrieuc to Rennes_STA      next node = Rennes_STA
    - west edfa in Stbrieuc to Rennes_STA      next node = Lannion_CAS

    the function supports fiber splitting, fused nodes and shall only be called if
    excel format is used for both network and service

    :param network: The network graph object.
    :type network: DiGraph
    :param corresp_ila: A dictionary mapping city names to lists of ILA names.
    :type corresp_ila: dict
    :param corresp_roadm: A dictionary mapping city names to lists of ROADM names.
    :type corresp_roadm: dict
    :return: A tuple containing updated correspondence for ILAs and the next node mapping.
    :rtype: Tuple[dict, dict]
    """
    next_node = {}
    # consolidate tables and create next_node table
    for ila_key, ila_list in corresp_ila.items():
        temp = copy(ila_list)
        for ila_elem in ila_list:
            # find the node with ila_elem string _in_ the node uid. 'in' is used instead of
            # '==' to find composed nodes due to fiber splitting in autodesign.
            # eg if elem_ila is 'east edfa in Stbrieuc to Rennes_STA',
            # node uid 'east edfa in Stbrieuc to Rennes_STA-_(1/2)' is possible
            correct_ila_name = next(n.uid for n in network.nodes() if ila_elem in n.uid)
            temp.remove(ila_elem)
            temp.append(correct_ila_name)
            ila_nd = next(n for n in network.nodes() if ila_elem in n.uid)
            next_nd = next(network.successors(ila_nd))
            # search for the next ILA or ROADM
            while isinstance(next_nd, (Fiber, Fused)):
                next_nd = next(network.successors(next_nd))
            # if next_nd is a ROADM, add the first found correspondance
            for key, val in corresp_roadm.items():
                # val is a list of possible names associated with key
                if next_nd.uid in val:
                    next_node[correct_ila_name] = key
                    break
            # if next_nd was not already added in the dict with the previous loop,
            # add the first found correspondance in ila names
            if correct_ila_name not in next_node:
                for key, val in corresp_ila.items():
                    # in case of splitted fibers the ila name might not be exact match
                    if [e for e in val if e in next_nd.uid]:
                        next_node[correct_ila_name] = key
                        break

        corresp_ila[ila_key] = temp
    return corresp_ila, next_node


def fiber_dest_from_source(city_name: str, links_by_city: DefaultDict[str, List[Link]]) -> List[str]:
    """Returns the list of cities connected to the specified city.

    :param city_name: The name of the city to check for connections.
    :type city_name: str
    :param links_by_city: A defaultdict mapping city names to lists of Link objects.
    :type links_by_city: DefaultDict[str, List[Link]]
    :return: A list of city names that are connected to the specified city.
    :rtype: List[str]
    """
    destinations = []
    links_from_city = links_by_city[city_name]
    for l in links_from_city:
        if l.from_city == city_name:
            destinations.append(l.to_city)
        else:
            destinations.append(l.from_city)
    return destinations


def fiber_link(from_city: str, to_city: str, links_by_city: DefaultDict[str, List[Link]]) -> str:
    """Returns the formatted UID for fibers between two cities.

    :param from_city: The starting city name.
    :type from_city: str
    :param to_city: The destination city name.
    :type to_city: str
    :param links_by_city: A defaultdict mapping city names to lists of Link objects.
    :type links_by_city: DefaultDict[str, List[Link]]
    :return: A formatted string representing the fiber link.
    :rtype: str
    """
    source_dest = (from_city, to_city)
    links = links_by_city[from_city]
    link = next(l for l in links if l.from_city in source_dest and l.to_city in source_dest)
    if link.from_city == from_city:
        fiber = f'fiber ({link.from_city} \u2192 {link.to_city})-{link.east_cable}'
    else:
        fiber = f'fiber ({link.to_city} \u2192 {link.from_city})-{link.west_cable}'
    return fiber


def midpoint(city_a: Node, city_b: Node) -> dict:
    """Computes the midpoint coordinates between two cities.

    :param city_a: The first Node object representing a city.
    :type city_a: Node
    :param city_b: The second Node object representing a city.
    :type city_b: Node
    :return: A dictionary containing the latitude and longitude of the midpoint.
    :rtype: dict
    """
    lats = city_a.latitude, city_b.latitude
    longs = city_a.longitude, city_b.longitude
    try:
        result = {
            'latitude': sum(lats) / 2,
            'longitude': sum(longs) / 2
        }
    except TypeError:
        result = {
            'latitude': 0,
            'longitude': 0
        }
    return result

# TODO get column size automatically from tupple size


NODES_COLUMN = 10
NODES_LINE = 4
LINKS_COLUMN = 16
LINKS_LINE = 3
EQPTS_LINE = 3
EQPTS_COLUMN = 14
ROADMS_LINE = 3
ROADMS_COLUMN = 6


def _do_convert():
    """Main function for xls(x) topology conversion to JSON format
    """
    parser = ArgumentParser()
    parser.add_argument('workbook', type=Path)
    parser.add_argument('-f', '--filter-region', action='append', default=[])
    parser.add_argument('--output', type=Path, help='Name of the generated JSON file')
    args = parser.parse_args()
    res = convert_file(args.workbook, args.filter_region, args.output)
    print(f'XLS -> JSON saved to {res}')


if __name__ == '__main__':
    _do_convert()

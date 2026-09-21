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
from copy import copy, deepcopy
from typing import Generator, Tuple, List, Dict, DefaultDict, Optional
from networkx import DiGraph

from gnpy.core.utils import silent_remove, transform_data, convert_pmd_lineic
from gnpy.core.exceptions import NetworkTopologyError
from gnpy.core.elements import Edfa, Fused, Fiber
from gnpy.tools.xls_utils import SheetType, all_rows, generic_open_workbook, get_row_slice, get_sheet, \
    XLS_EXCEPTIONS, is_type_cell_empty


_logger = getLogger(__name__)


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
    :ivar degree_association: Associations between parallel ROADM degrees.
    :vartype degree_association: list
    :ivar pair_id: Associations between parallel ROADM degrees.
    :vartype pair_id: str
    """
    def __init__(self, **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)
        self.degree_association = []

    def update_attr(self, kwargs):
        """Updates the attributes of the node based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != '' and v is not None}
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
        'preamp_restriction': '',
        'pair_id': None
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
    :ivar pair_id: Associations between parallel ROADM degrees.
    :vartype pair_id: str
    :ivar row_id: Associations between parallel ROADM degrees.
    :vartype row_id: int
    """

    def __init__(self, row_id: Optional[int], **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)
        self.distance_units = 'km'
        self.row_id = row_id

    def update_attr(self, kwargs):
        """Updates the attributes of the link based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != '' and v is not None}
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
        'east_cable': '',
        'row_id': None,
        'pair_id': None
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
    :ivar pair_id: Associations between parallel ROADM degrees.
    :vartype pair_id: str
    :ivar row_id: Associations between parallel ROADM degrees.
    :vartype row_id: int
    """
    def __init__(self, row_id: Optional[int], **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)
        self.row_id = row_id

    def update_attr(self, kwargs):
        """Updates the attributes of the equipment based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != '' and v is not None}
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
        'east_att_in': 0,
        'row_id': None,
        'pair_id': None
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
    :ivar pair_id: Associations between parallel ROADM degrees.
    :vartype pair_id: str
    :ivar row_id: Associations between parallel ROADM degrees.
    :vartype row_id: int
    """
    def __init__(self, row_id: Optional[int], **kwargs):
        """Constructor method
        """
        super().__init__()
        self.update_attr(kwargs)
        self.row_id = row_id

    def update_attr(self, kwargs):
        """Updates the attributes of the ROADM based on provided keyword arguments.

        :param kwargs: A dictionary of attributes to update.
        :type kwargs: dict
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != '' and v is not None}
        for k, v in self.default_values.items():
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)

    default_values = {
        'from_node': '',
        'to_node': '',
        'target_pch_out_db': None,
        'type_variety': None,
        'from_degrees': None,
        'impairment_ids': None,
        'row_id': None,
        'pair_id': None
    }


def read_header(my_sheet: SheetType, is_xlsx: bool, line: int, slice_: Tuple[int, int]) -> List[namedtuple]:
    """Return the list of headers in a specified range.

    header_i = [(header, header_column_index), ...]
    in a {line, slice1_x, slice_y} range

    :param my_sheet: The sheet object from which to read headers.
    :type my_sheet: SheetType
    :param line: The row index to read headers from.
    :type line: int
    :param slice_: A tuple specifying the start and end column indices.
    :type slice_: Tuple[int, int]
    :return: A list of namedtuples containing headers and their column indices.
    :rtype: List[namedtuple]
    """
    param_header = namedtuple('param_header', 'header colindex')
    try:
        cells = get_row_slice(my_sheet, line, slice_[0], slice_[1], is_xlsx)
        headers = [cell.value.strip() if cell.value else '' for cell in cells]
        header_i = [param_header(header, i + slice_[0]) for i, header in enumerate(headers) if header != '']
    except (AttributeError, IndexError):
        header_i = []
    if header_i != [] and header_i[-1].colindex != slice_[1]:
        header_i.append(param_header('', slice_[1]))
    return header_i


def read_slice(my_sheet: SheetType, is_xlsx: bool, line: int, slice_: Tuple[int, int], header: str) -> Tuple[int, int]:
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
    header_i = read_header(my_sheet, is_xlsx, line, slice_)
    slice_range = (-1, -1)
    if header_i != []:
        try:
            slice_range = next((h.colindex, header_i[i + 1].colindex)
                               for i, h in enumerate(header_i) if header in h.header)
        except StopIteration:
            pass
    return slice_range


def parse_headers(my_sheet: SheetType, is_xlsx: bool, input_headers_dict: Dict, headers: Dict[int, str],
                  start_line: int, slice_in: Tuple[int, int]) -> Dict[int, str]:
    """return a dict of header_slice

    - key = column index
    - value = header name

    :param my_sheet: The sheet object from which to read headers.
    :type my_sheet: SheetType
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
        slice_out = read_slice(my_sheet, is_xlsx, start_line, slice_in, h0)
        iteration = 1
        while slice_out == (-1, -1) and iteration < 10:
            # try next lines
            slice_out = read_slice(my_sheet, is_xlsx, start_line + iteration, slice_in, h0)
            iteration += 1
        if slice_out == (-1, -1):
            msg = f'missing header {h0}'
            if h0 in ('east', 'Node A', 'Node Z', 'City'):
                raise NetworkTopologyError(f'XLS error: {msg}')
            _logger.warning(msg)
        elif not isinstance(input_headers_dict[h0], dict):
            headers[slice_out[0]] = input_headers_dict[h0]
        else:
            headers = parse_headers(my_sheet, is_xlsx, input_headers_dict[h0], headers, start_line + 1, slice_out)
    if headers == {}:
        msg = 'XLS error: could not find any header to read _ ABORT'
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


def parse_sheet(my_sheet: SheetType, is_xlsx: bool, input_headers_dict: Dict, header_line: int,
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
    headers = parse_headers(my_sheet, is_xlsx, input_headers_dict, {}, header_line, (0, column))
    for row in all_rows(my_sheet, is_xlsx, start=start_line):
        if not is_type_cell_empty(row[0], is_xlsx):
            # Check required because openpyxl in read_only mode can return "ghost" rows at the end of the document
            # (ReadOnlyCell cells with no actual value but formatting information even for empty rows).
            yield parse_row(row[0: column], headers)


def _format_items(items: List[str]):
    """Format a list of items into a string.

    :param items: A list of items to format.
    :type items: List[str]
    :return: A formatted string with each item on a new line.
    :rtype: str
    """
    items = list(items)
    if len(items[0]) == 2:
        return '\n'.join(f' - {item[0]} -> {item[1]}' for item in items)
    return '\n'.join(f' - {item}' for item in items)


def sanity_check(nodes: List[Node], links: List[Link], roadms: List[Roadm],
                 nodes_by_city: Dict[str, List[Node]]) -> Tuple[List[Node], List[Link]]:
    """Perform sanity checks on nodes and links. Raise correct issues if xls(x) is not correct,
    Checks duplicate links, unreferenced nodes in links, in eqpts, unreferenced link in eqpts, duplicate items

    :param nodes: A list of Node objects.
    :type nodes: List[Node]
    :param links: A list of Link objects.
    :type links: List[Link]
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, List[Node]]
    :return: A tuple containing the validated lists of nodes and links.
    :rtype: Tuple[List[Node], List[Link]]
    """
    # unreferenced nodes are detected in assign_implicit_pair_ids()
    # no need to check "Links" for invalid nodes because that's already in parse_excel()
    # wrong duplicate links are detected in assign_implicit_pair_ids()
    # wrong eqpt links are detected in assign_implicit_pair_ids()
    bad_roadm = [(n.from_node, n.to_node) for n in roadms
                 if n.from_node not in nodes_by_city or n.to_node not in nodes_by_city]
    if bad_roadm:
        msg = 'XLS error: ' \
            + 'The Roadm sheet references nodes that are not defined in the Links sheet:\n' \
            + _format_items(bad_roadm)
        raise NetworkTopologyError(msg)
    bad_roadm_type = [n.from_node for n in roadms
                      if nodes_by_city[n.from_node][0].node_type.lower() != 'roadm']
    if bad_roadm_type:
        msg = 'XLS error: ' \
            + 'The Roadm sheet references nodes that are not Roadms:\n' \
            + _format_items(bad_roadm_type)
        raise NetworkTopologyError(msg)
    bad_roadm_degrees = []
    for node in roadms:
        possible_degrees = [n.to_city for n in links if n.from_city == node.from_node] \
            + [n.from_city for n in links if n.to_city == node.from_node]
        if node.to_node not in possible_degrees:
            bad_roadm_degrees.append((node.from_node, node.to_node))
    if bad_roadm_degrees:
        msg = 'XLS error: ' \
            + 'The Roadm sheet references degrees that are not defined in the Links sheet:\n' \
            + _format_items(bad_roadm_degrees)
        raise NetworkTopologyError(msg)
    return nodes, links


def find_roadm_degree_uid(node: Node, roadm: Roadm, eqpts_by_city: DefaultDict[str, List[Eqpt]],
                          links_by_city: DefaultDict[str, List[Link]], ) -> Optional[str]:
    """Find the UID associated with a ROADM degree."""

    return next(
        chain(
            (
                eqpt.east_uid
                for eqpt in eqpts_by_city[node.city]
                if eqpt.pair_id == roadm.pair_id
                and eqpt.to_city == roadm.to_node
            ),
            (
                link.east_uid
                for link in links_by_city[node.city]
                if link.pair_id == roadm.pair_id
                and link.to_city == roadm.to_node
            ),
            (
                link.west_uid
                for link in links_by_city[node.city]
                if link.pair_id == roadm.pair_id
                and link.from_city == roadm.to_node
            ),
        ),
        None,
    )


def create_roadm_element(node: Node, roadms_by_city: DefaultDict[str, List[Roadm]],
                         links_by_city: DefaultDict[str, List[Link]],
                         eqpts_by_city: DefaultDict[str, List[Eqpt]]) -> Dict:
    """Create the json element for a roadm node, including the different cases:

        - if there are restrictions
        - if there are per degree target power defined on a direction

    direction is defined by the booster name, so that booster must also be created in eqpt sheet
    if the direction is defined in roadm.

    :param node: The Node object representing the ROADM.
    :type node: Node
    :param roadms_by_city: A dictionary mapping city names to lists of ROADM objects.
    :type roadms_by_city: DefaultDict[str, List[Roadm]]
    :param links_by_city: A defaultdict mapping city names to lists of Link objects.
    :type links_by_city: DefaultDict[str, List[Link]]
    :param eqpts_by_city: A defaultdict mapping city names to lists of Eqpt objects.
    :type eqpts_by_city: DefaultDict[str, List[Eqpt]]
    :return: A dictionary representing the ROADM element in JSON format.
    :rtype: Dict
    """
    roadm = {'uid': node.roadm_uid}
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
            to_node = find_roadm_degree_uid(node, elem, eqpts_by_city, links_by_city)
            if to_node is None:
                raise NetworkTopologyError(
                    f'XLS error: Wrong definition in Roadms sheet: '
                    f'degree {elem.to_node} does not exist on Roadm {node.city}'
                )
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
    # add association for parallel links
    if node.degree_association:

        # do not export degree association for single pairs
        roadm_degree = []
        for (degree, paired_degree) in node.degree_association:
            temp = [e.east_uid for e in links_by_city[node.city] if e.pair_id == '0' and e.from_city == node.city] \
                + [e.west_uid for e in links_by_city[node.city] if e.pair_id == '0' and e.to_city == node.city] \
                + [e.east_uid for e in eqpts_by_city[node.city] if e.pair_id == '0']
            if degree in temp:
                # do not create a degree export for the first pair
                continue
            roadm_degree.append(
                {"degree": degree,
                 "paired-degree": paired_degree}
            )
        if 'params' not in roadm and roadm_degree:
            roadm['params'] = {'degree-association': roadm_degree}
        elif 'params' in roadm and roadm_degree:
            roadm['params']['degree-association'] = roadm_degree

    roadm['metadata'] = {'location': {'city':      node.city,      # noqa: E241
                                      'region':    node.region,    # noqa: E241
                                      'latitude':  node.latitude,  # noqa: E241
                                      'longitude': node.longitude}}
    roadm['type'] = 'Roadm'
    return roadm


def create_east_eqpt_element(eqpt: Eqpt, nodes_by_city: Dict[str, List[Node]]) -> dict:
    """Create the JSON element for the east-facing equipment.

    :param eqpt: The equipment definition.
    :type eqpt: Eqpt
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, List[Node]]
    :return: A dictionary representing the east equipment element in JSON format.
    :rtype: dict
    """
    eqpt_dict = {
        'uid': eqpt.east_uid,
        'metadata': {
            'location': {
                'city':      nodes_by_city[eqpt.from_city][0].city,      # noqa: E241
                'region':    nodes_by_city[eqpt.from_city][0].region,    # noqa: E241
                'latitude':  nodes_by_city[eqpt.from_city][0].latitude,  # noqa: E241
                'longitude': nodes_by_city[eqpt.from_city][0].longitude}}}
    if eqpt.east_amp_type.lower() != '' and eqpt.east_amp_type.lower() != 'fused':
        eqpt_dict['type'] = 'Edfa'
        eqpt_dict['type_variety'] = f'{eqpt.east_amp_type}'
        eqpt_dict['operational'] = {
            'gain_target': eqpt.east_amp_gain,
            'delta_p':     eqpt.east_amp_dp,   # noqa: E241
            'tilt_target': eqpt.east_tilt_vs_wavelength,
            'out_voa':     eqpt.east_att_out,  # noqa: E241
            'in_voa':      eqpt.east_att_in}   # noqa: E241
    elif eqpt.east_amp_type.lower() == '':
        eqpt_dict['type'] = 'Edfa'
        eqpt_dict['operational'] = {
            'gain_target': eqpt.east_amp_gain,
            'delta_p':     eqpt.east_amp_dp,   # noqa: E241
            'tilt_target': eqpt.east_tilt_vs_wavelength,
            'out_voa':     eqpt.east_att_out,  # noqa: E241
            'in_voa':      eqpt.east_att_in}   # noqa: E241
    elif eqpt.east_amp_type.lower() == 'fused':
        # fused edfa variety is a hack to indicate that there should not be
        # booster amplifier out the roadm.
        # If user specifies ILA in Nodes sheet and fused in Eqpt sheet, then assumes that
        # this is a fused nodes.
        eqpt_dict['type'] = 'Fused'
        eqpt_dict['params'] = {'loss': 0}
    return eqpt_dict


def create_west_eqpt_element(eqpt: Eqpt, nodes_by_city: Dict[str, List[Node]]) -> dict:
    """Create the JSON element for the west-facing equipment.

    :param eqpt: The equipment definition.
    :type eqpt: Eqpt
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, List[Node]]
    :return: A dictionary representing the east equipment element in JSON format.
    :rtype: dict
    """
    eqpt_dict = {
        'uid': eqpt.west_uid,
        'metadata': {
            'location': {
                'city':      nodes_by_city[eqpt.from_city][0].city,      # noqa: E241
                'region':    nodes_by_city[eqpt.from_city][0].region,    # noqa: E241
                'latitude':  nodes_by_city[eqpt.from_city][0].latitude,  # noqa: E241
                'longitude': nodes_by_city[eqpt.from_city][0].longitude}},
        'type': 'Edfa'}
    if eqpt.west_amp_type.lower() != '' and eqpt.west_amp_type.lower() != 'fused':
        eqpt_dict['type_variety'] = f'{eqpt.west_amp_type}'
        eqpt_dict['operational'] = {
            'gain_target': eqpt.west_amp_gain,
            'delta_p':     eqpt.west_amp_dp,    # noqa: E241
            'tilt_target': eqpt.west_tilt_vs_wavelength,
            'out_voa':     eqpt.west_att_out,   # noqa: E241
            'in_voa':      eqpt.west_att_in}    # noqa: E241
    elif eqpt.west_amp_type.lower() == '':
        eqpt_dict['operational'] = {
            'gain_target': eqpt.west_amp_gain,
            'delta_p':     eqpt.west_amp_dp,    # noqa: E241
            'tilt_target': eqpt.west_tilt_vs_wavelength,
            'out_voa':     eqpt.west_att_out,   # noqa: E241
            'in_voa':      eqpt.west_att_in}    # noqa: E241
    elif eqpt.west_amp_type.lower() == 'fused':
        eqpt_dict['type'] = 'Fused'
        eqpt_dict['params'] = {'loss': 0}
    return eqpt_dict


def pair_string(element: Node | Link | Eqpt) -> str:
    return f'-pair-{element.pair_id}' if element.pair_id not in ['0', None] else ''


def create_nodes_uid(links: List[Link], eqpts: List[Eqpt], nodes: List[Node],
                     eqpts_by_city: DefaultDict[str, List[Eqpt]]):
    """Assign JSON UIDs to links, equipment and network nodes.

    Fiber UIDs are generated from link endpoints, cable identifiers and pair
    identifiers. Equipment UIDs are generated from their endpoints and pair
    identifiers. ILA and FUSED node UIDs are generated according to their
    node type and pair identifier.

    :param links: A list of network links.
    :type links: List[Link]
    :param eqpts: A list of equipment definitions.
    :type eqpts: List[Eqpt]
    :param nodes: A list of network nodes.
    :type nodes: List[Node]
    :param eqpts_by_city: A dictionary mapping city names to equipment
        definitions.
    :type eqpts_by_city: Dict[str, List[Eqpt]]
    """
    for fiber in links:
        cable_id = f'-{fiber.east_cable}' if fiber.east_cable else '-'
        fiber.east_uid = f'fiber ({fiber.from_city} -> {fiber.to_city}){cable_id}{pair_string(fiber)}'
        cable_id = f'-{fiber.west_cable}' if fiber.west_cable else '-'
        fiber.west_uid = f'fiber ({fiber.to_city} -> {fiber.from_city}){cable_id}{pair_string(fiber)}'
    for eqpt in eqpts:
        eqpt.east_uid = f'east edfa in {eqpt.from_city} to {eqpt.to_city}{pair_string(eqpt)}'
        eqpt.west_uid = f'west edfa in {eqpt.from_city} to {eqpt.to_city}{pair_string(eqpt)}'
    for node in nodes:
        if node.node_type.lower() == 'ila' and node.city not in eqpts_by_city:
            node.west_uid = f'west edfa in {node.city}{pair_string(node)}'
            node.east_uid = f'east edfa in {node.city}{pair_string(node)}'
        if node.node_type.lower() == 'fused':
            node.west_uid = f'west fused spans in {node.city}{pair_string(node)}'
            node.east_uid = f'east fused spans in {node.city}{pair_string(node)}'
        if node.node_type.lower() == 'roadm':
            node.roadm_uid = f'roadm {node.city}'
            node.trx_uid = f'trx {node.city}'


def create_east_fiber_element(fiber: Link, nodes_by_city: Dict[str, List[Node]]) -> Dict:
    """Create fibers json elements for the east direction.

    :param fiber: The Link object representing the fiber spant.
    :type fiber: Link
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, List[Node]]
    :return: A dictionary representing the east fiber element in JSON format.
    :rtype: Dict
    """
    fiber_dict = {
        'uid': fiber.east_uid,
        'metadata': {'location': midpoint(nodes_by_city[fiber.from_city][0],
                                          nodes_by_city[fiber.to_city][0])},
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


def create_west_fiber_element(fiber: Link, nodes_by_city: Dict[str, List[Node]]) -> Dict:
    """Create fibers json elements for the west direction.

    :param fiber: The Link object representing the fiber span.
    :type fiber: Link
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, List[Node]]
    :return: A dictionary representing the west fiber element in JSON format.
    :rtype: Dict
    """
    fiber_dict = {
        'uid': fiber.west_uid,
        'metadata': {'location': midpoint(nodes_by_city[fiber.from_city][0],
                                          nodes_by_city[fiber.to_city][0])},
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


def create_ila_element(node: Node, direction: str) -> Dict:
    """Create an EDFA element for an ILA node in the specified direction.

    :param node: The Node object representing the ILA.
    :type node: Node
    :param direction: The direction of the ILA element, either ``east`` or
        ``west``.
    :type direction: str
    :return: A dictionary representing the ILA EDFA element in JSON format.
    :rtype: dict
    """
    uid = node.west_uid if direction == 'west' else node.east_uid
    element = {
        'uid': uid,
        'metadata': {'location': {
            'city': node.city, 'region': node.region, 'latitude': node.latitude, 'longitude': node.longitude}},
        'type': 'Edfa',
        'operational': {'gain_target': None, 'tilt_target': None}
    }
    return element


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
    assign_implicit_pair_ids(nodes, links, eqpts, roadms)
    if filter_region:
        nodes = [n for n in nodes if n.region.lower() in filter_region]
        cities = {n.city for n in nodes}
        links = [lnk for lnk in links if lnk.from_city in cities and lnk.to_city in cities]
        cities = {lnk.from_city for lnk in links} | {lnk.to_city for lnk in links}
        nodes = [n for n in nodes if n.city in cities]

    nodes_by_city = defaultdict(list)
    for node in nodes:
        nodes_by_city[node.city].append(node)

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

    nodes, links = sanity_check(nodes, links, roadms, nodes_by_city)
    create_nodes_uid(links, eqpts, nodes, eqpts_by_city)
    json_connections = \
        list(chain.from_iterable([eqpt_connection_by_city(node_city, eqpts_by_city, links_by_city, nodes_by_city)
                                  for node_city in nodes_by_city])) \
        + list(chain.from_iterable(zip(
            [{'from_node': x.trx_uid, 'to_node': x.roadm_uid}
                for x in nodes if x.node_type.lower() == 'roadm'],
            [{'from_node': x.roadm_uid, 'to_node': x.trx_uid}
                for x in nodes if x.node_type.lower() == 'roadm'])))
    json_elements = [
        {
            'uid': x.trx_uid,
            'metadata': {'location': {
                'city': x.city, 'region': x.region, 'latitude': x.latitude, 'longitude': x.longitude}},
            'type': 'Transceiver'
        } for x in nodes if x.node_type.lower() == 'roadm'] \
        + [create_roadm_element(x, roadms_by_city, links_by_city, eqpts_by_city)
           for x in nodes if x.node_type.lower() == 'roadm'] \
        + [
            {
                'uid': f'{x.west_uid}',
                'metadata': {'location': {
                    'city': x.city, 'region': x.region, 'latitude': x.latitude, 'longitude': x.longitude}},
                'type': 'Fused'
            } for x in nodes if x.node_type.lower() == 'fused'] \
        + [
            {
                'uid': f'{x.east_uid}',
                'metadata': {'location': {
                    'city': x.city, 'region': x.region, 'latitude': x.latitude, 'longitude': x.longitude}},
                'type': 'Fused'
            } for x in nodes if x.node_type.lower() == 'fused'] \
        + [create_east_fiber_element(x, nodes_by_city) for x in links] \
        + [create_west_fiber_element(x, nodes_by_city) for x in links] \
        + [create_ila_element(x, 'west')
           for x in nodes if x.node_type.lower() == 'ila' and x.city not in eqpts_by_city] \
        + [create_ila_element(x, 'east')
           for x in nodes if x.node_type.lower() == 'ila' and x.city not in eqpts_by_city] \
        + [create_east_eqpt_element(e, nodes_by_city) for e in eqpts] \
        + [create_west_eqpt_element(e, nodes_by_city) for e in eqpts]
    return {
        'elements': json_elements,
        'connections': json_connections
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
    nodes, links, eqpts, roadms = parse_excel(input_filename)
    assign_implicit_pair_ids(nodes, links, eqpts, roadms)
    fused = [n.uid for n in network.nodes() if isinstance(n, Fused)]
    ila = [n.uid for n in network.nodes() if isinstance(n, Edfa)]

    corresp_roadm = {x.city: [f'roadm {x.city}'] for x in nodes
                     if x.node_type.lower() == 'roadm'}
    corresp_fused = {x.city: [f'west fused spans in {x.city}{pair_string(x)}',
                              f'east fused spans in {x.city}{pair_string(x)}']
                     for x in nodes if x.node_type.lower() == 'fused'
                     and f'west fused spans in {x.city}{pair_string(x)}' in fused
                     and f'east fused spans in {x.city}{pair_string(x)}' in fused}
    corresp_ila = defaultdict(list)
    # add the special cases when an ila is changed into a fused
    for my_e in eqpts:
        name = f'east edfa in {my_e.from_city} to {my_e.to_city}{pair_string(my_e)}'
        if my_e.east_amp_type.lower() == 'fused' and name in fused:
            corresp_fused.get(my_e.from_city, []).append(name)
        name = f'west edfa in {my_e.from_city} to {my_e.to_city}{pair_string(my_e)}'
        if my_e.west_amp_type.lower() == 'fused' and name in fused:
            corresp_fused.get(my_e.from_city, []).append(name)
    # build corresp ila based on eqpt sheet
    # start with east direction
    for my_e in eqpts:
        for name in [f'east edfa in {my_e.from_city} to {my_e.to_city}{pair_string(my_e)}',
                     f'west edfa in {my_e.from_city} to {my_e.to_city}{pair_string(my_e)}']:
            if name in ila:
                corresp_ila[my_e.from_city].append(name)
    # complete with potential autodesign names: amplifiers
    for my_l in links:
        # create names whatever the type and filter them out
        # from-to direction
        names = [
            f'Edfa_preamp_roadm {my_l.from_city}_from_fiber ({my_l.to_city} -> {my_l.from_city})-{my_l.west_cable}{pair_string(my_e)}',  # noqa E501
            f'Edfa_booster_roadm {my_l.from_city}_to_fiber ({my_l.from_city} -> {my_l.to_city})-{my_l.east_cable}{pair_string(my_e)}']   # noqa E501
        for name in names:
            if name in ila:
                # "east edfa in Stbrieuc to Rennes_STA"  is equivalent name as
                # "Edfa_booster_roadm Stbrieuc_to_fiber (Lannion_CAS → Stbrieuc)-F056"
                # "west edfa in Stbrieuc to Rennes_STA"  is equivalent name as
                # "Edfa_preamp_roadm Stbrieuc_to_fiber (Rennes_STA → Stbrieuc)-F057"
                # in case fibers are splitted the name here is a
                corresp_ila[my_l.from_city].append(name)
        # to-from direction
        names = [f'Edfa_preamp_roadm {my_l.to_city}_from_fiber ({my_l.from_city} -> {my_l.to_city})-{my_l.east_cable}{pair_string(my_e)}',  # noqa E501
                 f'Edfa_booster_roadm {my_l.to_city}_to_fiber ({my_l.to_city} -> {my_l.from_city})-{my_l.west_cable}{pair_string(my_e)}']   # noqa E501
        for name in names:
            if name in ila:
                corresp_ila[my_l.to_city].append(name)
    for node in nodes:
        names = [f'east edfa in {node.city}{pair_string(node)}', f'west edfa in {node.city}{pair_string(node)}']
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


def assign_implicit_pair_ids(nodes: List[Node], links: List[Link], eqpts: List[Eqpt], roadms: List[Roadm]) -> None:
    """Assign pair identifiers to links, equipment and ROADM degrees.

    Pair identifiers are assigned in several steps:

    - parallel links not connected to an ILA or FUSED node;
    - links passing through ILA or FUSED nodes;
    - equipment defined in the Eqpt sheet;
    - ROADM degrees associated with links;
    - duplicated ILA or FUSED nodes created for different pair identifiers.

    :param nodes: A list of network nodes.
    :type nodes: List[Node]
    :param links: A list of network links.
    :type links: List[Link]
    :param eqpts: A list of equipment definitions.
    :type eqpts: List[Eqpt]
    :param roadms: A list of ROADM degree definitions.
    :type roadms: List[Roadm]
    :raises NetworkTopologyError: If the topology contains invalid or
        inconsistent pair definitions.
    """
    # 1. Links not connected to ILAs
    assign_parallel_ids(nodes, links)

    # 2. Links crossing ILA or Fused
    assign_ila_fused_pair_ids(nodes, links)

    # 2. Equipment defined by user
    assign_equipment_pair_ids(eqpts, links, nodes)

    # 3. Roadms associated to links
    assign_roadm_degree_pair_ids(roadms, links)

    # 4. Duplicate ILA or Fused nodes with different pair_ids
    duplicate_ila_fused(nodes, links)


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

    wb, is_xlsx = generic_open_workbook(input_filename)
    nodes_sheet = get_sheet(wb, 'Nodes', is_xlsx)
    links_sheet = get_sheet(wb, 'Links', is_xlsx)
    try:
        eqpt_sheet = get_sheet(wb, 'Eqpt', is_xlsx)
    except XLS_EXCEPTIONS:
        # eqpt_sheet is optional
        eqpt_sheet = None

    try:
        roadm_sheet = get_sheet(wb, 'Roadms', is_xlsx)
    except XLS_EXCEPTIONS:
        # roadm_sheet is optional
        roadm_sheet = None

    nodes = [Node(**node) for node in parse_sheet(nodes_sheet, is_xlsx, node_headers,
                                                  NODES_LINE, NODES_LINE + 1, NODES_COLUMN)]
    expected_node_types = {'ROADM', 'ILA', 'FUSED'}
    for n in nodes:
        if n.node_type not in expected_node_types:
            n.node_type = 'ILA'

    links = [Link(**link, row_id=i)
             for i, link in enumerate(parse_sheet(links_sheet, is_xlsx, link_headers,
                                                  LINKS_LINE, LINKS_LINE + 2, LINKS_COLUMN))]

    eqpts = []
    if eqpt_sheet is not None:
        eqpts = [Eqpt(**eqpt, row_id=i)
                 for i, eqpt in enumerate(parse_sheet(eqpt_sheet, is_xlsx, eqpt_headers,
                                                      EQPTS_LINE, EQPTS_LINE + 2, EQPTS_COLUMN))]
    roadms = []
    if roadm_sheet is not None:
        roadms = [Roadm(**roadm, row_id=i)
                  for i, roadm in enumerate(parse_sheet(roadm_sheet, is_xlsx, roadm_headers,
                                                        ROADMS_LINE, ROADMS_LINE + 2, ROADMS_COLUMN))]

    # sanity check
    all_cities = Counter(n.city for n in nodes if n.city)
    if len(all_cities) != len(nodes):
        msg = f'XLS error: Duplicate city: {all_cities}'
        raise NetworkTopologyError(msg)
    bad_links = []
    for lnk in links:
        if lnk.from_city not in all_cities or lnk.to_city not in all_cities:
            bad_links.append([lnk.from_city, lnk.to_city])

    if bad_links:
        msg = 'XLS error: ' \
              + 'The Links sheet references nodes that ' \
              + 'are not defined in the Nodes sheet:\n' \
              + _format_items(bad_links)
        raise NetworkTopologyError(msg)

    return nodes, links, eqpts, roadms


def endpoint_key(link: Link):
    """Return the ordered endpoint key of a link.

    The key preserves the direction stored in the link.

    :param link: The link for which the endpoint key is generated.
    :type link: Link
    :return: A tuple containing the source and destination city names.
    :rtype: tuple[str, str]
    """
    return (link.from_city, link.to_city)


def rev_endpoint_key(link: Link):
    """Return the reversed endpoint key of a link.

    The key reverses the direction stored in the link.

    :param link: The link for which the reversed endpoint key is generated.
    :type link: Link
    :return: A tuple containing the destination and source city names.
    :rtype: tuple[str, str]
    """
    return (link.to_city, link.from_city)


def assign_parallel_ids(nodes: list[Node], links: list[Link]) -> None:
    """Assign pair identifiers to parallel links outside ILA or FUSED nodes.

    Links connected to an ILA or FUSED node are handled separately by
    :func:`assign_ila_fused_pair_ids`. Other links sharing the same ordered
    endpoints are assigned incremental pair identifiers according to their
    row order.

    :param nodes: A list of network nodes.
    :type nodes: list[Node]
    :param links: A list of network links.
    :type links: list[Link]
    """
    ila_cities = {
        node.city
        for node in nodes
        if node.node_type.lower() in ['ila', 'fused']
    }

    groups = defaultdict(list)

    for link in links:
        # Les liens connectés à un ILA seront traités séparément.
        if (link.from_city not in ila_cities
                and link.to_city not in ila_cities):
            groups[endpoint_key(link)].append(link)

    for endpoints, group in groups.items():
        group.sort(key=lambda link: link.row_id or 0)

        for pair_id, link in enumerate(group):
            if link.pair_id is None:
                link.pair_id = str(pair_id)


def assign_equipment_pair_ids(eqpts: list[Eqpt], links: list[Link], nodes: list[Node]) -> None:
    """Assign link pair identifiers to equipment definitions.

    Equipment definitions are matched with links using their endpoints.
    Both link orientations are accepted. Equipment and links are sorted by
    their source row before pair identifiers are assigned.

    This function also checks that:

    - equipment endpoints refer to existing nodes;
    - equipment endpoints refer to existing links;
    - the number of equipment definitions does not exceed the number of
      corresponding links;
    - an ILA is not assigned the same pair identifier more than once.

    :param eqpts: A list of equipment definitions.
    :type eqpts: list[Eqpt]
    :param links: A list of network links.
    :type links: list[Link]
    :param nodes: A list of network nodes.
    :type nodes: list[Node]
    :raises NetworkTopologyError: If an equipment definition references an
        unknown node or link, or if duplicate equipment definitions are found.
    """
    nodes_by_city_name = {n.city: n for n in nodes}
    nodes_name = []
    links_by_endpoints = defaultdict(list)
    for link in links:
        links_by_endpoints[endpoint_key(link)].append(link)
        links_by_endpoints[rev_endpoint_key(link)].append(link)
        nodes_name.append(link.from_city)
        nodes_name.append(link.to_city)
    eqpts_by_endpoints = defaultdict(list)

    wrong_endpoints = []
    wrong_nodes = []
    for eqpt in eqpts:
        key = (eqpt.from_city, eqpt.to_city)
        for item in key:
            if item not in nodes_name:
                wrong_nodes.append(item)
        if key not in links_by_endpoints:
            wrong_endpoints.append(key)
            continue
        eqpts_by_endpoints[key].append(eqpt)

    if wrong_nodes:
        msg = 'XLS error: ' \
            + 'The Eqpt sheet refers to nodes that ' \
            + 'are not defined in the Nodes sheet:\n'\
            + _format_items(wrong_nodes)
        raise NetworkTopologyError(msg)
    if wrong_endpoints:
        msg = 'XLS error: ' \
            + 'The Eqpt sheet refers to links that ' \
            + 'are not defined in the Links sheet:\n'\
            + _format_items(wrong_endpoints)
        raise NetworkTopologyError(msg)

    wrong_duplicate_equipment = []
    for endpoints, eqpt_group in eqpts_by_endpoints.items():
        link_group = sorted(links_by_endpoints.get(endpoints, []), key=lambda link: link.row_id or 0)

        eqpt_group = sorted(eqpt_group, key=lambda eqpt: eqpt.row_id or 0)

        if len(eqpt_group) > len(link_group):
            wrong_duplicate_equipment.append(endpoints)
            # collect all mistakes before raising an error
            continue
        for eqpt, link in zip(eqpt_group, link_group):
            eqpt.pair_id = link.pair_id
    if wrong_duplicate_equipment:
        msg = 'XLS error: Duplicate eqpt definition in Eqpt for not duplicated links:\n' \
              + _format_items(wrong_duplicate_equipment)
        raise NetworkTopologyError(msg)

    ilas_by_endpoint = defaultdict(list)
    for eqpt in eqpts:
        if nodes_by_city_name[eqpt.from_city].node_type.lower() == 'ila':
            ilas_by_endpoint[eqpt.from_city].append(eqpt.pair_id)

    wrong_reverse_duplicate_equipment = []
    for endpoint, pair_ids in ilas_by_endpoint.items():
        if len(pair_ids) != len(set(pair_ids)):
            wrong_reverse_duplicate_equipment.append(endpoint)

    if wrong_reverse_duplicate_equipment:
        msg = (
            'XLS error: Duplicate eqpt definition in Eqpt for the same ILA:\n'
            + _format_items(wrong_reverse_duplicate_equipment)
        )
        raise NetworkTopologyError(msg)


def assign_ila_fused_pair_ids(nodes: list[Node], links: list[Link]) -> None:
    """Assign pair identifiers to links passing through ILA or FUSED nodes.

    Each ILA or FUSED node must have exactly two neighboring nodes. The links
    connected to both neighbors must have the same number of entries so that
    they can be paired. Pair identifiers are assigned according to the link
    row order.

    Existing pair identifiers are preserved and checked for consistency.

    :param nodes: A list of network nodes.
    :type nodes: list[Node]
    :param links: A list of network links.
    :type links: list[Link]
    :raises NetworkTopologyError: If an ILA or FUSED node is unreferenced, if
        it does not have exactly two neighbors, or if the two sides contain
        different numbers of links.
    :raises ValueError: If an already assigned pair identifier is inconsistent
        with the expected link order.
    """
    ila_fused_cities = {
        node.city
        for node in nodes
        if node.node_type.lower() in ['ila', 'fused']
    }
    wrong_duplications = []
    unreferenced_nodes = []
    for ila_fused_city in ila_fused_cities:
        links_by_neighbor = defaultdict(list)

        for link in links:
            if (link.from_city == ila_fused_city or link.to_city == ila_fused_city):
                neighbor = (link.to_city if link.from_city == ila_fused_city else link.from_city)
                links_by_neighbor[neighbor].append(link)

        if len(links_by_neighbor) == 0:
            unreferenced_nodes.append(ila_fused_city)
            # collect all mistakes before raising an error
            continue

        if len(links_by_neighbor) != 2:
            neighbor_list = [
                (n.from_city, n.to_city) for n in links if n.from_city == ila_fused_city or n.to_city == ila_fused_city]
            raise NetworkTopologyError(
                f'XLS error: ILA {ila_fused_city} must have exactly two neighbors:\n{_format_items(neighbor_list)}')  # noqa E231

        neighbors = list(links_by_neighbor)

        side_a = sorted(
            links_by_neighbor[neighbors[0]],
            key=lambda link: link.row_id or 0,
        )
        side_b = sorted(
            links_by_neighbor[neighbors[1]],
            key=lambda link: link.row_id or 0,
        )

        if len(side_a) != len(side_b):
            # duplication of links must be the same between two ROADMs
            wrong_duplications.append(ila_fused_city)
            # collect all mistakes before raising an error
            continue

        for pair_id, (link_a, link_b) in enumerate(zip(side_a, side_b)):
            # link may have been processed already
            if link_a.pair_id is None and link_b.pair_id is None:
                link_a.pair_id = str(pair_id)
                link_b.pair_id = str(pair_id)
            elif link_a.pair_id is not None and link_b.pair_id is None:
                # order should be kept
                if link_a.pair_id != str(pair_id):
                    raise ValueError(f'link_a pair_id should be {pair_id}')  # catching code mistake: this should never happen  # noqa E501
                link_b.pair_id = str(pair_id)
            elif link_a.pair_id is None and link_b.pair_id is not None:
                # order should be kept
                if link_b.pair_id != str(pair_id):
                    raise ValueError(f'link_b pair_id should be {pair_id}')  # catching code mistake: this should never happen  # noqa E501
                link_a.pair_id = str(pair_id)

    # All errors have been collected, now raise errors
    if unreferenced_nodes:
        raise NetworkTopologyError(
            'XLS error: The following nodes are not referenced from the Links sheet. '
            + f'If unused, remove them from the Nodes sheet:\n{_format_items(unreferenced_nodes)}')  # noqa E231
    if wrong_duplications:
        raise NetworkTopologyError(
            'XLS error: The following ILA or Fused have different numbers of links on their two sides:\n'
            + f'{_format_items(wrong_duplications)}')


def assign_roadm_degree_pair_ids(roadms: list[Roadm], links: list[Link]) -> None:
    """Assign link pair identifiers to ROADM degree definitions.

    ROADM degrees are matched with links using their endpoints. Both link
    orientations are accepted. When several ROADM definitions or parallel
    links share the same endpoints, entries are matched according to their
    source row order.

    :param roadms: A list of ROADM degree definitions.
    :type roadms: list[Roadm]
    :param links: A list of network links.
    :type links: list[Link]
    """
    links_by_endpoints = defaultdict(list)

    for link in links:
        links_by_endpoints[endpoint_key(link)].append(link)
        links_by_endpoints[rev_endpoint_key(link)].append(link)

    roadm_by_endpoints = defaultdict(list)

    for roadm in roadms:
        key = (roadm.from_node, roadm.to_node)
        roadm_by_endpoints[key].append(roadm)

    for endpoints, roadm_group in roadm_by_endpoints.items():
        link_group = sorted(
            links_by_endpoints.get(endpoints, []),
            key=lambda link: link.row_id or 0)

        roadm_group = sorted(
            roadm_group,
            key=lambda roadm: roadm.row_id or 0)

        for roadm, link in zip(roadm_group, link_group):
            roadm.pair_id = link.pair_id


def duplicate_ila_fused(nodes: list[Node], links: list[Link]) -> None:
    """Create duplicated ILA or FUSED nodes for distinct link pair identifiers.

    An ILA or FUSED node is duplicated when it is associated with several
    different pair identifiers. The original node keeps its first pair
    identifier and additional node instances are created for the others.

    :param nodes: A list of network nodes. New duplicated nodes are appended
        to this list.
    :type nodes: list[Node]
    :param links: A list of network links containing pair identifiers.
    :type links: list[Link]
    """
    nodes_by_city = {n.city: n for n in nodes}
    node_pair_by_city = {}
    for link in links:
        if (link.from_city, link.pair_id) not in node_pair_by_city:
            node_pair_by_city[(link.from_city, link.pair_id)] = nodes_by_city[link.from_city]
        if (link.to_city, link.pair_id) not in node_pair_by_city:
            node_pair_by_city[(link.to_city, link.pair_id)] = nodes_by_city[link.to_city]
    for (city, pair_id), node in node_pair_by_city.items():
        if node.node_type.lower() in ["ila", "fused"]:
            if node.pair_id is None:
                node.pair_id = pair_id
            else:
                duplicate_node = deepcopy(node)
                duplicate_node.pair_id = pair_id
                nodes.append(duplicate_node)


def eqpt_connection_by_city(city_name: str, eqpts_by_city: DefaultDict[str, List[Eqpt]],
                            links_by_city: DefaultDict[str, List[Link]], nodes_by_city: Dict[str, List[Node]]) -> list:
    """Returns the list of equipment installed in the specified city.

    :param city_name: The name of the city to check for equipment.
    :type city_name: str
    :param eqpts_by_city: A defaultdict mapping city names to lists of Eqpt objects.
    :type eqpts_by_city: DefaultDict[str, List[Eqpt]]
    :param links_by_city: A defaultdict mapping city names to lists of Link objects.
    :type links_by_city: DefaultDict[str, List[Link]]
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, List[Node]]
    :return: A list of connection dictionaries for the specified city.
    :rtype: list
    """
    other_cities = fiber_dest_from_source(city_name, links_by_city)
    subdata = []
    if nodes_by_city[city_name][0].node_type.lower() in {'ila', 'fused'}:
        # Then len(other_cities) == 2
        other_cities_by_pair = defaultdict(list)
        for (other_city, link) in other_cities:
            other_cities_by_pair[link.pair_id].append((other_city, link))
        for pair_id, other_cities in other_cities_by_pair.items():
            for i, direction in enumerate(['west', 'east']):
                incoming_city = other_cities[i][0]
                outgoing_city = other_cities[1 - i][0]
                from_ = fiber_link(incoming_city, city_name, links_by_city, pair_id)
                in_ = eqpt_in_city_to_city(city_name, other_cities[0][0], eqpts_by_city, nodes_by_city, pair_id,
                                           direction=direction)
                to_ = fiber_link(city_name, outgoing_city, links_by_city, pair_id)
                subdata += connect_eqpt(from_, in_, to_)

    elif nodes_by_city[city_name][0].node_type.lower() == 'roadm':
        for (other_city, link) in other_cities:
            from_ = nodes_by_city[city_name][0].roadm_uid
            in_ = eqpt_in_city_to_city(city_name, other_city, eqpts_by_city, nodes_by_city, link.pair_id,
                                       direction="east")
            to_ = fiber_link(city_name, other_city, links_by_city, link.pair_id)
            subdata += connect_eqpt(from_, in_, to_)
            degree = in_ if in_ else to_

            from_ = fiber_link(other_city, city_name, links_by_city, link.pair_id)
            in_ = eqpt_in_city_to_city(city_name, other_city, eqpts_by_city, nodes_by_city, link.pair_id,
                                       direction="west")
            to_ = nodes_by_city[city_name][0].roadm_uid
            subdata += connect_eqpt(from_, in_, to_)
            paired_degree = in_ if in_ else from_
            nodes_by_city[city_name][0].degree_association.append((degree, paired_degree))
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


def ila_direction(
    city_name: str,
    neighbor: str,
    pair_id: str,
    eqpts_by_city: DefaultDict[str, List[Eqpt]],
) -> str:
    """
    Détermine la direction de l'équipement ILA correspondant
    au voisin traversé.
    """

    for eqpt in eqpts_by_city.get(city_name, []):
        if eqpt.pair_id == pair_id and eqpt.to_city == neighbor:
            return 'west'

    return 'east'


def eqpt_in_city_to_city(in_city: str, to_city: str,
                         eqpts_by_city: DefaultDict[str, List[Eqpt]], nodes_by_city: Dict[str, List[Node]],
                         pair_id: str, direction: str = 'east') -> str:
    """Returns the formatted string corresponding to in_city types and direction.

    :param in_city: The city where the equipment is located.
    :type in_city: str
    :param to_city: The city where the equipment connects to.
    :type to_city: str
    :param eqpts_by_city: A defaultdict mapping city names to lists of Eqpt objects.
    :type eqpts_by_city: DefaultDict[str, List[Eqpt]]
    :param nodes_by_city: A dictionary mapping city names to Node objects.
    :type nodes_by_city: Dict[str, List[Node]]
    :param direction: The direction of the equipment (default is 'east').
    :type direction: str
    :return: A formatted string representing the equipment in the specified direction.
    :rtype: str
    """
    rev_direction = 'west' if direction == 'east' else 'east'
    return_eqpt = ''
    if in_city in eqpts_by_city:
        for e in eqpts_by_city[in_city]:
            if nodes_by_city[in_city][0].node_type.lower() == 'roadm':
                if e.to_city == to_city and e.pair_id == pair_id:
                    if direction == 'east':
                        return e.east_uid
                    elif direction == 'west' and e.pair_id == pair_id:
                        return e.west_uid
            elif nodes_by_city[in_city][0].node_type.lower() == 'ila':
                if e.to_city != to_city and e.pair_id == pair_id:
                    direction = rev_direction
                if direction == 'east':
                    return e.east_uid
                else:
                    return e.west_uid
    elif nodes_by_city[in_city][0].node_type.lower() in ['ila', 'fused']:
        node = next(n for n in nodes_by_city[in_city] if n.pair_id == pair_id)
        return node.east_uid if direction == 'east' else node.west_uid
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


def fiber_dest_from_source(city_name: str, links_by_city: DefaultDict[str, List[Link]]
                           ) -> List[Tuple[str, Link]]:
    """Return the neighboring cities and corresponding links.

    :param city_name: The name of the city whose neighbors are searched.
    :type city_name: str
    :param links_by_city: A mapping from city names to connected links.
    :type links_by_city: DefaultDict[str, List[Link]]
    :return: A list of tuples containing the neighboring city and the
        corresponding link.
    :rtype: List[Tuple[str, Link]]
    """
    destinations = []
    links_from_city = links_by_city[city_name]
    for link in links_from_city:
        if link.from_city == city_name:
            destinations.append((link.to_city, link))
        else:
            destinations.append((link.from_city, link))
    return destinations


def fiber_link(from_city: str, to_city: str, links_by_city: DefaultDict[str, List[Link]], pair_id) -> str:
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
    link = next(li for li in links
                if li.from_city in source_dest and li.to_city in source_dest and li.pair_id == pair_id)
    if link.from_city == from_city:
        fiber = link.east_uid
    else:
        fiber = link.west_uid
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

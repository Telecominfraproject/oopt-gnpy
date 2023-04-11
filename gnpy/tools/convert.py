#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from xlrd import open_workbook
from logging import getLogger
from argparse import ArgumentParser
from collections import namedtuple, Counter, defaultdict
from itertools import chain
from json import dumps
from pathlib import Path
from copy import copy

from gnpy.core.utils import silent_remove
from gnpy.core.exceptions import NetworkTopologyError
from gnpy.core.elements import Edfa, Fused, Fiber


_logger = getLogger(__name__)


def all_rows(sh, start=0):
    return (sh.row(x) for x in range(start, sh.nrows))


class Node(object):
    def __init__(self, **kwargs):
        super(Node, self).__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
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


class Link(object):
    """attribtes from west parse_ept_headers dict
    +node_a, node_z, west_fiber_con_in, east_fiber_con_in
    """

    def __init__(self, **kwargs):
        super(Link, self).__init__()
        self.update_attr(kwargs)
        self.distance_units = 'km'

    def update_attr(self, kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)
            k = 'west' + k.split('east')[-1]
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)

    def __eq__(self, link):
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
        'east_pmd': 0.1,
        'east_cable': ''
    }


class Eqpt(object):
    def __init__(self, **kwargs):
        super(Eqpt, self).__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            v_east = clean_kwargs.get(k, v)
            setattr(self, k, v_east)
            k = 'west' + k.split('east')[-1]
            v_west = clean_kwargs.get(k, v)
            setattr(self, k, v_west)

    default_values = {
        'from_city': '',
        'to_city': '',
        'east_amp_type': '',
        'east_att_in': 0,
        'east_amp_gain': None,
        'east_amp_dp': None,
        'east_tilt': 0,
        'east_att_out': None
    }


class Roadm(object):
    def __init__(self, **kwargs):
        super(Roadm, self).__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            v = clean_kwargs.get(k, v)
            setattr(self, k, v)

    default_values = {'from_node': '',
                      'to_node': '',
                      'target_pch_out_db': None
                      }


def read_header(my_sheet, line, slice_):
    """ return the list of headers !:= ''
    header_i = [(header, header_column_index), ...]
    in a {line, slice1_x, slice_y} range
    """
    Param_header = namedtuple('Param_header', 'header colindex')
    try:
        header = [x.value.strip() for x in my_sheet.row_slice(line, slice_[0], slice_[1])]
        header_i = [Param_header(header, i + slice_[0]) for i, header in enumerate(header) if header != '']
    except Exception:
        header_i = []
    if header_i != [] and header_i[-1].colindex != slice_[1]:
        header_i.append(Param_header('', slice_[1]))
    return header_i


def read_slice(my_sheet, line, slice_, header):
    """return the slice range of a given header
    in a defined range {line, slice_x, slice_y}"""
    header_i = read_header(my_sheet, line, slice_)
    slice_range = (-1, -1)
    if header_i != []:
        try:
            slice_range = next((h.colindex, header_i[i + 1].colindex)
                               for i, h in enumerate(header_i) if header in h.header)
        except Exception:
            pass
    return slice_range


def parse_headers(my_sheet, input_headers_dict, headers, start_line, slice_in):
    """return a dict of header_slice
    key = column index
    value = header name"""

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
            else:
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
    return {f: r.value for f, r in
            zip([label for label in headers.values()], [row[i] for i in headers])}


def parse_sheet(my_sheet, input_headers_dict, header_line, start_line, column):
    headers = parse_headers(my_sheet, input_headers_dict, {}, header_line, (0, column))
    for row in all_rows(my_sheet, start=start_line):
        yield parse_row(row[0: column], headers)


def _format_items(items):
    return '\n'.join(f' - {item}' for item in items)


def sanity_check(nodes, links, nodes_by_city, links_by_city, eqpts_by_city):

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


def create_roadm_element(node, roadms_by_city):
    """ create the json element for a roadm node, including the different cases:
    - if there are restrictions
    - if there are per degree target power defined on a direction
    direction is defined by the booster name, so that booster must also be created in eqpt sheet
    if the direction is defined in roadm
    """
    roadm = {'uid': f'roadm {node.city}'}
    if node.preamp_restriction != '' or node.booster_restriction != '':
        roadm['params'] = {
            'restrictions': {
               'preamp_variety_list': silent_remove(node.preamp_restriction.split(' | '), ''),
               'booster_variety_list': silent_remove(node.booster_restriction.split(' | '), '')}
                          }
    if node.city in roadms_by_city.keys():
        if 'params' not in roadm.keys():
            roadm['params'] = {}
        roadm['params']['per_degree_pch_out_db'] = {}
        for elem in roadms_by_city[node.city]:
            to_node = f'east edfa in {node.city} to {elem.to_node}'
            if elem.target_pch_out_db is not None:
                roadm['params']['per_degree_pch_out_db'][to_node] = elem.target_pch_out_db
    roadm['metadata'] = {'location': {'city':      node.city,
                                      'region':    node.region,
                                      'latitude':  node.latitude,
                                      'longitude': node.longitude}}
    roadm['type'] = 'Roadm'
    return roadm


def create_east_eqpt_element(node):
    """ create amplifiers json elements for the east direction.
    this includes the case where the case of a fused element defined instead of an
    ILA in eqpt sheet
    """
    eqpt = {'uid': f'east edfa in {node.from_city} to {node.to_city}',
            'metadata': {'location': {'city':      nodes_by_city[node.from_city].city,
                                      'region':    nodes_by_city[node.from_city].region,
                                      'latitude':  nodes_by_city[node.from_city].latitude,
                                      'longitude': nodes_by_city[node.from_city].longitude}}}
    if node.east_amp_type.lower() != '' and node.east_amp_type.lower() != 'fused':
        eqpt['type'] = 'Edfa'
        eqpt['type_variety'] = f'{node.east_amp_type}'
        eqpt['operational'] = {'gain_target': node.east_amp_gain,
                               'delta_p':     node.east_amp_dp,
                               'tilt_target': node.east_tilt,
                               'out_voa':     node.east_att_out}
    elif node.east_amp_type.lower() == '':
        eqpt['type'] = 'Edfa'
        eqpt['operational'] = {'gain_target': node.east_amp_gain,
                               'delta_p':     node.east_amp_dp,
                               'tilt_target': node.east_tilt,
                               'out_voa':     node.east_att_out}
    elif node.east_amp_type.lower() == 'fused':
        # fused edfa variety is a hack to indicate that there should not be
        # booster amplifier out the roadm.
        # If user specifies ILA in Nodes sheet and fused in Eqpt sheet, then assumes that
        # this is a fused nodes.
        eqpt['type'] = 'Fused'
        eqpt['params'] = {'loss': 0}
    return eqpt


def create_west_eqpt_element(node):
    """ create amplifiers json elements for the west direction.
    this includes the case where the case of a fused element defined instead of an
    ILA in eqpt sheet
    """
    eqpt = {'uid': f'west edfa in {node.from_city} to {node.to_city}',
            'metadata': {'location': {'city':      nodes_by_city[node.from_city].city,
                                      'region':    nodes_by_city[node.from_city].region,
                                      'latitude':  nodes_by_city[node.from_city].latitude,
                                      'longitude': nodes_by_city[node.from_city].longitude}},
            'type': 'Edfa'}
    if node.west_amp_type.lower() != '' and node.west_amp_type.lower() != 'fused':
        eqpt['type_variety'] = f'{node.west_amp_type}'
        eqpt['operational'] = {'gain_target': node.west_amp_gain,
                               'delta_p':     node.west_amp_dp,
                               'tilt_target': node.west_tilt,
                               'out_voa':     node.west_att_out}
    elif node.west_amp_type.lower() == '':
        eqpt['operational'] = {'gain_target': node.west_amp_gain,
                               'delta_p':     node.west_amp_dp,
                               'tilt_target': node.west_tilt,
                               'out_voa':     node.west_att_out}
    elif node.west_amp_type.lower() == 'fused':
        eqpt['type'] = 'Fused'
        eqpt['params'] = {'loss': 0}
    return eqpt

def xls_to_json_data(input_filename, filter_region=[]):
    nodes, links, eqpts, roadms = parse_excel(input_filename)
    if filter_region:
        nodes = [n for n in nodes if n.region.lower() in filter_region]
        cities = {n.city for n in nodes}
        links = [lnk for lnk in links if lnk.from_city in cities and
                 lnk.to_city in cities]
        cities = {lnk.from_city for lnk in links} | {lnk.to_city for lnk in links}
        nodes = [n for n in nodes if n.city in cities]

    global nodes_by_city
    nodes_by_city = {n.city: n for n in nodes}

    global links_by_city
    links_by_city = defaultdict(list)
    for link in links:
        links_by_city[link.from_city].append(link)
        links_by_city[link.to_city].append(link)

    global eqpts_by_city
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
             for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'] +
            [create_roadm_element(x, roadms_by_city)
             for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'] +
            [{'uid': f'west fused spans in {x.city}',
              'metadata': {'location': {'city': x.city,
                                        'region': x.region,
                                        'latitude': x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Fused'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'fused'] +
            [{'uid': f'east fused spans in {x.city}',
              'metadata': {'location': {'city': x.city,
                                        'region': x.region,
                                        'latitude': x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Fused'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'fused'] +
            [{'uid': f'fiber ({x.from_city} \u2192 {x.to_city})-{x.east_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.east_fiber,
              'params': {'length': round(x.east_distance, 3),
                         'length_units': x.distance_units,
                         'loss_coef': x.east_lineic,
                         'con_in': x.east_con_in,
                         'con_out': x.east_con_out}
              }
             for x in links] +
            [{'uid': f'fiber ({x.to_city} \u2192 {x.from_city})-{x.west_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.west_fiber,
              'params': {'length': round(x.west_distance, 3),
                         'length_units': x.distance_units,
                         'loss_coef': x.west_lineic,
                         'con_in': x.west_con_in,
                         'con_out': x.west_con_out}
              } for x in links] +
            [{'uid': f'west edfa in {x.city}',
              'metadata': {'location': {'city': x.city,
                                        'region': x.region,
                                        'latitude': x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Edfa',
              'operational': {'gain_target': None,
                              'tilt_target': 0}
              } for x in nodes_by_city.values() if x.node_type.lower() == 'ila' and x.city not in eqpts_by_city] +
            [{'uid': f'east edfa in {x.city}',
              'metadata': {'location': {'city': x.city,
                                        'region': x.region,
                                        'latitude': x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Edfa',
              'operational': {'gain_target': None,
                              'tilt_target': 0}
              } for x in nodes_by_city.values() if x.node_type.lower() == 'ila' and x.city not in eqpts_by_city] +
            [create_east_eqpt_element(e) for e in eqpts] +
            [create_west_eqpt_element(e) for e in eqpts],
        'connections':
            list(chain.from_iterable([eqpt_connection_by_city(n.city)
                                      for n in nodes]))
            +
            list(chain.from_iterable(zip(
                [{'from_node': f'trx {x.city}',
                  'to_node': f'roadm {x.city}'}
                 for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'],
                [{'from_node': f'roadm {x.city}',
                  'to_node': f'trx {x.city}'}
                 for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'])))
    }


def convert_file(input_filename, filter_region=[], output_json_file_name=None):
    data = xls_to_json_data(input_filename, filter_region)
    if output_json_file_name is None:
        output_json_file_name = input_filename.with_suffix('.json')
    with open(output_json_file_name, 'w', encoding='utf-8') as edfa_json_file:
        edfa_json_file.write(dumps(data, indent=2, ensure_ascii=False))
        edfa_json_file.write('\n')   # add end of file newline because json dumps does not.
    return output_json_file_name


def corresp_names(input_filename, network):
    """ a function that builds the correspondance between names given in the excel,
        and names used in the json, and created by the autodesign.
        All names are listed
    """
    nodes, links, eqpts, roadms = parse_excel(input_filename)
    fused = [n.uid for n in network.nodes() if isinstance(n, Fused)]
    ila = [n.uid for n in network.nodes() if isinstance(n, Edfa)]

    corresp_roadm = {x.city: [f'roadm {x.city}'] for x in nodes
                     if x.node_type.lower() == 'roadm'}
    corresp_fused = {x.city: [f'west fused spans in {x.city}', f'east fused spans in {x.city}']
                     for x in nodes if x.node_type.lower() == 'fused' and
                     f'west fused spans in {x.city}' in fused and
                     f'east fused spans in {x.city}' in fused}

    # add the special cases when an ila is changed into a fused
    for my_e in eqpts:
        name = f'east edfa in {my_e.from_city} to {my_e.to_city}'
        if my_e.east_amp_type.lower() == 'fused' and name in fused:
            if my_e.from_city in corresp_fused.keys():
                corresp_fused[my_e.from_city].append(name)
            else:
                corresp_fused[my_e.from_city] = [name]
        name = f'west edfa in {my_e.from_city} to {my_e.to_city}'
        if my_e.west_amp_type.lower() == 'fused' and name in fused:
            if my_e.from_city in corresp_fused.keys():
                corresp_fused[my_e.from_city].append(name)
            else:
                corresp_fused[my_e.from_city] = [name]
    # build corresp ila based on eqpt sheet
    # start with east direction
    corresp_ila = {e.from_city: [f'east edfa in {e.from_city} to {e.to_city}']
                   for e in eqpts if f'east edfa in {e.from_city} to {e.to_city}' in ila}
    # west direction, append name or create a new item in dict
    for my_e in eqpts:
        name = f'west edfa in {my_e.from_city} to {my_e.to_city}'
        if name in ila:
            if my_e.from_city in corresp_ila.keys():
                corresp_ila[my_e.from_city].append(name)
            else:
                corresp_ila[my_e.from_city] = [name]
    # complete with potential autodesign names: amplifiers
    for my_l in links:
        name = f'Edfa0_fiber ({my_l.to_city} \u2192 {my_l.from_city})-{my_l.west_cable}'
        if name in ila:
            if my_l.from_city in corresp_ila.keys():
                # "east edfa in Stbrieuc to Rennes_STA"  is equivalent name as
                # "Edfa0_fiber (Lannion_CAS → Stbrieuc)-F056"
                # "west edfa in Stbrieuc to Rennes_STA"  is equivalent name as
                # "Edfa0_fiber (Rennes_STA → Stbrieuc)-F057"
                # does not filter names: all types (except boosters) are created.
                # in case fibers are splitted the name here is a prefix
                corresp_ila[my_l.from_city].append(name)
            else:
                corresp_ila[my_l.from_city] = [name]
        name = f'Edfa0_fiber ({my_l.from_city} \u2192 {my_l.to_city})-{my_l.east_cable}'
        if name in ila:
            if my_l.to_city in corresp_ila.keys():
                corresp_ila[my_l.to_city].append(name)
            else:
                corresp_ila[my_l.to_city] = [name]
    # merge fused with ila:
    for key, val in corresp_fused.items():
        if key in corresp_ila.keys():
            corresp_ila[key].extend(val)
        else:
            corresp_ila[key] = val
        # no need of roadm booster
    return corresp_roadm, corresp_fused, corresp_ila


def parse_excel(input_filename):
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
            'att_in': 'east_att_in',
            'amp gain': 'east_amp_gain',
            'delta p': 'east_amp_dp',
            'tilt': 'east_tilt',
            'att_out': 'east_att_out'
        },
        'west': {
            'amp type': 'west_amp_type',
            'att_in': 'west_att_in',
            'amp gain': 'west_amp_gain',
            'delta p': 'west_amp_dp',
            'tilt': 'west_tilt',
            'att_out': 'west_att_out'
        }
    }
    roadm_headers = {'Node A': 'from_node',
                     'Node Z': 'to_node',
                     'per degree target power (dBm)': 'target_pch_out_db'
                     }

    with open_workbook(input_filename) as wb:
        nodes_sheet = wb.sheet_by_name('Nodes')
        links_sheet = wb.sheet_by_name('Links')
        try:
            eqpt_sheet = wb.sheet_by_name('Eqpt')
        except Exception:
            # eqpt_sheet is optional
            eqpt_sheet = None
        try:
            roadm_sheet = wb.sheet_by_name('Roadms')
        except Exception:
            # roadm_sheet is optional
            roadm_sheet = None

        nodes = []
        for node in parse_sheet(nodes_sheet, node_headers, NODES_LINE, NODES_LINE + 1, NODES_COLUMN):
            nodes.append(Node(**node))
        expected_node_types = {'ROADM', 'ILA', 'FUSED'}
        for n in nodes:
            if n.node_type not in expected_node_types:
                n.node_type = 'ILA'

        links = []
        for link in parse_sheet(links_sheet, link_headers, LINKS_LINE, LINKS_LINE + 2, LINKS_COLUMN):
            links.append(Link(**link))

        eqpts = []
        if eqpt_sheet is not None:
            for eqpt in parse_sheet(eqpt_sheet, eqpt_headers, EQPTS_LINE, EQPTS_LINE + 2, EQPTS_COLUMN):
                eqpts.append(Eqpt(**eqpt))

        roadms = []
        if roadm_sheet is not None:
            for roadm in parse_sheet(roadm_sheet, roadm_headers, ROADMS_LINE, ROADMS_LINE+2, ROADMS_COLUMN):
                roadms.append(Roadm(**roadm))

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


def eqpt_connection_by_city(city_name):
    other_cities = fiber_dest_from_source(city_name)
    subdata = []
    if nodes_by_city[city_name].node_type.lower() in {'ila', 'fused'}:
        # Then len(other_cities) == 2
        direction = ['west', 'east']
        for i in range(2):
            from_ = fiber_link(other_cities[i], city_name)
            in_ = eqpt_in_city_to_city(city_name, other_cities[0], direction[i])
            to_ = fiber_link(city_name, other_cities[1 - i])
            subdata += connect_eqpt(from_, in_, to_)
    elif nodes_by_city[city_name].node_type.lower() == 'roadm':
        for other_city in other_cities:
            from_ = f'roadm {city_name}'
            in_ = eqpt_in_city_to_city(city_name, other_city)
            to_ = fiber_link(city_name, other_city)
            subdata += connect_eqpt(from_, in_, to_)

            from_ = fiber_link(other_city, city_name)
            in_ = eqpt_in_city_to_city(city_name, other_city, "west")
            to_ = f'roadm {city_name}'
            subdata += connect_eqpt(from_, in_, to_)
    return subdata


def connect_eqpt(from_, in_, to_):
    connections = []
    if in_ != '':
        connections = [{'from_node': from_, 'to_node': in_},
                       {'from_node': in_, 'to_node': to_}]
    else:
        connections = [{'from_node': from_, 'to_node': to_}]
    return connections


def eqpt_in_city_to_city(in_city, to_city, direction='east'):
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


def corresp_next_node(network, corresp_ila, corresp_roadm):
    """ for each name in corresp dictionnaries find the next node in network and its name
        given by user in excel. for meshTopology_exampleV2.xls:
        user ILA name Stbrieuc covers the two direction. convert.py creates 2 different ILA
        with possible names (depending on the direction and if the eqpt was defined in eqpt
        sheet)
        - east edfa in Stbrieuc to Rennes_STA
        - west edfa in Stbrieuc to Rennes_STA
        - Edfa0_fiber (Lannion_CAS → Stbrieuc)-F056
        - Edfa0_fiber (Rennes_STA → Stbrieuc)-F057
        next_nodes finds the user defined name of next node to be able to map the path constraints
        - east edfa in Stbrieuc to Rennes_STA      next node = Rennes_STA
        - west edfa in Stbrieuc to Rennes_STA      next node Lannion_CAS

        Edfa0_fiber (Lannion_CAS → Stbrieuc)-F056 and Edfa0_fiber (Rennes_STA → Stbrieuc)-F057
        do not exist
        the function supports fiber splitting, fused nodes and shall only be called if
        excel format is used for both network and service
    """
    next_node = {}
    # consolidate tables and create next_node table
    for ila_key, ila_list in corresp_ila.items():
        temp = copy(ila_list)
        for ila_elem in ila_list:
            # find the node with ila_elem string _in_ the node uid. 'in' is used instead of
            # '==' to find composed nodes due to fiber splitting in autodesign.
            # eg if elem_ila is 'Edfa0_fiber (Lannion_CAS → Stbrieuc)-F056',
            # node uid 'Edfa0_fiber (Lannion_CAS → Stbrieuc)-F056_(1/2)' is possible
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
            if correct_ila_name not in next_node.keys():
                for key, val in corresp_ila.items():
                    # in case of splitted fibers the ila name might not be exact match
                    if [e for e in val if e in next_nd.uid]:
                        next_node[correct_ila_name] = key
                        break

        corresp_ila[ila_key] = temp
    return corresp_ila, next_node


def fiber_dest_from_source(city_name):
    destinations = []
    links_from_city = links_by_city[city_name]
    for l in links_from_city:
        if l.from_city == city_name:
            destinations.append(l.to_city)
        else:
            destinations.append(l.from_city)
    return destinations


def fiber_link(from_city, to_city):
    source_dest = (from_city, to_city)
    links = links_by_city[from_city]
    link = next(l for l in links if l.from_city in source_dest and l.to_city in source_dest)
    if link.from_city == from_city:
        fiber = f'fiber ({link.from_city} \u2192 {link.to_city})-{link.east_cable}'
    else:
        fiber = f'fiber ({link.to_city} \u2192 {link.from_city})-{link.west_cable}'
    return fiber


def midpoint(city_a, city_b):
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
ROADMS_COLUMN = 3


def _do_convert():
    parser = ArgumentParser()
    parser.add_argument('workbook', type=Path)
    parser.add_argument('-f', '--filter-region', action='append', default=[])
    parser.add_argument('--output', type=Path, help='Name of the generated JSON file')
    args = parser.parse_args()
    res = convert_file(args.workbook, args.filter_region, args.output)
    print(f'XLS -> JSON saved to {res}')


if __name__ == '__main__':
    _do_convert()

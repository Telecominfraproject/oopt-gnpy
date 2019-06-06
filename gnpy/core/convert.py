#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.convert
=================

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

from sys import exit
try:
    from xlrd import open_workbook
except ModuleNotFoundError:
    exit('Required: `pip install xlrd`')
from argparse import ArgumentParser
from collections import namedtuple, Counter, defaultdict
from itertools import chain
from json import dumps
from pathlib import Path
from difflib import get_close_matches
from gnpy.core.utils import silent_remove
import time

all_rows = lambda sh, start=0: (sh.row(x) for x in range(start, sh.nrows))

class Node(object):
    def __init__(self, **kwargs):
        super(Node, self).__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
        clean_kwargs = {k:v for k,v in kwargs.items() if v !=''}
        for k,v in self.default_values.items():
            v = clean_kwargs.get(k,v)
            setattr(self, k, v)

    default_values = \
    {
        'city':         '',
        'state':        '',
        'country':      '',
        'region':       '',
        'latitude':     0,
        'longitude':    0,
        'node_type':    'ILA',
        'booster_restriction' : '',
        'preamp_restriction'  : ''
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
        clean_kwargs = {k:v for k,v in kwargs.items() if v !=''}
        for k,v in self.default_values.items():
            v = clean_kwargs.get(k,v)
            setattr(self, k, v)
            k = 'west' + k.split('east')[-1]
            v = clean_kwargs.get(k,v)
            setattr(self, k, v)

    def __eq__(self, link):
        return (self.from_city == link.from_city and self.to_city == link.to_city) \
                or (self.from_city == link.to_city and self.to_city == link.from_city)

    default_values = \
    {
            'from_city':            '',
            'to_city':              '',
            'east_distance':        80,
            'east_fiber':           'SSMF',
            'east_lineic':          0.2,
            'east_con_in':          None,
            'east_con_out':         None,
            'east_pmd':             0.1,
            'east_cable':           ''
    }


class Eqpt(object):
    def __init__(self, **kwargs):
        super(Eqpt, self).__init__()
        self.update_attr(kwargs)

    def update_attr(self, kwargs):
        clean_kwargs = {k:v for k,v in kwargs.items() if v !=''}
        for k,v in self.default_values.items():
            v_east = clean_kwargs.get(k,v)
            setattr(self, k, v_east)
            k = 'west' + k.split('east')[-1]
            v_west = clean_kwargs.get(k,v)
            setattr(self, k, v_west)

    default_values = \
    {
            'from_city':        '',
            'to_city':          '',
            'east_amp_type':    '',
            'east_att_in':      0,
            'east_amp_gain':    None,
            'east_amp_dp':      None,
            'east_tilt':        0,
            'east_att_out':     None
    }


def read_header(my_sheet, line, slice_):
    """ return the list of headers !:= ''
    header_i = [(header, header_column_index), ...]
    in a {line, slice1_x, slice_y} range
    """
    Param_header = namedtuple('Param_header', 'header colindex')
    try:
        header = [x.value.strip() for x in my_sheet.row_slice(line, slice_[0], slice_[1])]
        header_i = [Param_header(header,i+slice_[0]) for i, header in enumerate(header) if header != '']
    except Exception:
        header_i = []
    if header_i != [] and header_i[-1].colindex != slice_[1]:
        header_i.append(Param_header('',slice_[1]))
    return header_i

def read_slice(my_sheet, line, slice_, header):
    """return the slice range of a given header
    in a defined range {line, slice_x, slice_y}"""
    header_i = read_header(my_sheet, line, slice_)
    slice_range = (-1,-1)
    if header_i != []:
        try:
            slice_range = next((h.colindex,header_i[i+1].colindex) \
                for i,h in enumerate(header_i) if header in h.header)
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
        while slice_out == (-1,-1) and iteration < 10:
            #try next lines
            #print(h0, iteration)
            slice_out = read_slice(my_sheet, start_line+iteration, slice_in, h0)
            iteration += 1
        if slice_out == (-1, -1):
            if h0 in ('east', 'Node A', 'Node Z', 'City') :
                print(f'\x1b[1;31;40m'+f'CRITICAL: missing _{h0}_ header: EXECUTION ENDS'+ '\x1b[0m')
                exit()
            else:
                print(f'missing header {h0}')
        elif not isinstance(input_headers_dict[h0], dict):
            headers[slice_out[0]] = input_headers_dict[h0]
        else:
            headers = parse_headers(my_sheet, input_headers_dict[h0], headers, start_line+1, slice_out)
    if headers == {}:
        print(f'\x1b[1;31;40m'+f'CRITICAL ERROR: could not find any header to read _ ABORT'+ '\x1b[0m')
        exit()
    return headers

def parse_row(row, headers):
    #print([label for label in ept.values()])
    #print([i for i in ept.keys()])
    #print(row[i for i in ept.keys()])
    return {f: r.value for f, r in \
            zip([label for label in headers.values()], [row[i] for i in headers])}
            #if r.ctype != XL_CELL_EMPTY}

def parse_sheet(my_sheet, input_headers_dict, header_line, start_line, column):
    headers = parse_headers(my_sheet, input_headers_dict, {}, header_line, (0,column))
    for row in all_rows(my_sheet, start=start_line):
        yield parse_row(row[0: column], headers)

def sanity_check(nodes, links, nodes_by_city, links_by_city, eqpts_by_city):

    duplicate_links = []
    for l1 in links:
        for l2 in links:
            if l1 is not l2 and l1 == l2 and l2 not in duplicate_links:
                print(f'\nWARNING\n \
                    link {l1.from_city}-{l1.to_city} is duplicate \
                    \nthe 1st duplicate link will be removed but you should check Links sheet input')
                duplicate_links.append(l1)
    #if duplicate_links != []:
        #time.sleep(3)
    for l in duplicate_links:
        links.remove(l)

    try :
        test_nodes = [n for n in nodes_by_city if not n in links_by_city]
        test_links = [n for n in links_by_city if not n in nodes_by_city]
        test_eqpts = [n for n in eqpts_by_city if not n in nodes_by_city]
        assert (test_nodes == [] or test_nodes == [''])\
                and (test_links == [] or test_links ==[''])\
                and (test_eqpts == [] or test_eqpts ==[''])
    except AssertionError:
        print(f'CRITICAL error: \nNames in Nodes and Links sheets do no match, check:\
            \n{test_nodes} in Nodes sheet\
            \n{test_links} in Links sheet\
            \n{test_eqpts} in Eqpt sheet')
        exit(1)

    for city,link in links_by_city.items():
        if nodes_by_city[city].node_type.lower()=='ila' and len(link) != 2:
            #wrong input: ILA sites can only be Degree 2
            # => correct to make it a ROADM and remove entry in links_by_city
            #TODO : put in log rather than print
            print(f'invalid node type ({nodes_by_city[city].node_type})\
 specified in {city}, replaced by ROADM')
            nodes_by_city[city].node_type = 'ROADM'
            for n in nodes:
                if n.city==city:
                    n.node_type='ROADM'
    return nodes, links

def convert_file(input_filename, names_matching=False, filter_region=[]):
    nodes, links, eqpts = parse_excel(input_filename)
    if filter_region:
        nodes = [n for n in nodes if n.region.lower() in filter_region]
        cities = {n.city for n in nodes}
        links = [lnk for lnk in links if lnk.from_city in cities and
                                         lnk.to_city in cities]
        cities = {lnk.from_city for lnk in links} | {lnk.to_city for lnk in links}
        nodes = [n for n in nodes if n.city in cities]

    global nodes_by_city
    nodes_by_city = {n.city: n for n in nodes}
    #create matching dictionary for node name mismatch analysis

    cities = {''.join(c.strip() for c in n.city.split('C+L')).lower(): n.city for n in nodes}
    cities_to_match = [k for k in cities]
    city_match_dic = defaultdict(list)
    for city in cities:
        if city in cities_to_match:
            cities_to_match.remove(city)
        matches = get_close_matches(city, cities_to_match, 4, 0.85)
        for m in matches:
            city_match_dic[cities[city]].append(cities[m])
    #check lower case/upper case
    for city in nodes_by_city:
        for match_city in nodes_by_city:
            if match_city.lower() == city.lower() and match_city != city:
                city_match_dic[city].append(match_city)

    if names_matching:
        print('\ncity match dictionary:',city_match_dic)
    with  open('name_match_dictionary.json', 'w', encoding='utf-8') as city_match_dic_file:
        city_match_dic_file.write(dumps(city_match_dic, indent=2, ensure_ascii=False))

    global links_by_city
    links_by_city = defaultdict(list)
    for link in links:
        links_by_city[link.from_city].append(link)
        links_by_city[link.to_city].append(link)

    global eqpts_by_city
    eqpts_by_city = defaultdict(list)
    for eqpt in eqpts:
        eqpts_by_city[eqpt.from_city].append(eqpt)

    nodes, links = sanity_check(nodes, links, nodes_by_city, links_by_city, eqpts_by_city)

    data = {
        'elements':
            [{'uid': f'trx {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Transceiver'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'] +
            [{'uid': f'roadm {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Roadm'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'roadm' \
                 and x.booster_restriction == '' and x.preamp_restriction == ''] +
            [{'uid': f'roadm {x.city}',
              'params' : {
                'restrictions': {
                  'preamp_variety_list': silent_remove(x.preamp_restriction.split(' | '),''),
                  'booster_variety_list': silent_remove(x.booster_restriction.split(' | '),'')
                  }
              },
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Roadm'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'roadm' and \
                 (x.booster_restriction != '' or x.preamp_restriction != '')] +
            [{'uid': f'west fused spans in {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Fused'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'fused'] +
            [{'uid': f'east fused spans in {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Fused'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'fused'] +
            [{'uid': f'fiber ({x.from_city} \u2192 {x.to_city})-{x.east_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.east_fiber,
              'params': {'length':   round(x.east_distance, 3),
                         'length_units':    x.distance_units,
                         'loss_coef': x.east_lineic,
                         'con_in':x.east_con_in,
                         'con_out':x.east_con_out}
            }
              for x in links] +
            [{'uid': f'fiber ({x.to_city} \u2192 {x.from_city})-{x.west_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.west_fiber,
              'params': {'length':   round(x.west_distance, 3),
                         'length_units':    x.distance_units,
                         'loss_coef': x.west_lineic,
                         'con_in':x.west_con_in,
                         'con_out':x.west_con_out}
            } # missing ILA construction
              for x in links] +
            [{'uid': f'east edfa in {e.from_city} to {e.to_city}',
              'metadata': {'location': {'city':      nodes_by_city[e.from_city].city,
                                        'region':    nodes_by_city[e.from_city].region,
                                        'latitude':  nodes_by_city[e.from_city].latitude,
                                        'longitude': nodes_by_city[e.from_city].longitude}},
              'type': 'Edfa',
              'type_variety': e.east_amp_type,
              'operational': {'gain_target': e.east_amp_gain,
                              'delta_p':     e.east_amp_dp,
                              'tilt_target': e.east_tilt,
                              'out_voa'    : e.east_att_out}
             }
             for e in eqpts if (e.east_amp_type.lower() != '' and \
                                e.east_amp_type.lower() != 'fused')] +
            [{'uid': f'west edfa in {e.from_city} to {e.to_city}',
              'metadata': {'location': {'city':      nodes_by_city[e.from_city].city,
                                        'region':    nodes_by_city[e.from_city].region,
                                        'latitude':  nodes_by_city[e.from_city].latitude,
                                        'longitude': nodes_by_city[e.from_city].longitude}},
              'type': 'Edfa',
              'type_variety': e.west_amp_type,
              'operational': {'gain_target': e.west_amp_gain,
                              'delta_p':     e.west_amp_dp,
                              'tilt_target': e.west_tilt,
                              'out_voa'    : e.west_att_out}
             }
             for e in eqpts if (e.west_amp_type.lower() != '' and \
                                e.west_amp_type.lower() != 'fused')] +
            # fused edfa variety is a hack to indicate that there should not be
            # booster amplifier out the roadm.
            # If user specifies ILA in Nodes sheet and fused in Eqpt sheet, then assumes that
            # this is a fused nodes.
            [{'uid': f'east edfa in {e.from_city} to {e.to_city}',
              'metadata': {'location': {'city':      nodes_by_city[e.from_city].city,
                                        'region':    nodes_by_city[e.from_city].region,
                                        'latitude':  nodes_by_city[e.from_city].latitude,
                                        'longitude': nodes_by_city[e.from_city].longitude}},
              'type': 'Fused',
              'params': {'loss': 0}
             }
             for e in eqpts if e.east_amp_type.lower() == 'fused'] +
            [{'uid': f'west edfa in {e.from_city} to {e.to_city}',
              'metadata': {'location': {'city':      nodes_by_city[e.from_city].city,
                                        'region':    nodes_by_city[e.from_city].region,
                                        'latitude':  nodes_by_city[e.from_city].latitude,
                                        'longitude': nodes_by_city[e.from_city].longitude}},
              'type': 'Fused',
              'params': {'loss': 0}
             }
             for e in eqpts if e.west_amp_type.lower() == 'fused'],
        'connections':
            list(chain.from_iterable([eqpt_connection_by_city(n.city)
            for n in nodes]))
            +
            list(chain.from_iterable(zip(
            [{'from_node': f'trx {x.city}',
              'to_node':   f'roadm {x.city}'}
             for x in nodes_by_city.values() if x.node_type.lower()=='roadm'],
            [{'from_node': f'roadm {x.city}',
              'to_node':   f'trx {x.city}'}
             for x in nodes_by_city.values() if x.node_type.lower()=='roadm'])))
    }

    suffix_filename = str(input_filename.suffixes[0])
    full_input_filename = str(input_filename)
    split_filename = [full_input_filename[0:len(full_input_filename)-len(suffix_filename)] , suffix_filename[1:]]
    output_json_file_name = split_filename[0]+'.json'
    with  open(output_json_file_name, 'w', encoding='utf-8') as edfa_json_file:
        edfa_json_file.write(dumps(data, indent=2, ensure_ascii=False))
    return output_json_file_name

def parse_excel(input_filename):
    link_headers = \
    {  'Node A': 'from_city',
       'Node Z': 'to_city',
       'east':{
            'Distance (km)':        'east_distance',
            'Fiber type':           'east_fiber',
            'lineic att':           'east_lineic',
            'Con_in':               'east_con_in',
            'Con_out':              'east_con_out',
            'PMD':                  'east_pmd',
            'Cable id':             'east_cable'
        },
        'west':{
            'Distance (km)':        'west_distance',
            'Fiber type':           'west_fiber',
            'lineic att':           'west_lineic',
            'Con_in':               'west_con_in',
            'Con_out':              'west_con_out',
            'PMD':                  'west_pmd',
            'Cable id':             'west_cable'
        }
    }
    node_headers = \
    {   'City':         'city',
        'State':        'state',
        'Country':      'country',
        'Region':       'region',
        'Latitude':     'latitude',
        'Longitude':    'longitude',
        'Type':         'node_type',
        'Booster_restriction': 'booster_restriction',
        'Preamp_restriction': 'preamp_restriction'
    }
    eqpt_headers = \
    {  'Node A': 'from_city',
       'Node Z': 'to_city',
       'east':{
            'amp type':         'east_amp_type',
            'att_in':           'east_att_in',
            'amp gain':         'east_amp_gain',
            'delta p':          'east_amp_dp',
            'tilt':             'east_tilt',
            'att_out':          'east_att_out'
       },
       'west':{
            'amp type':         'west_amp_type',
            'att_in':           'west_att_in',
            'amp gain':         'west_amp_gain',
            'delta p':          'west_amp_dp',
            'tilt':             'west_tilt',
            'att_out':          'west_att_out'
       }
    }

    with open_workbook(input_filename) as wb:
        nodes_sheet = wb.sheet_by_name('Nodes')
        links_sheet = wb.sheet_by_name('Links')
        try:
            eqpt_sheet = wb.sheet_by_name('Eqpt')
        except Exception:
            #eqpt_sheet is optional
            eqpt_sheet = None

        nodes = []
        for node in parse_sheet(nodes_sheet, node_headers, NODES_LINE, NODES_LINE+1, NODES_COLUMN):
            nodes.append(Node(**node))
        expected_node_types = {'ROADM', 'ILA', 'FUSED'}
        for n in nodes:
            if n.node_type not in expected_node_types:
                n.node_type = 'ILA'

        links = []
        for link in parse_sheet(links_sheet, link_headers, LINKS_LINE, LINKS_LINE+2, LINKS_COLUMN):
            links.append(Link(**link))
        #print('\n', [l.__dict__ for l in links])

        eqpts = []
        if eqpt_sheet != None:
            for eqpt in parse_sheet(eqpt_sheet, eqpt_headers, EQPTS_LINE, EQPTS_LINE+2, EQPTS_COLUMN):
                eqpts.append(Eqpt(**eqpt))

    # sanity check
    all_cities = Counter(n.city for n in nodes)
    if len(all_cities) != len(nodes):
        raise ValueError(f'Duplicate city: {all_cities}')
    if any(ln.from_city not in all_cities or
           ln.to_city   not in all_cities for ln in links):
        raise ValueError(f'Bad link.')

    return nodes, links, eqpts


def eqpt_connection_by_city(city_name):
    other_cities = fiber_dest_from_source(city_name)
    subdata = []
    if nodes_by_city[city_name].node_type.lower() in {'ila', 'fused'}:
        # Then len(other_cities) == 2
        direction = ['west', 'east']
        for i in range(2):
            from_ = fiber_link(other_cities[i], city_name)
            in_ = eqpt_in_city_to_city(city_name, other_cities[0],direction[i])
            to_ = fiber_link(city_name, other_cities[1-i])
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
    if in_ !='':
        connections = [{'from_node': from_, 'to_node': in_},
                      {'from_node': in_, 'to_node': to_}]
    else:
        connections = [{'from_node': from_, 'to_node': to_}]
    return connections


def eqpt_in_city_to_city(in_city, to_city, direction='east'):
    rev_direction = 'west' if direction == 'east' else 'east'
    amp_direction = f'{direction}_amp_type'
    amp_rev_direction = f'{rev_direction}_amp_type'
    return_eqpt = ''
    if in_city in eqpts_by_city:
        for e in eqpts_by_city[in_city]:
            if nodes_by_city[in_city].node_type.lower() == 'roadm':
                if e.to_city == to_city and getattr(e, amp_direction) != '':
                    return_eqpt = f'{direction} edfa in {e.from_city} to {e.to_city}'
            elif nodes_by_city[in_city].node_type.lower() == 'ila':
                if e.to_city != to_city:
                    direction = rev_direction
                    amp_direction = amp_rev_direction
                if getattr(e, amp_direction) != '':
                    return_eqpt = f'{direction} edfa in {e.from_city} to {e.to_city}'
    if nodes_by_city[in_city].node_type.lower() == 'fused':
        return_eqpt = f'{direction} fused spans in {in_city}'
    return return_eqpt


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
    link = links_by_city[from_city]
    l = next(l for l in link if l.from_city in source_dest and l.to_city in source_dest)
    if l.from_city == from_city:
        fiber = f'fiber ({l.from_city} \u2192 {l.to_city})-{l.east_cable}'
    else:
        fiber = f'fiber ({l.to_city} \u2192 {l.from_city})-{l.west_cable}'
    return fiber


def midpoint(city_a, city_b):
    lats  = city_a.latitude, city_b.latitude
    longs = city_a.longitude, city_b.longitude
    try:
        result = {
        'latitude':  sum(lats)  / 2,
        'longitude': sum(longs) / 2
                }
    except :
        result = {
        'latitude':  0,
        'longitude': 0
                }
    return result

#output_json_file_name = 'coronet_conus_example.json'
#TODO get column size automatically from tupple size

NODES_COLUMN = 10
NODES_LINE = 4
LINKS_COLUMN = 16
LINKS_LINE = 3
EQPTS_LINE = 3
EQPTS_COLUMN = 14
parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', type=Path , default='meshTopologyExampleV2.xls')
parser.add_argument('-f', '--filter-region', action='append', default=[])

if __name__ == '__main__':
    args = parser.parse_args()
    convert_file(args.workbook, args.filter_region)

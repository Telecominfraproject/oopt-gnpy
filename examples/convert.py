#!/usr/bin/env python3

from sys import exit
try:
    from xlrd import open_workbook
except ModuleNotFoundError:
    exit('Required: `pip install xlrd`')
from argparse import ArgumentParser
from collections import namedtuple, Counter, defaultdict
from itertools import chain
from json import dumps
from uuid import uuid4
import math
import numpy as np

#output_json_file_name = 'coronet_conus_example.json'
parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', default='meshTopologyExampleV2.xls')
parser.add_argument('-f', '--filter-region', action='append', default=[])
all_rows = lambda sh, start=0: (sh.row(x) for x in range(start, sh.nrows))

class Node(namedtuple('Node', 'city state country region latitude longitude node_type')):
    def __new__(cls, city, state='', country='', region='', latitude=0, longitude=0, node_type='ILA'):
        values = [latitude, longitude, node_type]
        default_values = [0, 0, 'ILA']
        [latitude, longitude, node_type] \
            = [x[0] if x[0] != '' else x[1] for x in zip(values,default_values)]
        return super().__new__(cls, city, state, country, region, latitude, longitude, node_type)

class Link(namedtuple('Link', 'from_city to_city \
    east_distance east_fiber east_lineic east_con_in east_con_out east_pmd east_cable \
    west_distance west_fiber west_lineic west_con_in west_con_out west_pmd west_cable \
    distance_units')):
    def __new__(cls, from_city, to_city,
      east_distance, east_fiber='SSMF', east_lineic=0.2, 
      east_con_in=0.5, east_con_out=0.5, east_pmd=0.1, east_cable='', 
      west_distance=-100, west_fiber='SSMF', west_lineic=0.2, 
      west_con_in=0.5, west_con_out=0.5, west_pmd=0.1, west_cable='',
      distance_units='km'):
        values = [from_city, to_city, 
            east_distance, east_fiber, east_lineic, east_con_in, east_con_out, east_pmd, east_cable,
            west_distance, west_fiber, west_lineic, west_con_in, west_con_out, west_pmd, west_cable]
        default_values = ['','',0,'SSMF',0.2,0.5,0.5,0.1,'',-100,'SSMF',0.2,0.5,0.5,0.1,'']
        [from_city, to_city, 
            east_distance, east_fiber, east_lineic, east_con_in, east_con_out, east_pmd, east_cable,
            west_distance, west_fiber, west_lineic, west_con_in, west_con_out, west_pmd, west_cable]\
            = [x[0] if x[0] != '' else x[1] for x in zip(values,default_values)]

        west_distance = east_distance if west_distance == -100 else west_distance

        return super().__new__(cls, from_city, to_city,
          east_distance, east_fiber, east_lineic, east_con_in, east_con_out, east_pmd, east_cable,
          west_distance, west_fiber, west_lineic, west_con_in, west_con_out, west_pmd, west_cable,
          distance_units)     

def convert_file(input_filename, filter_region=[]):
    nodes, links = parse_excel(input_filename)

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

    for city,link in links_by_city.items():
        if nodes_by_city[city].node_type.lower()=='ila' and len(link) != 2:
            #wrong input: ILA sites can only be Degree 2 
            # => correct to make it a ROADM and remove entry in links_by_city
            #TODO : put in log rather than print
            print(f'invalid node type ({nodes_by_city[city].node_type})\
 specified in {city}, replaced by ROADM')
            nodes_by_city[city] = nodes_by_city[city]._replace(node_type='ROADM')
            nodes = [n._replace(node_type='ROADM') if n.city==city else n for n in nodes]

    data = {
        'elements':
            [{'uid': f'trx {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Transceiver'}
             for x in nodes if x.node_type.lower() == 'roadm'] +
            [{'uid': f'roadm {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Roadm'}
             for x in nodes if x.node_type.lower() == 'roadm'] +             
            [{'uid': f'fiber ({x.from_city} → {x.to_city})-{x.east_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.east_fiber,
              'params': {'length':   round(x.east_distance, 3),
                         'length_units':    x.distance_units,
                         'loss_coef': x.east_lineic}
              }
             for x in links]+
            [{'uid': f'fiber ({x.to_city} → {x.from_city})-{x.west_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.west_fiber,
              'params': {'length':   round(x.west_distance, 3),
                         'length_units':    x.distance_units,
                         'loss_coef': x.west_lineic}
              }
              for x in links],             
        'connections':
            list(chain.from_iterable([fiber_connection_by_city(n.city)
            for n in nodes]))
            +
            list(chain.from_iterable(zip(
            [{'from_node': f'trx {x.city}',
              'to_node':   f'roadm {x.city}'}
             for x in nodes if x.node_type.lower()=='roadm'],
            [{'from_node': f'roadm {x.city}',
              'to_node':   f'trx {x.city}'}
             for x in nodes if x.node_type.lower()=='roadm'])))            
    }

    #print(dumps(data, indent=2))
    output_json_file_name = input_filename.split(".")[0]+".json"
    with  open(output_json_file_name,'w') as edfa_json_file:
        edfa_json_file.write(dumps(data, indent=2))

def parse_excel(input_filename):
    with open_workbook(input_filename) as wb:
        nodes_sheet = wb.sheet_by_name('Nodes')
        links_sheet = wb.sheet_by_name('Links')

        # sanity check
        """
        header = [x.value.strip() for x in nodes_sheet.row(4)]
        expected = ['City', 'State', 'Country', 'Region', 'Latitude', 'Longitude']
        if header != expected:
            raise ValueError(f'Malformed header on Nodes sheet: {header} != {expected}')
        """ 

        nodes = []
        for row in all_rows(nodes_sheet, start=5):
            nodes.append(Node(*(x.value for x in row)))
        #check input
        expected_node_types = ('ROADM', 'ILA', 'FUSED')
        nodes = [n._replace(node_type='ILA') 
                if not (n.node_type in expected_node_types) else n for n in nodes]

        # sanity check
        """
        header = [x.value.strip() for x in links_sheet.row(4)]
        expected = ['Node A', 'Node Z', 
            'Distance (km)', 'Fiber type', 'lineic att', 'Con_in', 'Con_out', 'PMD', 'Cable id',
            'Distance (km)', 'Fiber type', 'lineic att', 'Con_in', 'Con_out', 'PMD', 'Cable id']
        if header != expected:
            raise ValueError(f'Malformed header on Nodes sheet: {header} != {expected}')
        """
        links = []
        for row in all_rows(links_sheet, start=5):
            links.append(Link(*(x.value for x in row)))


    # sanity check
    all_cities = Counter(n.city for n in nodes)
    if len(all_cities) != len(nodes):
        ValueError(f'Duplicate city: {all_cities}')
    if any(ln.from_city not in all_cities or
           ln.to_city   not in all_cities for ln in links):
        ValueError(f'Bad link.')

    return nodes, links


def fiber_connection_by_city(city_name):
    other_cities = fiber_dest_from_source(city_name)
    subdata = []
    if nodes_by_city[city_name].node_type.lower() in ('ila', 'fused'):
        # Then len(other_cities) == 2
        subdata = [{'from_node': fiber_link(other_cities[0], city_name), 
                  'to_node': fiber_link(city_name, other_cities[1])},
                  {'from_node': fiber_link(other_cities[1], city_name), 
                  'to_node': fiber_link(city_name, other_cities[0])}]
    elif nodes_by_city[city_name].node_type.lower() == 'roadm':
        subdata = list(chain.from_iterable(zip(
                  [{'from_node': f'roadm {city_name}',
                  'to_node': fiber_link(city_name, other_city)} 
                  for other_city in other_cities],
                  [{'from_node': fiber_link(other_city, city_name),
                  'to_node': f'roadm {city_name}'}
                  for other_city in other_cities])))
    return subdata

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
    link = links_by_city[from_city]
    source_dest = (from_city, to_city)
    l = next(l for l in link if l.from_city in source_dest and l.to_city in source_dest)
    if l.from_city == from_city:
        fiber = f'fiber ({l.from_city} → {l.to_city})-{l.east_cable}'
    else:
        fiber = f'fiber ({l.to_city} → {l.from_city})-{l.west_cable}'
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

if __name__ == '__main__':
    args = parser.parse_args()
    convert_file(args.workbook, args.filter_region)

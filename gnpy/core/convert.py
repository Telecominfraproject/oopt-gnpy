#!/usr/bin/env python3
"""
@author: briantaylor
@author: giladgoldfarb
@author: jeanluc-auge
xls to json parser, that can be called directly from the transmission_main_example
xls examples are meshTopologyExampleV2.xls and CORONET_Global_Topology.xls
Require Nodes and Links sheets, Eqpt sheet is optional
*in Nodes sheet, only the 'City' column is mandatory. The column 'Type' is discovered based
on the topology: degree 2 = ILA, other degrees = ROADM. The value is also corrected if the user 
specifies an ILA of degree != 2.
*In Links sheet only the 3 first columns (Node A, Node Z and east Distance (km)) are mandatory. 
Missing west information are copied from east information so it is possible to input undir data
*in Eqpt sheet 

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

all_rows = lambda sh, start=0: (sh.row(x) for x in range(start, sh.nrows))

class Node(namedtuple('Node', 'city state country region latitude longitude node_type')):
    def __new__(cls, city, state='', country='', region='', latitude=0, longitude=0, node_type='ILA'):
        values = [latitude, longitude, node_type]
        default_values = [0, 0, 'ILA']
        values = [x[0] if x[0] != '' else x[1] for x in zip(values,default_values)]
        return super().__new__(cls, city, state, country, region, *values)

class Link(namedtuple('Link', 'from_city to_city \
    east_distance east_fiber east_lineic east_con_in east_con_out east_pmd east_cable \
    west_distance west_fiber west_lineic west_con_in west_con_out west_pmd west_cable \
    distance_units')):
    def __new__(cls, from_city, to_city,
      east_distance, east_fiber='SSMF', east_lineic=0.2, 
      east_con_in=0.5, east_con_out=0.5, east_pmd=0.1, east_cable='', 
      west_distance='', west_fiber='', west_lineic='', 
      west_con_in='', west_con_out='', west_pmd='', west_cable='',
      distance_units='km'):
        east_values = [east_distance, east_fiber, east_lineic, east_con_in, east_con_out, 
                        east_pmd, east_cable]
        west_values = [west_distance, west_fiber, west_lineic, west_con_in, west_con_out, 
                        west_pmd, west_cable]
        default_values = [80,'SSMF',0.2,0.5,0.5,0.1,'']
        east_values = [x[0] if x[0] != '' else x[1] for x in zip(east_values,default_values)]
        west_values = [x[0] if x[0] != '' else x[1] for x in zip(west_values,east_values)]
        return super().__new__(cls, from_city, to_city, *east_values, *west_values, distance_units)     

class Eqpt(namedtuple('Eqpt', 'from_city to_city \
    egress_amp_type egress_att_in egress_amp_gain egress_amp_tilt egress_amp_att_out\
    ingress_amp_type ingress_att_in ingress_amp_gain ingress_amp_tilt ingress_amp_att_out')):
    def __new__(cls, from_city='', to_city='',
    egress_amp_type='', egress_att_in=0, egress_amp_gain=0, egress_amp_tilt=0, egress_amp_att_out=0,
    ingress_amp_type='', ingress_att_in=0, ingress_amp_gain=0, ingress_amp_tilt=0, ingress_amp_att_out=0):
        values = [from_city, to_city,
            egress_amp_type, egress_att_in, egress_amp_gain, egress_amp_tilt, egress_amp_att_out,
            ingress_amp_type, ingress_att_in, ingress_amp_gain, ingress_amp_tilt, ingress_amp_att_out]
        default_values = ['','','',0,0,0,0,'',0,0,0,0]
        values = [x[0] if x[0] != '' else x[1] for x in zip(values,default_values)]
        return super().__new__(cls, *values)        

def sanity_check(nodes, nodes_by_city, links_by_city, eqpts_by_city):
    try :
        test_nodes = [n for n in nodes_by_city if not n in links_by_city]
        test_links = [n for n in links_by_city if not n in nodes_by_city]
        test_eqpts = [n for n in eqpts_by_city if not n in nodes_by_city]
        assert (test_nodes == [] or test_nodes == [''])\
                and (test_links == [] or test_links ==[''])\
                and (test_eqpts == [] or test_eqpts ==[''])
    except AssertionError:
        print(f'!names in Nodes and Links sheets do no match, check:\
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
            nodes_by_city[city] = nodes_by_city[city]._replace(node_type='ROADM')
            nodes = [n._replace(node_type='ROADM') if n.city==city else n for n in nodes]
    return nodes

def convert_file(input_filename, filter_region=[]):
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

    global links_by_city
    links_by_city = defaultdict(list)
    for link in links:
        links_by_city[link.from_city].append(link)
        links_by_city[link.to_city].append(link)

    global eqpts_by_city
    eqpts_by_city = defaultdict(list)
    for eqpt in eqpts:
        eqpts_by_city[eqpt.from_city].append(eqpt) 

    nodes = sanity_check(nodes, nodes_by_city, links_by_city, eqpts_by_city)

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
             for x in nodes_by_city.values() if x.node_type.lower() == 'roadm'] +
            [{'uid': f'ingress fused spans in {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Fused'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'fused'] +
            [{'uid': f'egress fused spans in {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Fused'}
             for x in nodes_by_city.values() if x.node_type.lower() == 'fused'] +                                    
            [{'uid': f'fiber ({x.from_city} → {x.to_city})-{x.east_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.east_fiber,
              'params': {'length':   round(x.east_distance, 3),
                         'length_units':    x.distance_units,
                         'loss_coef': x.east_lineic,
                         'connector_loss_in':x.east_con_in,
                         'connector_loss_out':x.east_con_out}
            }
              for x in links] +
            [{'uid': f'fiber ({x.to_city} → {x.from_city})-{x.west_cable}',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'type_variety': x.west_fiber,
              'params': {'length':   round(x.west_distance, 3),
                         'length_units':    x.distance_units,
                         'loss_coef': x.west_lineic,
                         'connector_loss_in':x.west_con_in,
                         'connector_loss_out':x.west_con_out}
            } # missing ILA construction 
              for x in links] +
            [{'uid': f'egress edfa in {e.from_city} to {e.to_city}',
              'metadata': {'location': {'city':      nodes_by_city[e.from_city].city,
                                        'region':    nodes_by_city[e.from_city].region,
                                        'latitude':  nodes_by_city[e.from_city].latitude,
                                        'longitude': nodes_by_city[e.from_city].longitude}},
              'type': 'Edfa',
              'type_variety': e.egress_amp_type,
              'operational': {'gain_target': e.egress_amp_gain,
                              'tilt_target': e.egress_amp_tilt}
            }
             for e in eqpts if e.egress_amp_type.lower() != ''] +
            [{'uid': f'ingress edfa in {e.from_city} to {e.to_city}',
              'metadata': {'location': {'city':      nodes_by_city[e.from_city].city,
                                        'region':    nodes_by_city[e.from_city].region,
                                        'latitude':  nodes_by_city[e.from_city].latitude,
                                        'longitude': nodes_by_city[e.from_city].longitude}},
              'type': 'Edfa',
              'type_variety': e.ingress_amp_type,
              'operational': {'gain_target': e.ingress_amp_gain,
                              'tilt_target': e.ingress_amp_tilt}
              }              
             for e in eqpts if e.ingress_amp_type.lower() != ''],
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

    #print(dumps(data, indent=2))
    # output_json_file_name = input_filename.split(".")[0]+".json"
    suffix_filename = str(input_filename.suffixes[0])
    full_input_filename = str(input_filename)
    split_filename = [full_input_filename[0:len(full_input_filename)-len(suffix_filename)] , suffix_filename[1:]]
    output_json_file_name = split_filename[0]+'.json'
    with  open(output_json_file_name,'w') as edfa_json_file:
        edfa_json_file.write(dumps(data, indent=2))
    return output_json_file_name

def parse_excel(input_filename):
    with open_workbook(input_filename) as wb:
        nodes_sheet = wb.sheet_by_name('Nodes')
        links_sheet = wb.sheet_by_name('Links')
        try:
            eqpt_sheet = wb.sheet_by_name('Eqpt')
        except:
            #eqpt_sheet is optional
            eqpt_sheet = None


        # sanity check
        """
        header = [x.value.strip() for x in nodes_sheet.row(4)]
        expected = ['City', 'State', 'Country', 'Region', 'Latitude', 'Longitude']
        if header != expected:
            raise ValueError(f'Malformed header on Nodes sheet: {header} != {expected}')
        """ 

        nodes = []
        for row in all_rows(nodes_sheet, start=5):
            nodes.append(Node(*(x.value for x in row[0:NODES_COLUMN])))
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
            links.append(Link(*(x.value for x in row[0:LINKS_COLUMN])))

        eqpts = []
        if eqpt_sheet != None:
            for row in all_rows(eqpt_sheet, start=5):
                eqpts.append(Eqpt(*(x.value for x in row[0:EQPTS_COLUMN])))

    # sanity check
    all_cities = Counter(n.city for n in nodes)
    if len(all_cities) != len(nodes):
        ValueError(f'Duplicate city: {all_cities}')
    if any(ln.from_city not in all_cities or
           ln.to_city   not in all_cities for ln in links):
        ValueError(f'Bad link.')

    return nodes, links, eqpts


def eqpt_connection_by_city(city_name):
    other_cities = fiber_dest_from_source(city_name)
    subdata = []
    if nodes_by_city[city_name].node_type.lower() in ('ila', 'fused'):
        # Then len(other_cities) == 2
        direction = ['ingress', 'egress']
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
            in_ = eqpt_in_city_to_city(city_name, other_city, "ingress")
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


def eqpt_in_city_to_city(in_city, to_city, direction='egress'):
    rev_direction = 'ingress' if direction == 'egress' else 'egress'
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

#output_json_file_name = 'coronet_conus_example.json'
#TODO get column size automatically from tupple size
NODES_COLUMN = 7
LINKS_COLUMN = 16
EQPTS_COLUMN = 12
parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', type=Path , default='meshTopologyExampleV2.xls')
parser.add_argument('-f', '--filter-region', action='append', default=[])

if __name__ == '__main__':
    args = parser.parse_args()
    convert_file(args.workbook, args.filter_region)

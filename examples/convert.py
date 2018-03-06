#!/usr/bin/env python3

from sys import exit
try:
    from xlrd import open_workbook
except ModuleNotFoundError:
    exit('Required: `pip install xlrd`')
from argparse import ArgumentParser
from collections import namedtuple, Counter
from itertools import chain
from json import dumps
from uuid import uuid4
import math
import numpy as np

output_json_file_name = 'coronet_conus_example.json'
Node = namedtuple('Node', 'city state country region latitude longitude')
class Link(namedtuple('Link', 'from_city to_city distance distance_units')):
    def __new__(cls, from_city, to_city, distance, distance_units='km'):
        return super().__new__(cls, from_city, to_city, distance, distance_units)

def define_span_range(min_span, max_span, nspans):
    srange = (max_span - min_span) + min_span*np.random.rand(nspans)
    return srange

def amp_spacings(min_span,max_span,length):
    nspans =  math.ceil(length/100)
    spans = define_span_range(min_span, max_span, nspans)
    tot = spans.sum()
    delta = length -tot
    if delta > 0 and delta < 25:
        ind  = np.where(np.min(spans))
        spans[ind] = spans[ind] + delta
    elif delta >= 25 and delta < 40:
        spans = spans + delta/float(nspans)
    elif delta > 40 and delta < 100:
        spans = np.append(spans,delta)
    elif delta > 100:
        spans  = np.append(spans, [delta/2, delta/2])
    elif delta < 0:
        spans = spans + delta/float(nspans)
    return list(spans)

def parse_excel(args):
    with open_workbook(args.workbook) as wb:
        nodes_sheet = wb.sheet_by_name('Nodes')
        links_sheet = wb.sheet_by_name('Links')

        # sanity check
        header = [x.value.strip() for x in nodes_sheet.row(4)]
        expected = ['City', 'State', 'Country', 'Region', 'Latitude', 'Longitude']
        if header != expected:
            raise ValueError(f'Malformed header on Nodes sheet: {header} != {expected}')

        nodes = []
        for row in all_rows(nodes_sheet, start=5):
            nodes.append(Node(*(x.value for x in row)))

        # sanity check
        header = [x.value.strip() for x in links_sheet.row(4)]
        expected = ['Node A', 'Node Z', 'Distance (km)']
        if header != expected:
            raise ValueError(f'Malformed header on Nodes sheet: {header} != {expected}')

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

parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', default='CORONET_Global_Topology.xls')
parser.add_argument('-f', '--filter-region', action='append', default=[])

all_rows = lambda sh, start=0: (sh.row(x) for x in range(start, sh.nrows))

def midpoint(city_a, city_b):
    lats  = city_a.latitude, city_b.latitude
    longs = city_a.longitude, city_b.longitude
    return {
        'latitude':  sum(lats)  / 2,
        'longitude': sum(longs) / 2,
    }

if __name__ == '__main__':
    args = parser.parse_args()
    nodes, links = parse_excel(args)

    if args.filter_region:
        nodes = [n for n in nodes if n.region.lower() in args.filter_region]
        cities = {n.city for n in nodes}
        links = [lnk for lnk in links if lnk.from_city in cities and
                                         lnk.to_city in cities]
        cities = {lnk.from_city for lnk in links} | {lnk.to_city for lnk in links}
        nodes = [n for n in nodes if n.city in cities]

    nodes_by_city = {n.city: n for n in nodes}

    data = {
        'elements':
            [{'uid': f'trx {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Transceiver'}
             for x in nodes] +
            [{'uid': f'roadm {x.city}',
              'metadata': {'location': {'city':      x.city,
                                        'region':    x.region,
                                        'latitude':  x.latitude,
                                        'longitude': x.longitude}},
              'type': 'Roadm'}
             for x in nodes] +             
            [{'uid': f'fiber ({x.from_city} → {x.to_city})',
              'metadata': {'location': midpoint(nodes_by_city[x.from_city],
                                                nodes_by_city[x.to_city])},
              'type': 'Fiber',
              'params': {'length':   round(x.distance, 3),
                         'length_units':    x.distance_units,
                         'loss_coef': 0.2,
                         'dispersion': 16.7E-6,
                         'gamma': 1.27E-3}
              }
             for x in links],
        'connections':
            list(chain.from_iterable(zip( # put bidi next to each other
            [{'from_node': f'roadm {x.from_city}',
              'to_node':   f'fiber ({x.from_city} → {x.to_city})'}
             for x in links],
            [{'from_node': f'fiber ({x.from_city} → {x.to_city})',
              'to_node':   f'roadm {x.from_city}'}
             for x in links])))
            +
            list(chain.from_iterable(zip(
            [{'from_node': f'fiber ({x.from_city} → {x.to_city})',
              'to_node':   f'roadm {x.to_city}'}
             for x in links],
            [{'from_node': f'roadm {x.to_city}',
              'to_node':   f'fiber ({x.from_city} → {x.to_city})'}
             for x in links])))
            +
            list(chain.from_iterable(zip(
            [{'from_node': f'trx {x.city}',
              'to_node':   f'roadm {x.city}'}
             for x in nodes],
            [{'from_node': f'roadm {x.city}',
              'to_node':   f'trx {x.city}'}
             for x in nodes])))            
    }

    print(dumps(data, indent=2))
    with  open(output_json_file_name,'w') as edfa_json_file:
        edfa_json_file.write(dumps(data, indent=2))
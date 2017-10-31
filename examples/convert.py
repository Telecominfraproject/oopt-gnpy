#!/usr/bin/env python3

from sys import exit
try:
    from xlrd import open_workbook
except ModuleNotFoundError:
    exit('Required: `pip install xlrd`')
from argparse import ArgumentParser
from collections import namedtuple, Counter
from json import dumps

Node = namedtuple('Node', 'city state country region latitutde longitude')
Link = namedtuple('Link', 'from_city to_city distance')

parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', default='CORONET_Global_Topology.xls')

all_rows = lambda sh, start=0: (sh.row(x) for x in range(start, sh.nrows))

if __name__ == '__main__':
    args = parser.parse_args()

    with open_workbook(args.workbook) as wb:
        nodes_sheet = wb.sheet_by_name('Nodes')
        links_sheet = wb.sheet_by_name('Links')

        # sanity check
        header = [x.value.strip() for x in nodes_sheet.row(4)]
        expected = ['City', 'State', 'Country', 'Region', 'Latitude', 'Longitude']
        if header != expected:
            exit(f'Malformed header on Nodes sheet: {header} != {expected}')

        nodes = []
        for row in all_rows(nodes_sheet, start=5):
            nodes.append(Node(*(x.value for x in row)))

        # sanity check
        header = [x.value.strip() for x in links_sheet.row(4)]
        expected = ['Node A', 'Node Z', 'Distance (km)']
        if header != expected:
            exit(f'Malformed header on Nodes sheet: {header} != {expected}')

        links = []
        for row in all_rows(links_sheet, start=5):
            link = Link(*(x.value for x in row))
            link = link._replace(distance=link.distance * 1000) # base units
            links.append(link)

    # sanity check
    all_cities = Counter(n.city for n in nodes)
    if len(all_cities) != len(nodes):
        exit(f'Duplicate city: {all_cities}')
    if any(ln.from_city not in all_cities or
           ln.to_city   not in all_cities for ln in links):
        exit(f'Bad link.')

    conus_nodes = [n for n in nodes if n.region == 'CONUS']
    conus_cities = {n.city for n in conus_nodes}
    conus_links = [ln for ln in links if ln.from_city in conus_cities
                                     and ln.to_city   in conus_cities]

    data = {
        "nodes": [{"node": x.city}
            for x in conus_nodes],
        "links": [{"from_node": x.from_city,
                   "to_node": x.to_city,
                   "distance": x.distance}
            for x in conus_links],
    }

    print(dumps(data, indent=2))

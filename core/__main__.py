from argparse import ArgumentParser
from json import load
from sys import exit
from pathlib import Path
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG

from matplotlib.pyplot import show, axis
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels)

from . import network_from_json
from .elements import City, Fiber

logger = getLogger('gnpy.core')

def main(args):
    with open(args.filename) as f:
        json_data = load(f)

    network = network_from_json(json_data)
    pos    = {n: (n.longitude, n.latitude) for n in network.nodes()}
    labels_pos = {n: (long-.5, lat-.5) for n, (long, lat) in pos.items()}
    size   = [20        if isinstance(n, Fiber) else 80    for n in network.nodes()]
    color  = ['green'   if isinstance(n, City)  else 'red' for n in network.nodes()]
    labels = {n: n.city if isinstance(n, City)  else ''    for n in network.nodes()}
    draw_networkx_nodes(network, pos=pos, node_size=size, node_color=color)
    draw_networkx_edges(network, pos=pos)
    draw_networkx_labels(network, pos=labels_pos, labels=labels, font_size=14)
    axis('off')
    show()

parser = ArgumentParser()
parser.add_argument('filename', nargs='?', type=Path,
  default= Path(__file__).parent / '../examples/coronet.conus.json')
parser.add_argument('-v', '--verbose', action='count')

if __name__ == '__main__':
    args = parser.parse_args()
    level = {1: INFO, 2: DEBUG}.get(args.verbose, ERROR)
    logger.setLevel(level)
    basicConfig()
    exit(main(args))

#!/usr/bin/env python3
from itertools import tee, islice

from networkx import shortest_simple_paths, DiGraph, bfs_edges

nwise = lambda g, n=2: zip(*(islice(g, i, None) for i, g in enumerate(tee(g, n))))

def bfs_simple_paths(network, source, sink, limit=5):
    g = DiGraph()
    for path in islice(shortest_simple_paths(network, source, sink), limit):
        for u, v in nwise(path):
            g.add_edge(u, v)
    return bfs_edges(g, source)

def propagate(network, source, sink, spectral_info):
    state = {source: [spectral_info]}
    for from_node, to_node in bfs_simple_paths(network, source, sink):
        state[to_node] = list(to_node(*state[from_node]))
    return state

if __name__ == '__main__':
    from collections import namedtuple, deque
    from networkx import DiGraph
    from pprint import pformat

    class City(namedtuple('City', 'uid')):
        def __call__(self, spectral_info):
            yield spectral_info

    class Fiber(namedtuple('Fiber', 'uid')):
        def __call__(self, spectral_info):
            yield spectral_info._replace(signal = spectral_info.signal / 2,
                                         noise  = spectral_info.noise  * 2)

    SpectralInformation = namedtuple('SpectralInformation', 'signal noise')

    nodes = {'nyc': City('nyc'),
             'sf':  City('sf'),
             'eastbound': Fiber('eastbound'),
             'westbound': Fiber('westbound')}

    g = DiGraph()
    g.add_edge(nodes['nyc'],       nodes['westbound'])
    g.add_edge(nodes['westbound'], nodes['sf'])
    g.add_edge(nodes['sf'],        nodes['eastbound'])
    g.add_edge(nodes['eastbound'], nodes['nyc'])

    si = SpectralInformation(signal=10, noise=1)

    experiments = {'propagate east': (nodes['sf'], nodes['nyc']),
                   'propagate west': (nodes['nyc'], nodes['sf'])}

    for title, (source, sink) in experiments.items():
        results = propagate(g, source, sink, si)

        print(title.center(80, "-"))
        print(f'{source.uid:<8}{results[source]!r:<72}')
        print(f'{"east →":<8}{results.get(nodes["eastbound"])!r:<72}')
        print(f'{"west ←":<8}{results.get(nodes["westbound"])!r:<72}')
        print(f'{sink.uid:<8}{results[sink]!r:<72}')

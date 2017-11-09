#!/usr/bin/env python3

from itertools import tee, islice

def propagate(path, spectral_info):
    for node in path:
        rv = node(spectral_info)
        yield spectral_info, node, rv
        spectral_info = rv

if __name__ == '__main__':
    from collections import namedtuple, deque
    from networkx import DiGraph, all_simple_paths
    from pprint import pformat

    class City(namedtuple('City', 'uid')):
        def __call__(self, spectral_info):
            return spectral_info

    class Fiber(namedtuple('Fiber', 'uid')):
        def __call__(self, spectral_info):
            return spectral_info._replace(signal = spectral_info.signal / 2,
                                         noise  = spectral_info.noise  * 2)

    SpectralInformation = namedtuple('SpectralInformation', 'signal noise')

    nodes = {'nyc': City('nyc'),
             'sf':  City('sf'),
             'chi': City('chi'),
             'eastbound 1': Fiber('eastbound 1'),
             'eastbound 2': Fiber('eastbound 2'),
             'eastbound 3': Fiber('eastbound 3'),
             'westbound 1': Fiber('westbound 1'),
             'westbound 2': Fiber('westbound 2'),
             'westbound 3': Fiber('westbound 3'),}

    g = DiGraph()
    g.add_edge(nodes['nyc'],         nodes['westbound 1'])
    g.add_edge(nodes['nyc'],         nodes['westbound 2'])
    g.add_edge(nodes['westbound 1'], nodes['chi'])
    g.add_edge(nodes['westbound 2'], nodes['chi'])
    g.add_edge(nodes['chi'],         nodes['westbound 3'])
    g.add_edge(nodes['westbound 3'], nodes['sf'])

    g.add_edge(nodes['sf'],          nodes['eastbound 1'])
    g.add_edge(nodes['sf'],          nodes['eastbound 2'])
    g.add_edge(nodes['eastbound 1'], nodes['chi'])
    g.add_edge(nodes['eastbound 2'], nodes['chi'])
    g.add_edge(nodes['chi'],         nodes['eastbound 3'])
    g.add_edge(nodes['eastbound 3'], nodes['nyc'])

    si = SpectralInformation(signal=10, noise=1)

    experiments = {'propagate east': (nodes['sf'], nodes['nyc']),
                   'propagate west': (nodes['nyc'], nodes['sf'])}

    for title, (source, sink) in experiments.items():
        print(title.center(20).center(120, '='))
        for path in all_simple_paths(g, source, sink):
            print('path'.center(20).center(120, '-'))
            for idx, (in_si, node, out_si) in enumerate(propagate(path, si)):
                print(f'{node!s:<35}{"" if idx else in_si!s:<40} → {out_si!s:<40}')

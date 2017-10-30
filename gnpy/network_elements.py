from networkx import DiGraph, all_simple_paths
from collections import defaultdict
from itertools import product

import gnpy
from . import utils


class Opath:

    def __init__(self, nw, path):
        self.nw, self.path = nw, path

        self.edge_list = {(elem, path[en + 1])
                          for en, elem in enumerate(path[:-1])}
        self.elem_dict = {elem: self.find_io_edges(elem)
                          for elem in self.path}

    def find_io_edges(self, elem):
        iedges = set(self.nw.g.in_edges(elem) ) & self.edge_list
        oedges = set(self.nw.g.out_edges(elem)) & self.edge_list
        return {'in': iedges, 'out': oedges}

    def propagate(self):
        for elem in self.path:
            elem.propagate(path=self)


class Network:

    def __init__(self, config):
        self.config = config
        self.nw_elems = defaultdict(list)
        self.g = DiGraph()

        for elem in self.config['elements']:
            ne_type = TYPE_MAP[elem['type']]
            params = elem.pop('parameters')
            ne = ne_type(self, **elem, **params)
            self.nw_elems[ne_type].append(ne)
            self.g.add_node(ne)

        for gpath in self.config['topology']:
            for u, v in utils.nwise(gpath):
                n0 = utils.find_by_node_id(self.g, u)
                n1 = utils.find_by_node_id(self.g, v)
                self.g.add_edge(n0, n1, channels=[])

        # define all possible paths between tx's and rx's
        self.tr_paths = []
        for tx, rx in product(self.nw_elems[Tx], self.nw_elems[Rx]):
            for spath in all_simple_paths(self.g, tx, rx):
                self.tr_paths.append(Opath(self, spath))

    def propagate_all_paths(self):
        for opath in self.tr_paths:
            opath.propagate()


class NetworkElement:

    def __init__(self, nw, *, id, type, name, description, **kwargs):
        self.nw = nw
        self.id, self.type = id, type
        self.name, self.description = name, description

    def fetch_edge(self, edge):
        u, v = edge
        return self.nw.g[u][v]

    def edge_dict(self, chan, osnr, d_power):
        return {'frequency': chan['frequency'],
                'osnr': osnr if osnr else chan['osnr'],
                'power': chan['power'] + d_power}

    def __repr__(self):
        return f'NetworkElement(id={self.id}, type={self.type})'


class Fiber(NetworkElement):

    def __init__(self, *args, length, loss, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = length
        self.loss = loss

    def propagate(self, path):
        attn = self.length * self.loss

        for oedge in path.elem_dict[self]['out']:
            edge = self.fetch_edge(oedge)
            for pedge in (self.fetch_edge(x)
                          for x in path.elem_dict[self]['in']):
                for chan in pedge['channels']:
                    dct = self.edge_dict(chan, None, -attn)
                    edge['channels'].append(dct)


class Edfa(NetworkElement):

    def __init__(self, *args, gain, nf, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = gain
        self.nf = nf

    def propagate(self, path):
        gain = self.gain[0]
        for inedge in path.elem_dict[self]['in']:

            in_channels = self.fetch_edge(inedge)['channels']
            for chan in in_channels:
                osnr = utils.chan_osnr(chan, self)
                for edge in (self.fetch_edge(x)
                             for x in path.elem_dict[self]['out']):
                    dct = self.edge_dict(chan, osnr, gain)
                    edge['channels'].append(dct)


class Tx(NetworkElement):

    def __init__(self, *args, channels, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels

    def propagate(self, path):
        for edge in (self.fetch_edge(x) for x in path.elem_dict[self]['out']):
            for chan in self.channels:
                dct = self.edge_dict(chan, None, 0)
                edge['channels'].append(dct)


class Rx(NetworkElement):

    def __init__(self, *args, sensitivity, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitivity = sensitivity

    def propagate(self, path):
        self.channels = {}
        for iedge in path.elem_dict[self]['in']:
            edge = self.fetch_edge(iedge)
            self.channels[path] = edge['channels']


TYPE_MAP = {
    'fiber': Fiber,
    'tx':    Tx,
    'rx':    Rx,
    'edfa':  Edfa,
}

import networkx as nx
from .utils import Utils
import gnpy
from pprint import pprint as pp

class Params:
    def __init__(self, *args):
        req_params = args[0]
        params = args[1].get('parameters')
        missing_params = list(set(req_params) - set(params.keys()))
        if len(missing_params):
            print("missing params:", ','.join(missing_params))
            raise ValueError
        for k, v in params.items():
            setattr(self, k, v)


class Opath:

    def __init__(self, nw, path):
        self.path = path
        self.edge_list = [(elem, path[en + 1])
                          for en, elem in enumerate(path[:-1])]
        self.elem_dict = {elem: self.find_io_edges(nw, elem)
                          for elem in self.path}

    def find_io_edges(self, nw, elem):
        iedges = set(nw.g.in_edges(elem)).intersection(self.edge_list)
        oedges = set(nw.g.out_edges(elem)).intersection(self.edge_list)
        return {'in': list(iedges),
                'out': list(oedges)}

        ies = []
        for ie in list(iedges):
            ies.append(nw.g[ie[0]][ie[1]])
        oes = []
        for ie in list(oedges):
            ies.append(nw.g[ie[0]][ie[1]])
        return {'in': ies, 'out': oes}

    #def propegate(self):
    #    for elem in self.path:
    #        print(elem)


class Network:

    g = nx.DiGraph()
    nw_elems = {}

    def __init__(self, network_config):
        self.config = Utils.read_config(network_config)
        for elem in self.config['elements']:
            ne_type = elem['type'].capitalize()
            if ne_type not in self.nw_elems:
                self.nw_elems[ne_type] = []
            ne = getattr(gnpy, ne_type)(self, **elem)
            self.nw_elems[ne_type].append(ne)
            self.g.add_node(ne)

        for gpath in self.config['topology']:
            n0 = Utils.find_by_node_id(self.g, gpath[0])
            for nid in gpath[1:]:
                n1 = Utils.find_by_node_id(self.g, nid)
                self.g.add_edge(n0, n1, channels=[])
                n0 = n1

        # define all possible paths between tx's and rx's
        self.tr_paths = []
        for tx in self.nw_elems['Tx']:
            for rx in self.nw_elems['Rx']:
                for spath in nx.all_simple_paths(self.g, tx, rx):
                    self.tr_paths.append(Opath(self, spath))

    def propagate_all_paths(self):
        for opath in self.tr_paths:
            print(opath.path)


class NetworkElement:

    def __init__(self, nw, **kwargs):
        self.nw = nw
        self.id = kwargs.get('id')
        self.type = kwargs.get('type')
        self.name = kwargs.get('name')
        self.description = kwargs.get('description')
        self.params = Params(self.required_params, kwargs)

    def fetch_edge(self, edge):
        return self.nw.g[edge[0]][edge[1]]

    def __repr__(self):
        return self.id


class Fiber(NetworkElement):
    required_params = ['length', 'loss']

    def propagate(self, path):
        attn = self.params.length * self.params.loss
        for oedge in path.elem_dict[self]['out']:
            edge = self.fetch_edge(oedge)
            for inedge in path.elem_dict[self]['in']:
                pedge = self.fetch_edge(inedge)
                for chan in pedge['channels']:
                    edge['channels'].append(Utils.edge_dict(chan, None, -attn))


class Edfa(NetworkElement):
    required_params = ['gain', 'nf']

    def propagate(self, path):
        gain = self.params.gain[0]
        for inedge in path.elem_dict[self]['in']:
            in_channels = self.fetch_edge(inedge)['channels']
            for chan in in_channels:
                osnr = Utils.chan_osnr(chan, self.params)
                for oedge in path.elem_dict[self]['out']:
                    edge = self.fetch_edge(oedge)
                    edge['channels'].append(Utils.edge_dict(chan, osnr, gain))


class Tx(NetworkElement):
    required_params = ['channels']

    def propagate(self, path):
        for oedge in path.elem_dict[self]['out']:
            edge = self.fetch_edge(oedge)
            for chan in self.params.channels:
                edge['channels'].append(Utils.edge_dict(chan, None, 0))


class Rx(NetworkElement):
    required_params = ['sensitivity']

    def propagate(self, path):
        for iedge in path.elem_dict[self]['in']:
            edge = self.fetch_edge(iedge)
            print(path)
            pp(edge['channels'])

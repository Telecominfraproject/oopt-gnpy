import networkx as nx
from .utils import Utils
import gnpy


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


class Channels():
    pass


class Network:
    g = nx.DiGraph()
    nw_elems = {}

    def __init__(self, network_config=None):
        if network_config:
            self.config = Utils.read_config(network_config)
            for elem in self.config['elements']:
                ne_type = elem['type'].capitalize()
                if ne_type not in self.nw_elems:
                    self.nw_elems[ne_type] = []
                ne = getattr(gnpy, ne_type)(**elem)
                self.nw_elems[ne_type].append(ne)
                self.g.add_node(ne)

            for gpath in self.config['topology']:
                n0 = Utils.find_by_node_id(self.g, gpath[0])
                for nid in gpath[1:]:
                    n1 = Utils.find_by_node_id(self.g, nid)
                    self.g.add_edge(n0, n1, channels=Channels())
                    n0 = n1


class NetworkElement:

    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.type = kwargs.get('type')
        self.name = kwargs.get('name')
        self.description = kwargs.get('description')
        self.params = Params(self.required_params, kwargs)

    def __repr__(self):
        return self.id


class Fiber(NetworkElement):
    required_params = ['length', 'loss']


class Edfa(NetworkElement):
    required_params = ['gain', 'nf']


class Tx(NetworkElement):
    required_params = ['channels']


class Rx(NetworkElement):
    required_params = ['sensitivity']

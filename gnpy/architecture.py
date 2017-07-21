from networkx import DiGraph
from networkx.algorithms import all_simple_paths
from collections import namedtuple
from scipy.spatial.distance import cdist
from numpy import array
from itertools import product, islice, tee, count
from networkx import (draw_networkx_nodes,
                      draw_networkx_edges,
                      draw_networkx_labels,
                      draw_networkx_edge_labels,
                      spring_layout)
from matplotlib.pyplot import show, figure
from warnings import catch_warnings, simplefilter
from argparse import ArgumentParser

from logging import getLogger
logger = getLogger(__name__)

# remember me?
nwise = lambda g, n=2: zip(*(islice(g, i, None) for i, g in enumerate(tee(g, n))))

# here's an example that includes a little
#   bit of complexity to help suss out details
#   of the proposed architecture

# let's pretend there's a working group whose
#   job is to minimise latency of a network
# that working group gets the namespace LATENCY

# they interact with a working group whose
#   job is to capture physical details of a network
#   such as the geographic location of each node
#   and whether nodes are mobile or fixed
# that working group gets the namespace PHYSICAL

# each working group can put any arbitrary Python
#   object as the data for their given namespace

# the PHYSICAL group captures the following details
#   of a NODE: - whether it is mobile or fixed
#              - its (x, y) position
#   of an EDGE: - the physical length of this connection
#               - the speed of transmission over this link

# NOTE: for this example, we will consider network
#         objects to be MUTABLE just to simplify
#         the code
#       if the graph object is immutable, then
#         computations/transformations would return copies
#         of the original graph. This can be done easily via
#         `G.copy()`, but you'll have to account for the
#         semantic difference between shallow-vs-deep copy

# NOTE: we'll put the Node & Edge information for these
#         two working groups inside classes just for the
#         purpose of namespacing & just so that we can
#         write all the code for this example
#         in a single .py file: normally these pieces
#         would be in separate modules so that you can
#        `from tpe.physical import Node, Edge`

class Physical:
    # for Node: neither fixed nor position are computed fields
    #   - fixed cannot be changed (immutable)
    #   - position can be changed (mutable)
    class Node:
        def __init__(self, fixed, position):
            self._fixed = fixed
            self.position = position
        @property
        def fixed(self):
            return self._fixed
        @property
        def position(self):
            return self._position
        @position.setter
        def position(self, value):
            if len(value) != 2:
                raise ValueError('position must be (x, y) value')
            self._position = value
        def __repr__(self):
            return f'Node({self.fixed}, {self.position})'

    # for Edge:
    #   - speed (m/s) cannot be changed (immutable)
    #   - distance is COMPUTED
    class Edge(namedtuple('Edge', 'speed endpoints')):
        def distance(self, graph):
            from_node, to_node = self.endpoints
            positions = [graph.node[from_node]['physical'].position], \
                        [graph.node[to_node]['physical'].position]
            return cdist(*positions)[0][0]

    # NOTE: in this above, the computed edge data
    #         is computed on a per-edge basis
    #         which forces loops into Python
    #       however, the PHYSICAL working group has the
    #         power to decide what their API looks like
    #         and they could just as easily have provided
    #         some top-level function as part of their API
    #         to compute this "update" graph-wise
    @staticmethod
    def compute_distances(graph):
        # XXX: here you can see the general clumsiness of moving
        #        in and out of the `numpy`/`scipy` computation "domain"
        #        which exposes another potential flaw in our model
        #      our model is very "naturalistic": we have Python objects
        #         that match to real-world objects like routers (nodes)
        #         and links (edges)
        #      however, from a computational perspective, we may find
        #         it more efficient to work within a mathematical
        #         domain where are objects are simply (sparse) matrices of
        #         graph data
        #      moving between these "naturalistic" and "mathematical"
        #         domains can be very clumsy
        #      note that there's also clumsiness in that the "naturalistic"
        #         modelling pushes data storage onto individual Python objects
        #         such as the edge data dicts whereas the mathematical
        #         modelling keeps the data in a single place (probably in
        #         the graph data dict); moving data between the two is also clumsy
        data = {k:v['physical'].position for k, v in graph.node.items()}
        positions = array(list(data.values()))
        distances = cdist(positions, positions)

        # we can either store the above information back onto the graph itself:
        ##  graph['physical'].distances = distances

        # or back onto the edge data itself:
        ##  for (i, u), (j, v) in product(enumerate(data), enumerate(data)):
        ##      if (u, v) not in graph.edge:
        ##          continue
        ##      edge, redge = graph.edge[u][v], graph.edge[v][u]
        ##      dist = distances[i, j]
        ##      edge['physical'].computed_distance = dist

# as part of the latency group's API, they specify that:
#   - they consume PHYSICAL data
#   - they modify PHYSICAl data
#   - they do not add their own data
class Latency:
    @staticmethod
    def latency(graph, u, v):
        paths = list(all_simple_paths(graph, u, v))
        data = [(graph.get_edge_data(a, b)['physical'].speed,
                 graph.get_edge_data(a, b)['physical'].distance(graph))
                for path in paths
                for a, b in nwise(path)]
        return min(distance/speed for speed, distance in data)

    @staticmethod
    def total_latency(graph):
        return sum(Latency.latency(graph, u, v) for u, v in graph.edges())

    @staticmethod
    def nudge(u, v, precision=4):
        (ux, uy), (vx, vy) = u, v
        return (round(ux + (vx - ux) / 2, precision),
                round(uy + (vy - uy) / 2, precision),)

    @staticmethod
    def gradient(graph, nodes):
        # independently move each mobile node in the direction of one
        #   of its neighbors and compute the change in total_latency
        for u in nodes:
            for v in nodes[u]:
                upos, vpos = graph.node[u]['physical'].position, \
                             graph.node[v]['physical'].position
                new_upos = Latency.nudge(upos, vpos)
                before = Latency.total_latency(graph)
                graph.node[u]['physical'].position = new_upos
                after = Latency.total_latency(graph)
                graph.node[u]['physical'].position = upos
                logger.info(f'Gradient {u} ⇋ {v}; u to {new_upos}; grad {after-before}')
                yield u, v, new_upos, after - before

    # their public API may include only the following
    #   function for minimizing latency over a network
    @staticmethod
    def minimize(graph, *, n=5, threshold=1e-5 * 1e-9, d=None):
        mobile = {k: list(graph.edge[k]) for k, v in graph.node.items()
                                         if not v['physical'].fixed}
        # XXX: VERY sloppy optimization repeatedly compute gradients
        #        nudging nodes in the direction of the best latency improvement
        for it in count():
            gradients = u, v, pos, grad = min(Latency.gradient(graph, mobile),
                                              key=lambda rv: rv[-1])
            logger.info(f'Best gradient {u} ⇋ {v}; u to {pos}; grad {grad}')
            logger.info(f'Moving {u} in dir of {v} for {grad/1e-12:.2f} ps gain')
            graph.node[u]['physical'].position = pos
            if d:
                d.send((f'step #{it}', graph))
            if it > n or abs(grad) < threshold: # stop after N iterations
                break                           #   or if improvement < threshold

# our Network object is just a networkx.DiGraph
#   with some additional storage for graph-level
#   data
# NOTE: this may actually need to be a networkx.MultiDiGraph?
#         in the event that graphs may have multiple links
#         with distance edge data connecting them
def Network(*args, data=None, **kwargs):
    n = DiGraph()
    n.data = {} if data is None else data
    return n

def draw_changes():
    ''' simple helper to draw changes to the network '''
    fig = figure()
    for n in count():
        data = yield
        if not data:
            break
        for i, ax in enumerate(fig.axes):
            ax.change_geometry(n+1, 1, i+1)
        ax = fig.add_subplot(n+1, 1, n+1)
        title, network, *edge_labels = data
        node_data = {u: (u, network.node[u]['physical'].position)
                     for u in network.nodes()}
        edge_data = {(u, v): (network.get_edge_data(u, v)['physical'].distance(network),
                              network.get_edge_data(u, v)['physical'].speed,)
                     for u, v in network.edges()}
        labels = {u: f'{n}' for u, (n, p) in node_data.items()}
        distances = {(u, v): f'dist = {d:.2f} m\nspeed = {s/1e6:.2f}e6 m/s'
                     for (u, v), (d,s) in edge_data.items()}

        pos = {u: p for u, (_, p) in node_data.items()}
        label_pos = pos

        draw_networkx_edges(network, alpha=.25, width=.5, pos=pos, ax=ax)
        draw_networkx_nodes(network, node_size=600, alpha=.5, pos=pos, ax=ax)
        draw_networkx_labels(network, labels=labels, pos=pos, label_pos=.3, ax=ax)
        if edge_labels:
            draw_networkx_edge_labels(network, edge_labels=distances, pos=pos, font_size=8, ax=ax)

        ax.set_title(title)
        ax.set_axis_off()

    with catch_warnings():
        simplefilter('ignore')
        show()
    yield

parser = ArgumentParser()
parser.add_argument('-v', action='count')

if __name__ == '__main__':
    from logging import basicConfig, INFO
    args = parser.parse_args()
    if args.v:
        basicConfig(level=INFO)

    print('''
        Sample network has nodes:
            a ⇋ b ⇋ c ⇋ d

        signals a ⇋ b travel at speed of light through copper
        signals b ⇋ c travel at speed of light through water
        signals c ⇋ d travel at speed of light through water

        all connections are bidirectional

        a, c, d are fixed position
        b is mobile

        How can we move b to maximise speed of transmission a ⇋ d?
    ''')

    # create network
    n = Network()
    for name, fixed, (x, y) in [('a', True,  ( 0,  0)),
                                ('b', False, ( 5,  5)),
                                ('c', True,  (10, 10)),
                                ('d', True,  (20, 20)),]:
        n.add_node(name, physical=Physical.Node(fixed=fixed, position=(x,y)))
    for u, v, speed in [('a', 'b', 299790000),
                        ('b', 'c', 225000000),
                        ('c', 'd', 225000000),]:
        n.add_edge(u, v, physical=Physical.Edge(speed=speed, endpoints=(u, v)))
        n.add_edge(v, u, physical=Physical.Edge(speed=speed, endpoints=(v, u)))

    d = draw_changes(); next(d)
    d.send(('initial', n, True))

    # core computation
    latency = Latency.latency(n, 'a', 'd')
    total_latency = Latency.total_latency(n)
    Latency.minimize(n, d=d)
    total_latency = Latency.total_latency(n)

    print( 'Before:')
    print(f'  Current latency from a ⇋ d: {latency/1e-9:.2f} ns')
    print(f'  Total latency on n:         {total_latency/1e-9:.2f} ns')

    print( 'After:')
    print(f'  Total latency on n:         {total_latency/1e-9:.2f} ns')

    next(d)

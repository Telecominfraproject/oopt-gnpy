import matplotlib.pyplot as plt
from gnpy.utils import Utils
import networkx as nx
plt.rcdefaults()


# [print(d) for d in dir(gnpy) if '__' not in d]

config_fn = './gnpy/examples/config/config_ex1.json'
config, g = Utils.load_network(config_fn)


def find_by_node_id(g, nid):
    return next((n for n in g.nodes() if n.id == nid), None)


for gpath in config['topology']:
    n0 = find_by_node_id(g, gpath[0])
    for en, nid in enumerate(gpath[1:]):
        n1 = find_by_node_id(g, nid)
        g.add_edge(n0, n1)
        n0 = n1


graph_pos = nx.fruchterman_reingold_layout(g)
nx.draw_networkx_nodes(g, graph_pos, node_size=1000, node_color='b', alpha=0.2)
nx.draw_networkx_edges(g, graph_pos, width=2, alpha=0.3, edge_color='green')
nx.draw_networkx_labels(g, graph_pos, font_size=10)
plt.show()

import matplotlib.pyplot as plt
import networkx as nx
import gnpy
plt.rcdefaults()

config_fn = './gnpy/examples/config/config_ex1.json'
nw = gnpy.Network(config_fn)
nw.propagate_all_paths()

# output OSNR propagation
for path in nw.tr_paths:
    print(path.path)
    for edge in path.edge_list:
        print(edge, nw.g[edge[0]][edge[1]]['channels'])


if 0:
    layout = nx.spring_layout(nw.g)
    nx.draw_networkx_nodes(nw.g, layout, node_size=1000,
                           node_color='b', alpha=0.2)
    nx.draw_networkx_labels(nw.g, layout)
    nx.draw_networkx_edges(nw.g, layout, width=2,
                           alpha=0.3, edge_color='green')
    nx.draw_networkx_edge_labels(nw.g, layout, font_size=6)
    plt.show()

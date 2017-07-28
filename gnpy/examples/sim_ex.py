import matplotlib.pyplot as plt
import gnpy

config_fn = './gnpy/examples/config/config_ex1-copy.json'
nw = gnpy.Network(config_fn)
nw.propagate_all_paths()

# output OSNR propagation
edge_fmt = "ν: {frequency}\nOSNR:{osnr:^7.2f}\npow:{power:^8.1f}"
elabels = {}
for path in nw.tr_paths:
    print("Path", path.path)
    print("{:^8}{:^11}{:^9}{:^5}{:^11}".format("n0", "n1", "ν", "OSNR", "pow"))
    for edge in path.edge_list:
        spectrum = nw.g[edge[0]][edge[1]]['channels']
        pspec = [edge_fmt.format(**s) for s in spectrum]
        elabels[edge] = '\n'.join(pspec)
        print("{:^8}..{:^8}" .format(str(edge[0]), str(edge[1])),
              pspec[0].replace('\n', ' ').replace('OSNR', '')
              .replace('pow', "").replace("ν", "").replace(":", ""))

if 0:
    import networkx as nx
    layout = nx.spring_layout(nw.g)
    nx.draw_networkx_nodes(nw.g, layout, node_size=1000,
                           node_color='b', alpha=0.2, node_shape='s')
    nx.draw_networkx_labels(nw.g, layout)
    nx.draw_networkx_edges(nw.g, layout, width=2,
                           alpha=0.3, edge_color='green')
    nx.draw_networkx_edge_labels(
        nw.g, layout, edge_labels=elabels, font_size=10)
    plt.rcdefaults()
    plt.axis('off')
    plt.show()

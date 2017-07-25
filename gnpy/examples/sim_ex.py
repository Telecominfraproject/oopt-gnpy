import matplotlib.pyplot as plt
import networkx as nx
import gnpy
from pprint import pprint as pp
plt.rcdefaults()


config_fn = './gnpy/examples/config/config_ex1.json'
nw = gnpy.Network(config_fn)


def calc_path_osnr(nw, opath):
    print(opath)
    pp(opath[0].params.channels) 
    return None
    for en, chan in enumerate(opath[0].params.channels):
        for pen, p in enumerate(opath):
            print(en, type(p), p, pen)
        print("*** ")


def calc_osnr(nw):
    for tx in nw.nw_elems['Tx']:
        for rx in nw.nw_elems['Rx']:
            for opath in nx.all_simple_paths(nw.g, tx, rx):
                calc_path_osnr(nw, opath)


calc_osnr(nw)


if 0:
    graph_pos = nx.fruchterman_reingold_layout(nw.g)
    nx.draw_networkx_nodes(nw.g, graph_pos, node_size=1000, node_color='b', alpha=0.2)
    nx.draw_networkx_edges(nw.g, graph_pos, width=2, alpha=0.3, edge_color='green')
    nx.draw_networkx_labels(nw.g, graph_pos, font_size=10)
    plt.show()

import matplotlib.pyplot as plt
import networkx as nx
import gnpy
import numpy as np
from pprint import pprint as pp
plt.rcdefaults()

config_fn = './gnpy/examples/config/config_ex1.json'
nw = gnpy.Network(config_fn)


def propagate(opath):

    print(opath.edge_list)
    for elem in opath.path:
        print(elem, opath.elem_dict[elem])
        #try:
        if 1:
            elem.propagate(path=opath)
        #except Exception as e:
        #    print(e)
        #    pass


for opath in nw.tr_paths:
    propagate(opath)
    print("*"*10)

if 1:
    layout = nx.spring_layout(nw.g)
    nx.draw_networkx_nodes(nw.g, layout, node_size=1000,
                           node_color='b', alpha=0.2)
    nx.draw_networkx_labels(nw.g, layout)
    nx.draw_networkx_edges(nw.g, layout, width=2,
                           alpha=0.3, edge_color='green')
    nx.draw_networkx_edge_labels(nw.g, layout, font_size=6)
    plt.show()


exit()




def calc_path_osnr(nw, opath):
    for en, leg in enumerate(opath[:-1]):
        edge = nw.g[opath[en]][opath[en + 1]]
        suc_class = opath[en].__class__
        if suc_class == gnpy.network_elements.Tx:
            for chan in opath[en].params.channels:
                edge['channels'].append(edge_dict(chan, None, 0))
        elif suc_class == gnpy.network_elements.Fiber:
            attn = opath[en].params.length * opath[en].params.loss
            for inedge in nw.g.in_edges([opath[en]]):
                pedge = nw.g[inedge[0]][inedge[1]]
                for chan in pedge['channels']:
                    edge['channels'].append(edge_dict(chan, None, -attn))
        elif suc_class == gnpy.network_elements.Edfa:
            for inedge in nw.g.in_edges([opath[en]]):
                pedge = nw.g[inedge[0]][inedge[1]]
                gain = opath[en].params.gain
                for chan in pedge['channels']:
                    osnr = chan_osnr(chan, opath[en].params)
                    edge['channels'].append(edge_dict(chan, osnr, gain[0]))
        print(leg, edge['channels'])


def calc_osnr(nw):
    for tx in nw.nw_elems['Tx']:
        for rx in nw.nw_elems['Rx']:
            for opath in nx.all_simple_paths(nw.g, tx, rx):
                calc_path_osnr(nw, opath)
                print("*" * 10)


calc_osnr(nw)

if 1:
    layout = nx.spring_layout(nw.g)
    nx.draw_networkx_nodes(nw.g, layout, node_size=1000,
                           node_color='b', alpha=0.2)
    nx.draw_networkx_labels(nw.g, layout)
    nx.draw_networkx_edges(nw.g, layout, width=2,
                           alpha=0.3, edge_color='green')
    #nx.draw_networkx_edge_labels(nw.g, layout, font_size=10)
    plt.show()

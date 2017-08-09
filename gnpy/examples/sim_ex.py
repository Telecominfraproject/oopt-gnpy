from gnpy import Network
from gnpy.utils import read_config
from os.path import realpath, join, dirname

if __name__ == '__main__':
    basedir = dirname(realpath(__file__))
    filename = join(basedir, 'config/config_ex1.json')
    config = read_config(filename)
    nw = Network(config)
    nw.propagate_all_paths()

    # output OSNR propagation
    for path in nw.tr_paths:
        print(' → '.join(x.id for x in path.path))
        for u, v in path.edge_list:
            channels = nw.g[u][v]['channels']
            channel_info = ('\n' + ' ' * 24).join(
                '    '.join([f'freq: {x["frequency"]:7.2f}',
                             f'osnr: {x["osnr"]:7.2f}',
                             f'power: {x["power"]:7.2f}'])
                for x in channels)
            print(f'{u.id:^10s} → {v.id:^10s} {channel_info}')

    if 1:  # plot network graph
        import networkx as nx
        import matplotlib.pyplot as plt
        layout = nx.spring_layout(nw.g)
        nx.draw_networkx_nodes(nw.g, layout, node_size=1000,
                               node_color='b', alpha=0.2, node_shape='s')
        nx.draw_networkx_labels(nw.g, layout)
        nx.draw_networkx_edges(nw.g, layout, width=2,
                               alpha=0.3, edge_color='green')
        nx.draw_networkx_edge_labels(nw.g, layout, font_size=10)
        plt.rcdefaults()
        plt.axis('off')
        plt.show()

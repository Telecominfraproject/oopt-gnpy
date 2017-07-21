# import network_elements as ne
import networkx as nx
import matplotlib.pyplot as plt
plt.rcdefaults()


import network_elements.optical_elements as oe

wls = [1550]
rbw = 0.1

amps = [oe.Edfa(wavelengths=wls, rbw=rbw) for _ in range(3)]
fibers = [oe.Span(span_length=d*50) for d in range(1, 3)]

G = nx.DiGraph()
G.add_nodes_from(fibers)
G.add_nodes_from(amps)



print(G.nodes())


graph_pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='b', alpha=0.2)
#nx.draw_networkx_edges(G, graph_pos, edgelist=edges[0], width=2, alpha=0.3, edge_color='green')
nx.draw_networkx_labels(G, graph_pos, font_size=10)
plt.show()




exit()


fbrs = [ne.Fiber(100, 0.2), ne.Fiber(80, 0.21)]
amps = [ne.Amp(20, 5), ne.Amp(18, 5.1), ne.Amp(21, 5.3)]

G = nx.DiGraph()


G.add_nodes_from(fbrs)

#for f in fbrs:
#    G.add_node(f)
#for a in amps:
#    G.add_node(a)




graph_pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='b', alpha=0.2)
#nx.draw_networkx_edges(G, graph_pos, edgelist=edges[0], width=2, alpha=0.3, edge_color='green')
nx.draw_networkx_labels(G, graph_pos, font_size=10, font_family='sans-serif')
plt.show()

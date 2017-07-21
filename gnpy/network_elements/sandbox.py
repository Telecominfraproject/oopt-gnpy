#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:29:12 2017

@author: briantaylor
"""

import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()

G.add_node(1)

G.add_nodes_from([2, 3])
H = nx.path_graph(10)
G.add_nodes_from(H)
G = nx.path_graph(8)
nx.draw_spring(G)
plt.show()



class NetworkElement(nx.node)


import networkx as nx
import matplotlib.pyplot as plt


fixed_position = {1:(2.0, 5.5), 2:(3.1, 5.1), 3:(1.2, 4.1), 4:(2.3, 3.7), 5:(3.5, 3.7),
                   6:(3.5, 2.0), 7:(1.5, 2.0), 8:(2.5, 1.3), 9:(4.0, 0.5), 10:(3.0, 0)}

def show_mincut(my_graph, edges_to_color):

    pos = nx.spring_layout(my_graph, pos=fixed_position, fixed = fixed_position.keys())

    fig, ax = plt.subplots(figsize=(5,4))
    nx.draw_networkx_nodes(my_graph, node_size=750, alpha = 0.15, node_color='green', pos = pos)


    nx.draw_networkx_edges(my_graph, edgelist= edges_to_color, edge_color = "red", pos=pos, style=':', )

    nx.draw_networkx_edges(my_graph, alpha=0.15, edge_color = "black", pos=pos, style='-', )

    text = nx.draw_networkx_labels(my_graph, pos, font_size=12, font_color='black', font_weight='bold')

    _ = ax.axis('off')


def show_sparsest_cut(my_graph, nodi_1, nodi_2):

    fig, ax = plt.subplots(figsize=(5,4))

    pos = pos = nx.spring_layout(my_graph, pos=fixed_position, fixed = fixed_position.keys())

    nx.draw_networkx_nodes(my_graph, nodelist = nodi_1, node_size=750, alpha = 0.15, node_color='green', pos = pos)
    nx.draw_networkx_nodes(my_graph, nodelist = nodi_2, node_size=750, alpha = 0.15, node_color='blue', pos = pos)

    common = [(u,v) for u,v in my_graph.edges(nodi_1) if u in nodi_1 and v in nodi_2]
    nx.draw_networkx_edges(my_graph, edgelist=common, edge_color = "red", pos=pos, style=':', )

    nx.draw_networkx_edges(my_graph, edgelist=[(u,v) for u,v in my_graph.edges(nodi_1) if u and v in nodi_1],
                            edge_color = "#8fce00", pos=pos)

    nx.draw_networkx_edges(my_graph, edgelist=[(u,v) for u,v in my_graph.edges(nodi_2) if u and v in nodi_2],
                            edge_color = "#8e7cc3", pos=pos)

    text = nx.draw_networkx_labels(my_graph, pos, font_size=12, font_color='black', font_weight='bold')

    _ = ax.axis('off')



import matplotlib.pyplot as plt

def show_subnet(my_class, show, output_directory='plot_folder/community', ):

    this_graph = my_class.subgraph.to_undirected()

    palette = plt.get_cmap('tab20', my_class.n_of_subclass)
    colors = [palette(i) for i in range(my_class.n_of_subclass)]

    unique_labels = list(set(my_class.node_labels.values()))
    color_match = dict(zip(unique_labels, colors))

    node_colors = {k: color_match[v] for k, v in my_class.node_labels.items()}

    plt.figure(figsize=(10, 5))

    nx.draw_networkx_nodes(this_graph, my_class.pos, node_color = list(node_colors.values()), node_size=25, alpha=0.70)
    nx.draw_networkx_edges(this_graph, my_class.pos, edge_color = "gray", alpha=0.25, style="-")
    
    legend_labels = {topic + ": "+ str(my_class.topic_count[topic]): color for i, (topic, color) in enumerate(color_match.items())}

    plt.legend(handles=[plt.Line2D([], [], color=color, marker='o', linestyle='', markersize=5, label=cat) for cat, color in legend_labels.items()], 
              loc='upper left', bbox_to_anchor=(0.95,0.80), prop={'size': 6.2})


    plt.title(f"'{my_class.topic_name}' Articles from Wikipedia")
    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists(output_directory):
      os.makedirs(output_directory)

    plt.savefig(output_directory + f'/{my_class.topic_name.lower()}_network.png')

    if show: plt.show()
    else: plt.close()




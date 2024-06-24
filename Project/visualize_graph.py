
## -------------------------------- ##
import os                           ##
import numpy as np                  ##            
import networkx as nx               ##               
import matplotlib.pyplot as plt     ##                       
## -------------------------------- ##


## -------------------------------- ##
## ------------------------------------------------------------------------------ ##
def show_subnet(my_class, show, output_directory='plot_folder/community'):

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
## -------------------------- ##
## ------------------------------------------------------------------------------ ##


## ------------------------------------------------------------------------------------------------- ##
def show_communities(graph_handler, partizione, show=False, output_directory="plot_folder/louvain"):

  this_graph = graph_handler.subgraph.to_undirected()

  lengths = [len(x) for x in partizione]

  community_dict = {}
  for i, community in enumerate(partizione):
    for node in community:
        community_dict[node] = i

  unique_communities = list(set(community_dict.values()))

  colors = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))
  community_color_map = {community: colors[i] for i, community in enumerate(unique_communities)}

  node_colors = [community_color_map[community_dict[node]] for node in graph_handler.sub_nodes]

  plt.figure(figsize=(10, 5))

  nx.draw_networkx_nodes(this_graph, graph_handler.pos, node_color=node_colors, node_size=50, alpha=0.8)
  nx.draw_networkx_edges(this_graph, graph_handler.pos, alpha=0.3, edge_color="gray")

  handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
  
  labels = [f"Community {i}: "+str(lengths[i]) for i in unique_communities]
  plt.legend(handles, labels, loc='best')

  plt.title(f"Louvain Community Detection on '{graph_handler.topic_name}'")
  plt.axis('off')

  if not os.path.exists(output_directory):
      os.makedirs(output_directory)

  plt.savefig(output_directory + f'/louvain_{graph_handler.topic_name.lower()}.png')

  if show: plt.show()
  else: plt.close()
## -------------------------- ##
## ------------------------------------------------------------------------------------------------- ##


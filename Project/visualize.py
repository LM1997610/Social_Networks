
import os
from tabulate import tabulate

import matplotlib.pyplot as plt




def visualize_degree_distrib(nodes, degree_distrib, output_dir='plot_folder/'):
  fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 3))
  
  ax1.plot(nodes, degree_distrib, '-', linewidth=2.5)
  ax1.set_title('\n Degree Distribution\n')
  ax1.set_xlabel('Degree\n')
  ax1.set_ylabel('Number of Nodes\n')
  ax1.grid()
  
  ax2.plot(nodes, degree_distrib, 'o', markersize=3.5, color='green')
  ax2.set_title('\n Degree Distribution (log-log scale)\n')
  ax2.set_xlabel('Degree\n(log scale)')
  ax2.set_ylabel('Number of Nodes\n(log scale)')
  ax2.set_xscale("log"); ax2.set_yscale("log")
  ax2.grid()
  
  plt.subplots_adjust(wspace=0.3)
  #plt.tight_layout()

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  plt.savefig(output_dir+"degree_distrib.png")
  plt.show()


def show_length_of_paths(sorted_counter, output_dir='plot_folder/'):

  new_dict = {str(key): value for key, value in sorted_counter.items()}
  new_dict["path\nnot\nfound"] = new_dict.pop('0')
  
  keys = list(new_dict.keys())
  values = list(new_dict.values())

  plt.bar(keys, values)

  plt.grid()
  plt.xticks(keys)
  plt.xlabel("Length of Paths \n", fontsize= 14)
  plt.ylabel("N. of Articles \n", fontsize= 14)
  plt.bar(keys[-1], values[-1], color='orange')
  plt.tight_layout()

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  plt.savefig(output_dir+"length_of_paths.png")
  plt.show()



def deg_separation_plot(data, total_nodes, output_dir='plot_folder/'):

  table_data = []

  for key, value_list in data.items():
    
    percentages = [str((round((size / total_nodes) * 100, 5)))+" %" for size in value_list]
    table_data.append([key] + percentages)

    plt.plot(range(len(value_list)), value_list, alpha = 0.65, label=key)

  headers = ["Node"] + [f"Hop {i}" for i in range(len(value_list))]
  this_table = tabulate(table_data, headers=headers, tablefmt="pretty")

  plt.gca().spines['right'].set_color('none')
  plt.gca().spines['top'].set_color('none')

  plt.title("\n\n")
  plt.xlabel("\n Degrees of Separation \n", fontsize=14)
  plt.ylabel("N. of Articles \n", fontsize= 14)


  plt.grid()
  plt.legend(loc ="lower right", prop={'size': 9})
  plt.tight_layout()

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  plt.savefig(output_dir+"degrees_separation.png")
  plt.show()

  return data, this_table



def dead_end_plot(G, dead_node, names):

  predecessors = list(G.predecessors(dead_node)

  subgraph_nodes = predecessors + [dead_node]
  H = G.subgraph(subgraph_nodes)

  pos = nx.shell_layout(H) 

  node_colors = ['#e6b800' if node == dead_node else '#0099cc' for node in H.nodes()]
  labels={node: names[node] for node in H.nodes()}

  plt.figure(figsize=(9, 4))
  nx.draw_networkx_nodes(H, node_size=350, alpha = 0.75, node_color= node_colors, pos = pos)
  nx.draw_networkx_edges(H, edge_color = "black", alpha = 0.35, pos=pos )

  for node, (x, y) in pos.items():
    plt.text(x, y + 0.05, labels[node], fontsize=12, color='black', ha='center')

  plt.axis('off')
  plt.show()
  

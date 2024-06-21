
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tabulate import tabulate

def topic_distrib_plot(counted_topics, output_directory='plot_folder'):

  sorted_data = dict(sorted(counted_topics.items(), key=lambda item: item[1]))

  plt.figure(figsize=(8, 4))

  colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_data)))
  colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_data)))  # Utilizzare la colormap rainbow

  plt.barh(list(sorted_data.keys()), list(sorted_data.values()), color=colors)

  plt.grid(axis='x', linestyle='--', alpha=0.7)
  plt.title("\n Wikipedia Articles by Topics \n", fontsize=14)
  plt.xlabel("\n N. of Articles \n", fontsize= 12)
  plt.tight_layout()

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  plt.savefig(output_directory + "/topic_distrib.png")
  plt.show()
  
def visualize_degree_distrib(nodes, degree_distrib, output_directory = 'plot_folder'):
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
  
  # plt.subplots_adjust(wspace=0.3)
  plt.tight_layout()

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  plt.savefig(output_directory + "/degree_distrib.png")
  plt.show()

def show_length_of_paths(sorted_counter, output_directory='plot_folder'):

  new_dict = {str(key): value for key, value in sorted_counter.items()}
  new_dict["Not\nExist"] = new_dict.pop('0')
  
  keys = list(new_dict.keys())
  values = list(new_dict.values())
  print()

  plt.figure(figsize=(6, 4))
  plt.bar(keys, values, color = "#34c7ea")

  plt.grid(linestyle='--', alpha=0.7)
  plt.xticks(keys)
  plt.xlabel("Length of Paths \n", fontsize= 14)
  plt.ylabel("N. of Articles \n", fontsize= 14)
  plt.bar(keys[-1], values[-1], color='#b30000')
  plt.tight_layout()

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  plt.savefig(output_directory + "/length_of_paths.png")
  plt.show()

  print(); print()
  show_data = [  ["Lenght of path:"] + [str(x) if x!=0 else "Not Exist" for x in sorted_counter.keys()],
                 ["Nodes (couples):", *[str(sorted_counter[i]) for i in range(len(sorted_counter))] ]]

  t = tabulate(show_data, headers="firstrow",tablefmt="fancy_grid", numalign="center")
  print(t)

def deg_separation_plot(data, total_nodes, output_directory = 'plot_folder'):

  table_data = []

  for key, value_list in data.items():
    
    percentages = [str((round((size / total_nodes) * 100, 5)))+" %" for size in value_list]
    table_data.append([key] + percentages)

    plt.plot(range(len(value_list)), value_list, alpha = 0.65, label=key)

  headers = ["Node"] + [f"Hop {i}" for i in range(len(value_list))]
  this_table = tabulate(table_data, headers=headers, tablefmt="pretty")
  unreachable = 100 - float(percentages[-1].split()[0])

  plt.gca().spines['right'].set_color('none')
  plt.gca().spines['top'].set_color('none')

  plt.title("\n\n")
  plt.xlabel("\n Degrees of Separation \n", fontsize=14)
  plt.ylabel("N. of Articles \n", fontsize= 14)


  plt.grid()
  plt.legend(loc ="lower right", prop={'size': 9})
  plt.tight_layout()

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  plt.savefig(output_directory + "/degrees_separation.png")
  plt.show()

  return data, this_table, round(unreachable, 4)



def dead_end_plot(G, dead_node, names):

  predecessors = list(G.predecessors(dead_node))

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
  

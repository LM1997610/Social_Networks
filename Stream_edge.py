
import networkx as nx
import matplotlib.pyplot as plt
import imageio


# ------------------------------------------------- #
def check_add_edge(u,v, edge_set):
  
  nodes = ['a', 'b', 'c', 'd', 'e' ]

  graph = nx.Graph()
  graph.add_nodes_from(nodes)
  graph.add_edges_from(edge_set)

  if u == v: return False
  elif not graph.has_edge(u, v): return True
  else: return False
# ------------------------------------------------- #


# -------------------------------------------------------------- #
def do_gif(folder_path, n_img):

    filenames = [f'{folder_path}/time_{x}.png' for x in n_img]

    with imageio.get_writer('streaming_algorithm.gif', mode='I', duration=400) as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
# -------------------------------------------------------------- #


def do_plot(edge_list, window, time, folder_path, discon = False):
  
  nodes = ['a', 'b', 'c', 'd', 'e', ]

  if len(window) > 5: print("ERROEEE window", len(window))
  if len(edge_list) > 4: print("ERROEEE edge", len(edge_list), edge_list)


  G = nx.Graph()
  G.add_nodes_from(nodes)
  G.add_edges_from(window)

  plt.figure(figsize=(6, 5))
  plt.title(f"\n Time: {time} \n")

  pos = {"a":(4, 4), "b":(6, 4), "c":(3.5, 2), "d":(6.5, 2), "e":(5, 1)}
  
  #pos = {"a":(4, 4), "b":(6, 4), "c":(3.4, 2.8), "d":(6, 1.5), "e":(5, 1), "f":(3.7, 1.2), "g": (6.5, 3)}
                   
  nx.draw_networkx_nodes(G, node_color='white', node_size=250, pos=pos, edgecolors="black")
  nx.draw_networkx_labels(G, pos, font_size=14, font_color='black', font_weight='bold')

  nx.draw_networkx_edges(G, edge_color = "lightgreen", pos=pos, arrows=True, width=2, edgelist= list(G.edges()))
  nx.draw_networkx_edges(G, edge_color = "blue", edgelist=edge_list, pos=pos, arrows=True, connectionstyle="arc3,rad=0.05", alpha=0.85)

  _ = plt.axis('off')

  if discon:
      nodo_isolato = list(nx.isolates(G))
      nx.draw_networkx_nodes(G, node_color='red', node_size=250, pos=pos, edgecolors="black", nodelist= nodo_isolato, alpha=0.5)

      
  plt.savefig(f'{folder_path}/time_{time}.png',  bbox_inches="tight")
  plt.close()

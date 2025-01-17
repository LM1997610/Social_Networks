
import os 

import numpy as np
import networkx as nx

from tqdm import tqdm
from collections import Counter


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

class SubgraphBuilder:

    def __init__(self, main_graph, topic_name, label_id_diz, wikivitals_labels, wikivitals_labels_hierarchy, wikivitals_names_labels_hierarchy):
    
        self.topic_name = topic_name.split()[0]
        self.topic_long_name = topic_name
        self.topic_id = label_id_diz[self.topic_long_name]
        
        self.subgraph, self.pos = self.build_subgraph(main_graph, wikivitals_labels)

        self.n_of_nodes = self.subgraph.number_of_nodes()
        self.n_of_edges = self.subgraph.number_of_edges()
        
        self.node_labels = self.get_sub_labels(wikivitals_labels_hierarchy, wikivitals_names_labels_hierarchy)
        self.n_of_subclass = len(set(self.node_labels.values()))
        self.topic_count = dict(Counter(self.node_labels.values()))

        self.avg_clustering = round(nx.average_clustering(self.subgraph.to_undirected()), 3)
        
        self.info = ("["+self.topic_name+"]", self.n_of_nodes, 
                      self.n_of_edges, self.n_of_subclass, 
                      self.n_of_isolated_nodes,self.n_connected_components)

    def build_subgraph(self, main_graph, wikivitals_labels):

        self.sub_nodes = [x for x, v in enumerate(wikivitals_labels) if v == self.topic_id]
        sub_graph = main_graph.subgraph(self.sub_nodes)

        self.n_of_isolated_nodes = 0
        self.isolated_nodes = [node for node in sub_graph.nodes if sub_graph.degree(node) == 0]

        if self.isolated_nodes: 
            self.n_of_isolated_nodes = len(self.isolated_nodes)
            self.sub_nodes = [x for x in self.sub_nodes if x not in self.isolated_nodes]

            one_time_copy = sub_graph.copy()
            one_time_copy.remove_nodes_from(self.isolated_nodes)
            sub_graph = one_time_copy
            
        self.n_connected_components = 1
        self.connected_components = list(nx.connected_components(sub_graph.to_undirected()))
        
        if len(self.connected_components) > 1:
          self.n_connected_components = len(self.connected_components)
          self.unpacked_list = [element for subset in self.connected_components[1:] for element in subset]
          sub_graph.remove_nodes_from(self.unpacked_list)
          self.sub_nodes = [x for x in self.sub_nodes if x not in self.unpacked_list]

        pos = nx.spring_layout(sub_graph.to_undirected())

        return sub_graph, pos

    def get_sub_labels(self, wikivitals_labels_hierarchy, wikivitals_names_labels_hierarchy):
        node_labels = {}

        for node in self.sub_nodes:
            this_node_sublabel = wikivitals_labels_hierarchy[node]
            this_label_name = wikivitals_names_labels_hierarchy[this_node_sublabel]
            name = this_label_name.split("|||")
            node_labels[node] = name

        assert all(category == self.topic_long_name for category in [sublist[0] for sublist in node_labels.values()])
        result = {node: label_name[1] for node, label_name in node_labels.items()}
        return result

    def compute_distribution(self):

      degrees = [self.subgraph.degree(n) for n in self.subgraph.nodes()]
      degrees = dict(Counter(degrees))
      degrees = dict(sorted(degrees.items(), key=lambda item: item[0], reverse=True))

      out_deg = [len(list(self.subgraph.successors(n))) for n in self.subgraph.nodes()]
      out_deg = dict(Counter(out_deg))
      out_deg = dict(sorted(out_deg.items(), key=lambda item: item[0], reverse=True))

      in_deg = [len(list(self.subgraph.predecessors(n))) for n in self.subgraph.nodes()]
      in_deg = dict(Counter(in_deg))
      in_deg = dict(sorted(in_deg.items(), key=lambda item: item[0], reverse=True))

      return degrees, out_deg, in_deg

    def get_articles_names(self, some_dict, wikivitals_names):
      names_dict = {wikivitals_names[k]:v for k,v in some_dict.items()}
      return names_dict

    def get_top_centralities(self, centrality_func,  wikivitals_names, top_n=3):

      centrality = centrality_func(self.subgraph.to_undirected())

      ##
      self.do_new_plot(centrality.values(), centrality_func.name)
      ##
      top_centrality = dict(sorted(centrality.items(), key=lambda item: item[1], reverse=True)[:top_n])
      article_names = self.get_articles_names(top_centrality, wikivitals_names)
      article_names = {k:round(v, 3) for k,v in article_names.items()}
      return article_names

    def get_all_centralities(self, wikivitals_names):
      
      self.top_articles = {}
      centrality_list = [nx.closeness_centrality, nx.betweenness_centrality, nx.degree_centrality, nx.pagerank]

      for nx_function in tqdm(centrality_list):
        self.top_articles[nx_function.name] = self.get_top_centralities(nx_function, wikivitals_names)

      return self.top_articles

    def find_hubs(self, percentile = 95):

      degrees = dict(self.subgraph.degree())

      degree_values = np.array(list(degrees.values()))
      threshold = np.percentile(degree_values, percentile)

      self.hubs = {node: degree for node, degree in degrees.items() if degree > threshold}
      return self.hubs

    def visualize_hubs(self, output_directory = 'plot_folder/centrality_measures'):

      this_g = self.subgraph.to_undirected()

      fig = plt.subplots(figsize=(12, 8))

      nx.draw_networkx_nodes(this_g, self.pos, node_size=15, alpha=0.75)

      hub_nodes = list(self.hubs.keys())
      nx.draw_networkx_nodes(this_g, self.pos, nodelist=hub_nodes, node_color='#C01818', node_size=50, alpha=0.9)

      nx.draw_networkx_edges(this_g, self.pos, edge_color='gray', alpha=0.3)

      plt.title(f"'{self.topic_name}' Network Hubs", fontsize= 14)
      plt.axis('off')

      if not os.path.exists(output_directory):
        os.makedirs(output_directory)

      plt.savefig(output_directory + f'/{self.topic_name.lower()}_hubs.png')
      plt.close()
    
    def do_new_plot(self, deg_centrality_values, c_misura, output_directory = "plot_folder/centrality_measures"):
      
      cent = np.fromiter(deg_centrality_values, float)
      sizes = cent / np.max(cent) * 200
      normalize = mcolors.Normalize(vmin=cent.min(), vmax=cent.max())
      colormap = cm.viridis

      scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
      scalarmappaple.set_array(cent)

      fig, ax = plt.subplots(figsize=(12, 8))

      plt.colorbar(scalarmappaple, ax=ax)
      
      nx.draw_networkx_nodes(self.subgraph.to_undirected(), self.pos, node_size=sizes, node_color=sizes, alpha=0.7, ax=ax)
      nx.draw_networkx_edges(self.subgraph.to_undirected(), self.pos, edge_color='gray', alpha=0.3, ax=ax)

      plt.title(f"{self.topic_name} Graph based on {c_misura}")
      plt.axis('off')
      
      if not os.path.exists(output_directory):
        os.makedirs(output_directory)

      plt.savefig(output_directory + f'/{self.topic_name.lower()}_{c_misura.split("_")[0]}.png')
      plt.close()
    
    

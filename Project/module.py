import os 

import networkx as nx

from tqdm import tqdm
from collections import Counter

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

      centrality = centrality_func(self.subgraph)
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
    

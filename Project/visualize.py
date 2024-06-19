import matplotlib.pyplot as plt



def visualize_degree_distrib(nodes, degree_distrib):
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
  plt.show()


def show_length_of_paths(sorted_counter):

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
  plt.show()

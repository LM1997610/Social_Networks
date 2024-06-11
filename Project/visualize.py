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

import numpy as np
from tree import Tree

number_nodes = 10
config = {'number_segments':10, 'p_new_event': 0.3}
config['n_reads_sample'] = 1000
config['length_segments'] = np.ones(config['number_segments'])
config['n_samples'] = 10

tree = Tree(number_nodes,config)
tree.generate_events()
tree.generate_samples()
print(tree.get_log_posterior())
print(tree)
import numpy as np
from tree import Tree

# This code samples a random phylogeny with the parameters
# number of nodes: number_nodes 
# number of segments in the genome: number_segments (at the moment only 1 chromosome)
# length of the segments: at the moment all have the same length
# number of single cell or minibulk samples: n_samples
# number of reads per sample: n_reads_sample (at the moment same number of reads per sample)

number_nodes = 20
config = {'number_segments':10, 'p_new_event': 0.8}
config['n_reads_sample'] = 1000
config['length_segments'] = np.ones(config['number_segments'])
config['n_samples'] = 10

tree = Tree(number_nodes,config) # generates a tree with random topology
tree.generate_events() # generates a tree with random topology
tree.generate_samples() # Assigns randomly a sample to a node and samples the reads per segment from the copy number profile
print(tree.get_log_posterior()) #prints the (unnormalised up to an additive constant) log posterior distribution, at the moment the number of nodes is fixed
print(tree) #prints the tree
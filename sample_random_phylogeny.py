import numpy as np
from tree.tree import Tree

# This code samples a random phylogeny with the parameters
# number of nodes: number_nodes 
# number of segments in the genome: number_segments (at the moment only 1 chromosome)
# length of the segments: at the moment all have the same length
# number of single cell or minibulk samples: n_samples
# number of reads per sample: n_reads_sample (at the moment same number of reads per sample)

# events = sample_events([{'start':3,'len':2},{'start':6,'len':1},{'start':8,'len':5}],0.95)
# for ev in events:
# 	print(ev)
# exit()
number_nodes = 10
number_samples = 10
n_reads_sample = 1000*np.ones(number_samples)

config = {'number_segments':10, 'p_new_event': 0.999}
config['length_segments'] = np.ones(config['number_segments'])
 

tree = Tree(number_nodes,config) # generates a tree with random topology
tree.generate_events() # generates a tree with random topology
tree.generate_samples(n_reads_sample) # Assigns randomly a sample to a node and samples the reads per segment from the copy number profile
print(tree.get_log_posterior()) #prints the (unnormalised up to an additive constant) log posterior distribution, at the moment the number of nodes is fixed
print(tree)

read_counts = []
for node in tree.nodes:
	for sample in node.samples:
		read_counts.append(sample.read_count)

tree2 = Tree(number_nodes,config) # generates a tree with random topology
tree2.randomly_assign_samples(read_counts)
print(tree2.get_log_posterior()) #prints the (unnormalised up to an additive constant) log posterior distribution, at the moment the number of nodes is fixed
print(tree2)

# for i in range(10000):
# 	tree = Tree(number_nodes,config) # generates a tree with random topology
# 	tree.generate_events() # generates a tree with random topology
# 	tree.generate_samples() # Assigns randomly a sample to a node and samples the reads per segment from the copy number profile
# 	print(tree.get_log_posterior()) #prints the (unnormalised up to an additive constant) log posterior distribution, at the moment the number of nodes is fixed
# 	if math.isnan(tree.get_log_posterior()):
# 		print(tree)
# 		for sample in tree.get_samples():
# 			print(sample.node.id_,sample.get_log_likelihood())
# 		break 


from tree.tree import Tree
from inference.moves import move,p_add_event
import numpy as np
import copy

#TODO: use random number generator of numpy instead!
np.random.seed(2022)

def inference():
	pass
	#propose move (7 possible ones)
	#	- add event
	#	- remove event
	#	- swap nodes events
	#	- change sample assigment
	#	- prune_and_reatach
	#	- add new node
	#	- merge 2 nodes

def inference_number_nodes_known():
	pass
	#propose move (5 possible ones)
	#	- add event
	#	- remove event
	#	- swap nodes events
	#	- change sample assigment
	#	- prune_and_reatach
		
def inference_tree_topology_known(tree_start,config,n_iter):
	pass
	#propose move (4 possible ones)
	#	- add event
	#	- remove event
	#	- swap nodes events
	#	- change sample assigment
	
def inference_tree_topology_and_sample_assigments_known(tree_start,move_weights,config,n_iter):
	moves_allowed = ["swap_events_2_nodes", "add_event", "remove_event"]
	list_trees = [copy.deepcopy(tree_start)]
	tree = tree_start
	p = [move_weights[mv] for mv in moves_allowed]
	p = np.array(p)/np.sum(p)

	for i in range(n_iter):
		move_type = np.random.choice(moves_allowed,p = p)
		move_type = "add_event" #TODO remove
		new_tree,info = move(tree,move_type)
		print("move_type",move_type)
		print("info", info)
		if move_type == "swap_events_2_nodes":
			#symetrical move
			log_proposal_ratio = 0
		else:
			event = info["event"]
			for n in tree.nodes:
				if n.id_ == info["node.id_"]:
					node = n
					break

			if move_type == "add_event":
				log_p_forw_mv = np.log(move_weights[move_type]) + p_add_event(tree,node,event)
				log_p_back_mv = np.log(move_weights["remove_event"]) - np.log(new_tree.get_number_events())
			else:
				log_p_forw_mv = np.log(move_weights[move_type]) - np.log(new_tree.get_number_events())
				log_p_back_mv = np.log(move_weights["remove_event"]) + p_add_event(new_tree,node,event)
			log_proposal_ratio = log_p_back_mv-log_p_forw_mv 

		log_density_ratio = new_tree.get_log_posterior()-tree.get_log_posterior()
		p_accept = min(1, np.exp(log_density_ratio+log_proposal_ratio))
		if np.random.choice(2, p=[1-p_accept,p_accept]):
			list_trees.append(copy.deepcopy(new_tree))
			tree = new_tree
		print(i, 'p_acceptance:', (len(list_trees)-1)/(i+1))

def inference_tree_known(tree_start,config,n_iter):
	list_trees = [copy.deepcopy(tree_start)]
	tree = tree_start
	for i in range(n_iter):
		new_tree = move(tree,"modify_sample_attachments")
		#symetrical move and only modifies likelihood
		#TODO: I should improve this since only 2 terms in the likelihood are modified
		p_accept = min(1, np.exp(new_tree.get_log_likelihood()-tree.get_log_likelihood()))
		print(tree.get_log_likelihood())
		if np.random.choice(2, p=[1-p_accept,p_accept]):
			list_trees.append(copy.deepcopy(new_tree))
			tree = new_tree

		print(i, 'p_acceptance:', (len(list_trees)-1)/(i+1))

	return list_trees

def simulation_tree_known(number_nodes,
	n_reads_sample,
	config,
	n_iter):

	tree = Tree(number_nodes,config) # generates a tree with random topology
	tree.generate_events() # generates a tree with random topology
	tree_start = copy.deepcopy(tree)
	tree.generate_samples(n_reads_sample) # Assigns randomly a sample to a node and samples the reads per segment from the copy number profile
	read_counts = []
	for node in tree.nodes:
		for sample in node.samples:
			read_counts.append(sample.read_count)
	tree_start.randomly_assign_samples(read_counts)
	list_trees = inference_tree_known(tree_start,config,n_iter)
	for i,t in enumerate(list_trees):
		print(i,t.get_log_likelihood())

	print(tree_start)
	print(tree)
	print(list_trees[-1])

def simulation_tree_topology_and_sample_assigments_known(number_nodes,
	n_reads_sample,
	config,
	n_iter,
	move_weights):
	
	tree = Tree(number_nodes,config) # generates a tree with random topology
	tree.generate_events() # generates a tree with random topology
	tree.generate_samples(n_reads_sample)
	tree_start = copy.deepcopy(tree)
	for node in tree_start.nodes:
		node.events = []
	tree_start._update_profiles()
	list_trees = inference_tree_topology_and_sample_assigments_known(tree_start,move_weights,config,n_iter)
	for i,t in enumerate(list_trees):
		print(i,t.get_log_posterior())
	print(tree_start)
	print(tree)
	print(list_trees[-1])


number_nodes = 9
number_samples = 10
n_reads_sample = 1000*np.ones(number_samples)
config = {'number_segments':8, 'p_new_event': 0.75}
config['length_segments'] = np.ones(config['number_segments'])
n_iter = 1000

move_weights = {"swap_events_2_nodes":1,"add_event":1,"remove_event":1,"modify_sample_attachments":1}

simulation_tree_topology_and_sample_assigments_known(number_nodes,
	n_reads_sample,
	config,
	n_iter,
	move_weights)

#tree_modified = move(tree,"add_event") 
#tree_modified = move(tree,"swap_events_2_nodes")
#tree_modified = move(tree,"remove_event") 
#tree_modified = move(tree,"modify_sample_attachments")
#print(tree)
# for node in tree.nodes:
# 	print(node.id_, [str(event) for event in node.events])
# print(tree_modified)
# for node in tree_modified.nodes:
# 	print(node.id_, [str(event) for event in node.events])
# print(tree.get_log_posterior()) #prints the (unnormalised up to an additive constant) log posterior distribution, at the moment the number of nodes is fixed
# print(tree_modified.get_log_posterior()) #prints the (unnormalised up to an additive constant) log posterior distribution, at the moment the number of nodes is fixed


from tree.tree import Tree
import copy

def inference(number_nodes,read_counts,config,n_iter):
	tree = Tree(number_nodes,config) # generates a tree with random topology
	tree.randomly_assign_samples(read_counts)

	list_trees = [copy.deepcopy(tree)]

	for i in range(n_iter):
		pass
		
		#propose move
		
		#compute probability acceptance
		
		#if accepted, update tree and add it to list_trees


	#moves:
		#prune and reatach
			# choose 1 node at random
			# choose reatachment point at random
		#swap nodes
			#choose 2 nodes at random and swap them
		#add event
			# choose node at random
			# from the possible events, sample 1 uniformly at random
		#remove event:
			# choose node at random
			# remove 1 event at random
		#modify event:
			# choose node at random
			# choose 1 event at random
			# chose side at random
			# lengthen or shorten event by 1 segment
		#change sample attachments:
			# choose sample at random
			# choose new node attachment at random
import numpy as np
import copy

from node import Node
from sample import Sample

#TODO: for now 1 diploid chromosome, need to change that

class Tree():
	"""
	Class used to represent a phylogeny

	...

	Attributes
	----------
	number_nodes: int
		Number of nodes in the tree
	nodes: list of objects of the class node
		list of nodes
	config: dict
		dictionary containing configuration parameters
	root: Node
		node a the root

	Methods
	-------
	__init__:
		Initialises a (labeled) tree with random topology.
		Parameters (other than self):
			number_nodes: attribute number_nodes
			config: attribute config
	
	generate_events:
		Samples random CN events at each node (in a single node they cannot be overlaping)
	generate_samples:
		Creates config['n_samples'] samples, assigns them uniformly at random to a node
		then samples the distribution of the coverage of each segment from the CN profile 
		of the assigned node.
	get_samples: 
		returns the list of samples
	get_log_prior:
		returns the log of the prior of the phylogeny, from the contribution:
		prior tree size 
		prior tree topology
		prior events
		prior sample assignation
	get_log_likelihood:
		returns the log of the likelihood of the segment coverage given the CN profile
	get_log_posterior:
		returns the unnormalised log posterior (i.e. up to an additive constant).
		It actually returns the log of the joint distribution
	__str__:
		returns a string representation of the tree
	"""
	def __init__(self,number_nodes,config):
		self.config = config
		self.number_nodes = number_nodes
		self.nodes = [Node(i,config) for i in range(number_nodes)]

		if number_nodes == 0:
			self.root = None
			return

		self.root = self.nodes[0] 
		if number_nodes == 1:
			return
		elif number_nodes == 2:
			self._add_directed_edge((0,1))
			return
		else:
			prufer = np.random.randint(0, high=number_nodes, size=number_nodes-2) 
			edges = list(decode_prufer(prufer))
			connected_nodes = set([self.root.id_])
			while(len(edges)>0):
				for i,edge in enumerate(edges):
					if edge[0] in connected_nodes:
						self._add_directed_edge(edge)
						connected_nodes.add(edge[1])
						del edges[i]
						break 

					elif edge[1] in connected_nodes:
						self._add_directed_edge((edge[1],edge[0]))
						connected_nodes.add(edge[0])
						del edges[i]
						break 

	def _add_directed_edge(self,edge):
		#Internal method used by __init__
		self.nodes[edge[0]].children.append(self.nodes[edge[1]])
		self.nodes[edge[1]].parent = self.nodes[edge[0]]

	def generate_events(self):
		self._DFS(self.root,'sample_events')

	def generate_samples(self):
		for i in range(self.config['n_samples']):
			node = self.nodes[np.random.randint(self.number_nodes)]
			sample = Sample(node,self.config)
			sample.generate_read_counts_from_CN()
			node.samples.append(sample)

	def get_samples(self):
		samples = []
		for node in self.nodes:
			samples += copy.deepcopy(node.samples)
		return samples

	def get_log_prior(self):
		tree_size_term = -np.log(2)*self.config['n_samples']*self.number_nodes # TODO: Do we really want this?
		tree_topology_term = - (self.number_nodes-1)*np.log(self.number_nodes)

		#prior events
		events_term = 0
		for node in self.nodes:
			# TODO: change this such that if part of the tree is modified, no need to redo all the calculations
			events_term -= node.get_log_prior_events(update = True)

		sample_assignation_term = - self.config['n_samples']*np.log(self.number_nodes)
		return tree_size_term + tree_topology_term + events_term + sample_assignation_term

	def get_log_likelihood(self):
		log_likelihood = 0
		for sample in self.get_samples():
			log_likelihood += sample.get_log_likelihood()
		return log_likelihood

	def get_log_posterior(self):
		#Unnormalised (i.e. up to an additive constant)
		log_prior = self.get_log_prior()
		log_likelihood = self.get_log_likelihood()
		return log_prior + log_likelihood

	def _DFS(self,node,method_name):
		method = getattr(node, method_name)
		method()
		for child in node.children:
			self._DFS(child,method_name)

	def _update_profiles(self):
		self._DFS(self.root,'update_profile')

	def __str__(self):
		def DFS_str(depth,node,string):
			string += depth*'  '+'-id: '+str(node.id_)+' CN: '+str(node.get_profile())+'\n'
			for sample in node.samples:
				string += (depth+1)*'  '+'sample: '+ str(sample.read_count)+'\n'
			for child in node.children:
				string = DFS_str(depth+1,child,string)
			return string

		self._update_profiles()
		return DFS_str(0,self.root,'')

def decode_prufer(p):
	"""
	Generative function that coverts iteratively a prufer sequence into a list of directed edges
	To get the whole list at once call list(decode_prufer(p))
	
	...
	
	Parameters
	---------
	p : list of ints
		prufer sequence
	"""
	p = list(p)
	vertices = set(range(len(p) + 2))
	for (i, u) in enumerate(p):
		v = min(vertices.difference(p[i:]))
		vertices.remove(v)
		yield u, v
	yield tuple(vertices)

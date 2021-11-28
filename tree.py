import numpy as np
from node import Node
from sample import Sample

#TODO: for now 1 diploid chromosome, need to change that

def decode_prufer(p):
    p = list(p)
    vertices = set(range(len(p) + 2))
    for (i, u) in enumerate(p):
        v = min(vertices.difference(p[i:]))
        vertices.remove(v)
        yield u, v
    yield tuple(vertices)

class Tree():
	def __init__(self,number_nodes,config):
		# TODO: case n<2
		#initialises tree at random 
		self.config = config
		self.number_nodes = number_nodes
		self.nodes = [Node(i,config) for i in range(number_nodes)]
		self.root = 0 #TODO: should be node and not name
		prufer = np.random.randint(0, high=number_nodes, size=number_nodes-2) 
		edges = list(decode_prufer(prufer))
		connected_nodes = set([self.root])
		while(len(edges)>0):
			for i,edge in enumerate(edges):
				if edge[0] in connected_nodes:
					self.add_directed_edge(edge)
					connected_nodes.add(edge[1])
					del edges[i]
					break 

				elif edge[1] in connected_nodes:
					self.add_directed_edge((edge[1],edge[0]))
					connected_nodes.add(edge[0])
					del edges[i]
					break 

	def add_directed_edge(self,edge):
		self.nodes[edge[0]].children.append(self.nodes[edge[1]])
		self.nodes[edge[1]].parent = self.nodes[edge[0]]

	def DFS(self,node,method_name):
		method = getattr(node, method_name)
		method()
		for child in node.children:
			self.DFS(child,method_name)

	def generate_events(self):
		self.DFS(self.nodes[self.root],'sample_events')

	def update_profiles(self):
		self.DFS(self.nodes[self.root],'update_profile')

	def generate_samples(self):
		self.samples = []
		for i in range(self.config['n_samples']):
			node = self.nodes[np.random.randint(self.number_nodes)]
			sample = Sample(node,self.config)
			sample.generate_sample_from_CN()
			node.samples.append(sample)
			self.samples.append(sample)

	def compute_prior_events(self,node):
		# TODO: change this such that if part of the tree is modified, no need to redo all the calculations
		log_likelihood = 0 
		for child in node.children:
			log_likelihood +=compute_prior_events(child)
		return log_likelihood

	def get_log_prior(self):
		#Unormalised
		log_prior = -np.log(2)*self.config['n_samples']*self.number_nodes #prior tree size
		log_prior -= (self.config['n_samples']-1)*np.log(self.config['n_samples']) #prior tree topology

		#prior events
		self.DFS(self.nodes[self.root],'compute_prior_events')
		for node in self.nodes:
			log_prior -= node.log_prior

		log_prior -= np.log(self.number_nodes)*self.config['n_samples']
		return log_prior

	def get_log_likelihood(self):
		log_likelihood = 0
		for sample in self.samples:
			log_likelihood += sample.get_log_likelihood()
		return log_likelihood

	def get_log_posterior(self):
		#Unnormalised (i.e. up to an additive constant)
		log_prior = self.get_log_prior()
		log_likelihood = self.get_log_likelihood()
		return log_prior + log_likelihood

	def __str__(self):
		def DFS_str(depth,node,string):
			string += depth*'  '+'- '+str(node.name)+' '+str(node.get_profile())+'\n'
			for sample in node.samples:
				string += (depth+1)*'  '+'sample: '+ str(sample.read_count)+'\n'
			for child in node.children:
				string = DFS_str(depth+1,child,string)
			return string

		self.update_profiles()
		return DFS_str(0,self.nodes[self.root],'')


import numpy as np
from scipy.stats import multinomial

class Sample():
	"""
	Class used to represent a minibulk sample and its assignation to a node. 

	...

	Attributes
	----------
	read_count: np.array
		list containing the reads counts for each segment
	p_read: np.array
		probability that a read spans the corresponding segment
	node: Node
		Node to which the sample is assigned
	config: dict
		dictionary containing configuration parameters

	Methods
	-------
	set_p_read:
		updates the p_read by using the CN porfile of the corresponding node
	generate_sample_from_CN:
		generates a random read_count from p_read and n_reads_sample in the config file
	get_log_likelihood:
		returns the log likelihood of read_count
	"""
	def __init__(self,id_,node,config,read_count = None):
		self.id_ = id_
		self.config = config
		self.node = node
		self.read_count = read_count
		self.p_read = self.set_p_read()
	
	def set_p_read(self):
		self.p_read = np.multiply(self.config['length_segments'],self.node.get_profile())
		self.p_read = self.p_read/np.sum(self.p_read)

	def generate_read_counts_from_CN(self,total_number_reads):
		#TODO case when everything deleted!
		self.set_p_read()
		self.read_count = np.random.multinomial(total_number_reads,self.p_read)

	def get_log_likelihood(self):
		#TODO: vectorise instead of for loop
		#TODO: check that it is working

		self.set_p_read()
		L = 0
		for i in range(len(self.p_read)):
			if self.p_read[i]==0:
				if self.read_count[i]!=0:
					return float('-inf')
			else:
				L+= self.read_count[i]*np.log(self.p_read[i])
		return L
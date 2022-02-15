import numpy as np
from scipy.stats import multinomial
import math

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
		self.log_likelihood = None
		self.update_log_likelihood()
	

	def generate_read_counts_from_CN(self,total_number_reads):
		#TODO case when everything deleted!
		self.node.update_p_read()
		self.read_count = np.random.multinomial(total_number_reads,self.node.p_read)

	def update_log_likelihood(self):
		#TODO: vectorise instead of for loop
		#TODO: change when mutlipeo chromosomes
		if self.read_count is None:
			return
		if np.all(self.node.get_profile() == 0):
			self.log_likelihood = float('-inf')
			return 
		
		self.log_likelihood = 0
		for i in range(len(self.node.p_read)):
			if (self.node.p_read[i]==0) and (self.read_count[i]!=0):
				self.log_likelihood = float('-inf')
				return
			else:
				self.log_likelihood+= self.read_count[i]*np.log(self.node.p_read[i])
				if (math.isnan(self.log_likelihood)):
					print(self.read_count[i], self.node.p_read[i])
					print("NaN value encountered in log likelihood")
					exit()
	
	def get_log_likelihood(self,update = False):
		if update:
			self.update_log_likelihood()
		return self.log_likelihood

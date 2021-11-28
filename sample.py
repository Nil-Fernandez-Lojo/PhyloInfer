import numpy as np
from scipy.stats import multinomial

class Sample():
	def __init__(self,node,config,read_count = None):
		self.config = config
		self.node = node
		self.read_count = read_count
		self.p_read = self.set_p_read()
	
	def set_p_read(self):
		self.p_read = np.multiply(self.config['length_segments'],self.node.get_profile())
		self.p_read = self.p_read/np.sum(self.p_read)

	def generate_sample_from_CN(self):
		self.set_p_read()
		self.read_count = np.random.multinomial(self.config['n_reads_sample'],self.p_read)

	def get_log_likelihood(self):
		self.set_p_read()
		return multinomial.logpmf(self.read_count, n=np.sum(self.read_count), p=self.p_read)

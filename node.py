import numpy as np

from util import events_to_vector, update_reagions_available_event
from event import Event, sample_events

def get_regions_available_profile(profile):
	n_seg = len(profile)
	regions_available = []
	segment_present = profile != 0
	for i in range(len(segment_present)):
		if i == 0:
			prev = 0
		else:
			prev = segment_present[i-1]

		if prev != segment_present[i]:
			if prev == 0:
				regions_available.append({'start':i,'len':0})
			else:
				regions_available[-1]['len'] = i - regions_available[-1]['start']
	if segment_present[-1] == 1:
		regions_available[-1]['len'] = n_seg - regions_available[-1]['start']
	return regions_available

class Node():
	def __init__(self,name,config,parent = None):
		self.name = name
		self.parent = parent
		self.children = []
		self.events = []
		self.samples = []
		self.config = config
		self.profile = 2*np.ones(config['number_segments'])
		self.log_prior = 0

	def add_child(self,node):
		self.children.append(node)
		node.parent = self

	def sample_events(self):
		self.update_profile()
		regions_available = get_regions_available_profile(self.profile)
		self.events = sample_events(regions_available,self.config['p_new_event'])
		self.update_profile()

	def update_profile(self):
		#get profile_parents 
		if self.parent is None:
			self.profile = 2*np.ones(self.config['number_segments'])
		else:
			self.profile = self.parent.get_profile()

		#add events to profile
		change_profile = events_to_vector(self.events,self.config['number_segments'])

		assert (np.count_nonzero(self.profile)!= len(self.profile)) or (np.all(change_profile[self.profile == 0] == 0)),"if segment lost no more events can happen"
		self.profile += change_profile

	def compute_prior_events(self):
		if self.parent is None:
			regions_available = [{'start':0,'len':self.config['number_segments']}]
		else:
			regions_available = get_regions_available_profile(self.parent.get_profile())

		self.log_prior = 0
		for event in self.events:
			K = np.array([r['len'] for r in regions_available])
			self.log_prior -= np.log(np.sum(np.multiply(K,K+1)))
			regions_available = update_reagions_available_event(regions_available,event)
		self.log_prior += np.log(np.math.factorial(len(self.events))) #TODO investigate how good this is

	def get_profile(self):
		return np.copy(self.profile)

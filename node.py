import numpy as np

from util import events_to_vector, update_reagions_available_event
from event import Event, sample_events

def get_regions_available_profile(profile):
	"""
	Given a copy number profile, returns the list of sorted regions where events can occur.
	I.e. list of adjancent segments that do not have a copy number of 0
	e.g. [0,1,2,2,1,1,0,0,0,1,1] -> [{'start':1,'len':5}, {'start':9,'len':2}]

	Parameters
	----------
	profile: np.array
		1 dimensional numpy array of signed integers
	"""

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
	"""
	Class used to represent a node in the tree. 

	...

	Attributes
	----------
	id_: int
		unique id of the node
	parent: Node
		pointer to parent Node, if root set to None
	children: list of objects of the class Node
		list of pointers to children
	events: List of objects of the class Event
		list of NON OVERLAPING CN events that are new to this node
	samples: list of objects of the class Sample
		list of samples assigned to this node
	config: dict
		dictionary containing configuration parameters
	profile: np.array
		1D numpy array corresponding to the CN of samples assigned to this node.
		Computed from this node events and its ancestries node events
	log_prior: float
		log of the prior probability of the events of this node

	Methods
	-------
	add_child: 
		adds a child to the node
		Parameters:
			node: bject of the type Node
	sample_events: 
		Samples CN events at this node and updates its CN profile
		ATTENTION, the ordering for calling this method is important
		This method must have been called at the parent's node before calling it for this node
	update_profile: 
		updates the CN profile of the node, by getting a copy of the parent node CN
		that is assumed to be correct (hence imortance of the ordering of calling methods)
		and adding the modifications encoded in events
	compute_prior_events:
		updates the attribute log_prior by computing the log of the prior probability 
		of the events. ATTENTION is then multiplied by n! (where n is the number of events)
		This is because the order does not matter however the distribution of the events is 
		not the same if you interchange their order... Longer events are more likely in the first 
		events, so this correction is not mathematically correct, need to investigate in how to 
		solve this
	get_profile:
		returns a copy of profile
	"""

	def __init__(self,id_,config,parent = None):
		self.id_ = id_
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
		# ATTENTION, the ordering for calling this method is important
		# This method must have been called at the parent's node before calling it for this node
		# Otherwise the regions available for events may be wrong
		# i.e. a segment is lost in the parents node after sampling a loss or a gain of that segment at this node 
		self.events = []
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

	def get_log_prior_events(self,update = False):
		# TODO: change how much the prior changes if order events is swapped (i.e. factorial term well justified)
		if update:
			if self.parent is None:
				regions_available = [{'start':0,'len':self.config['number_segments']}]
			else:
				regions_available = get_regions_available_profile(self.parent.get_profile())

			self.log_prior = 0
			for event in self.events:
				K = np.array([r['len'] for r in regions_available])
				self.log_prior -= np.log(np.sum(np.multiply(K,K+1)))
				regions_available = update_reagions_available_event(regions_available,event)

			#TODO investigate how good this approximation is
			self.log_prior += np.log(np.math.factorial(len(self.events)))
		return self.log_prior

	def get_profile(self):
		return np.copy(self.profile)

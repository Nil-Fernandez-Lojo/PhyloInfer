import numpy as np
import copy
from tree.util import events_to_vector,N_events_combinations
from tree.event import Event, sample_events

def change_profile_to_events(change_profile,regions_available):
	events = []
	previous_change = 0
	segments = []

	for seg in regions_available:
		change = change_profile[seg]
		#if outside event
		if change == 0:
			# if end of event
			if previous_change != 0:
				events.append(Event(copy.copy(segments),gain))
				segments = []
		
		#if inside event	
		else:
			#if continue same event
			if change == previous_change:
				segments.append(seg)

			# otherwise there is a new event
			else:
				if previous_change != 0:
					events.append(Event(copy.copy(segments),gain))
					segments = []

				segments.append(seg)
				gain = (change + 1)/2
		previous_change = change
	
	if previous_change != 0:
		events.append(Event(copy.copy(segments),gain))	

	return events

def get_regions_available_profile(profile):
	"""
	TODO should just store it as a binary vector
	Given a copy number profile, returns the list of sorted regions where events can occur.
	I.e. list of adjancent segments that do not have a copy number of 0
	e.g. [0,1,2,2,1,1,0,0,0,1,1] -> [{'start':1,'len':5}, {'start':9,'len':2}]

	Parameters
	----------
	profile: np.array
		1 dimensional numpy array of signed integers
	"""
	return np.where(profile>0)[0]

	# n_seg = len(profile)
	# regions_available = []
	# segment_present = profile != 0

	# for i in range(len(segment_present)):
	# 	if i == 0:
	# 		prev = 0
	# 	else:
	# 		prev = segment_present[i-1]

	# 	if prev != segment_present[i]:
	# 		if prev == 0:
	# 			regions_available.append({'start':i,'len':0})
	# 		else:
	# 			regions_available[-1]['len'] = i - regions_available[-1]['start']
	# if segment_present[-1] == 1:
	# 	regions_available[-1]['len'] = n_seg - regions_available[-1]['start']
	# return regions_available

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
		of the events.
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
		self.profile += change_profile

	def update_events(self):
		#This is needed for when a move 
		if len(self.events) <2:
			return

		if self.parent is None:
			regions_available = np.arange(self.config['number_segments'])
		else:
			regions_available = get_regions_available_profile(self.parent.get_profile())
		
		#TODO this is not clean
		#check if now impossible events after change of topology tree
		#Then do not update events since this is not done correctly in that case...
		#In this case we do not update the events vector
		for event in self.events:
			for segment in event.segments:
				if segment not in regions_available:
					return

		change_profile = events_to_vector(self.events,self.config['number_segments'])
		self.events = change_profile_to_events(change_profile,regions_available)

	def get_log_prior_events(self,update = True):
		if update:
			if self.parent is None:
				regions_available = np.arange(self.config['number_segments'])
			else:
				regions_available = get_regions_available_profile(self.parent.get_profile())

			profile = self.get_profile()
			if np.any(profile <0):
				return float("-inf")

			#If all DNA material removed
			#TODO: need to change this for when multiuple chr
			if np.all(profile == 0):
				return float("-inf")

			#check if event spans regions deleted, if so prior equal zero
			for event in self.events:
				if not set(event.segments).issubset(regions_available):
					return float("-inf")

			if self.parent is None:
				K = self.config['number_segments']
			else:
				K = np.count_nonzero(self.parent.get_profile())

			list_p_events = self.config['p_new_event']**np.arange(K+1)
			list_p_events = list_p_events/np.sum(list_p_events)
			
			n_events = len(self.events)
			# if self.parent is not None:
			# 	print("parent profile",self.parent.get_profile())
			# for event in self.events:
			# 	print(event)
			# print(n_events,K)
			p_n_events = list_p_events[n_events]

			self.log_prior = np.log(p_n_events)# due to number of events
			if n_events>0:
				self.log_prior -= np.log(N_events_combinations(n_events,K))# due to possible combinations of events 
		return self.log_prior

	def get_profile(self):
		return np.copy(self.profile)

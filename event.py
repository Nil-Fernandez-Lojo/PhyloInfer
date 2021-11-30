import numpy as np
from util import update_reagions_available_event

def sample_events(regions_available,p_new_event):
	"""
	Samples iteratively objects of the class Event that are NON OVERLAPING.
	At each iteration, sample a new event if these 2 conditions are met:
		- there is a region available for an event to occur
		- generates a bernouilli sample = 1 with probability p_new_event
	The event is then sampled uniformly at random from all possible events from all the regions availables

	Otherwise breaks and returns the current list of events.
	
	...

	Parameters 
    ----------
    regions_available: list of dictionaries 
    	List of different regions availables for an event to span them
    	the regions available are represented by a dictionary with the keys:
    		- start: signed int
    			index of the first segment of the region
    		- length: int (>0)
    			number of segments in the region
    p_new_event: float (in [0,1])
    	probability of adding a new event at each iteration

    Output
    ------
  	List of sampled events of the class Event
	"""

	events = []
	while (np.random.binomial(1, p_new_event) and len(regions_available)>0):
		if len(regions_available) == 1:
			region_idx = 0
		else:
			p_region = np.array([r['len']*(r['len']+1) for r in regions_available])
			p_region =p_region/np.sum(p_region)
			region_idx = np.argmax(np.random.multinomial(1,p_region))

		# TODO: include the option of adding a weight to each length
		region = regions_available[region_idx]
		p_start = np.arange(region['len'],0,-1)/(region['len']*(region['len']+1)/2)
		rel_start = np.argmax(np.random.multinomial(1,p_start)) 
		length = np.random.randint(1,region['len']-rel_start+1)
		start = region['start'] + rel_start
		gain = np.random.randint(2)
		event = Event(start,length,gain)
		regions_available = update_reagions_available_event(regions_available,event,region_idx)
		events.append(event)
	return events

class Event():
	"""
	Class used to represent single copy number gain or loss. 
 	
 	...

    Attributes
    ----------
    start: signed int
    	index of the first segment gained or lost
    length: int (>0)
    	number of adjacent segments gained or lost
    gain: int
    	represents if the event is a gain (gain = 1) or a loss (gain = 0)
	"""
	def __init__(self,start,length,gain):
		assert(gain in [0,1]), 'gain must be either 0 (loss) or 1 (gain)'
		assert(start >= 0), 'start index cannot be negative'
		assert(length >= 1), 'length of an event must be at least 1 segment'
		self.start = start
		self.length = length
		self.gain = gain

	def __str__(self):
		return str(self.start)+" "+str(self.length)+" "+str(self.gain)

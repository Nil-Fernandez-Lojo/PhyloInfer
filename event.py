import numpy as np
from util import update_reagions_available_event

def sample_events(regions_available,p_new_event):
	#probability of adding new event
	events = []
	while (np.random.binomial(1, p_new_event) and len(regions_available)>0):
		if len(regions_available) == 1:
			region_idx = 0
		else:
			p_region = np.array([r['len']*(r['len']+1) for r in regions_available])
			p_region =p_region/np.sum(p_region)
			region_idx = np.argmax(np.random.multinomial(1,p_region)) #TODO: not clean
		region = regions_available[region_idx]
		p_start = region['len'] - np.array(list(range(region['len'])))
		p_start = p_start/np.sum(p_start)
		rel_start = np.argmax(np.random.multinomial(1,p_start)) #TODO: not clean
		length = np.random.randint(1,region['len']-rel_start+1)
		start = region['start'] + rel_start
		gain = np.random.randint(2)
		event = Event(start,length,gain)
		regions_available = update_reagions_available_event(regions_available,region_idx,event)
		events.append(event)
	return events

class Event():
	def __init__(self,start,length,gain):
		self.start = start
		self.length = length
		self.gain = gain

	def __str__(self):
		return str(self.start)+" "+str(self.length)+" "+str(self.gain)

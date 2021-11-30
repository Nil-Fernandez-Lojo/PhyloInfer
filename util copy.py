import numpy as np

# File containing utility functions used mutliple times across different methods and functions

def events_to_vector(events,n_seg):
	"""
	returns the copy number profile change corresping to a list of events

	...

	Parameters
	----------
	events: list of objects of the class event
	n_seg: int, number of segments in the chromosome
	"""
	vector = np.zeros(n_seg)
	for event in events:
		vector[event.start:event.start+event.length] = -1 + 2*event.gain
	return vector

def update_reagions_available_event(regions_available,event,region_event_idx = None):
	"""
	Returns the regions available for new events after adding one event. 
	(Since events cannot be overlapping in the same node)

	Parameters
	----------
	regions_available: list of dictionaries where each entry has the keys
		check docstring of sample_events for more details
	event: Event
	region_event_idx: signed int, optional
		index indicates in which region the event is located.
		if not provided, it is computed	
	"""

	# TODO: change this into binary search, regions are sorted (but should we put an assertion?)
	if region_event_idx is None:
		for i,r in enumerate(regions_available):
			if event.start >= r['start'] and event.start < r['start'] + r['len']:
				region_event_idx = i
				break
		assert (region_event_idx is not None)
	
	region = regions_available[region_event_idx]
	rel_start = event.start - region['start']
	new_regions = []
	if rel_start>1:
		new_regions.append({'start':region['start'], 'len':rel_start-1})
	if rel_start + event.length < region['len']-1:
		new_regions.append({'start':event.start+event.length+1, 'len':region['len']-event.length-rel_start-1})

	return regions_available[:region_event_idx] + new_regions + regions_available[region_event_idx+1:]
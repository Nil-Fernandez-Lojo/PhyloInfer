import numpy as np

def events_to_vector(events,n_seg):
	vector = np.zeros(n_seg)
	for event in events:
		vector[event.start:event.start+event.length] = -1 + 2*event.gain
	return vector

def update_reagions_available_event(regions_available,region_event_idx,event):
	region = regions_available[region_event_idx]
	rel_start = event.start - region['start']
	new_regions = []
	if rel_start>1:
		new_regions.append({'start':region['start'], 'len':rel_start-1})
	if rel_start + event.length < region['len']-1:
		new_regions.append({'start':event.start+event.length+1, 'len':region['len']-event.length-rel_start-1})
	return regions_available[:region_event_idx] + new_regions + regions_available[region_event_idx+1:]

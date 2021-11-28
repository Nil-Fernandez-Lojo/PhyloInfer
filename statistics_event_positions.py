import numpy as np
import matplotlib.pyplot as plt


from util import  events_to_vector
from event import sample_events

def get_probability_segment_covered(n_seg,p_new_event,n_sim=1000):
	results = np.zeros((n_sim,n_seg))
	for i in range(n_sim):
		events = sample_events([{'start':0,'len':n_seg}],p_new_event)
		results[i,:] = events_to_vector(events,n_seg)
	return np.sum(np.absolute(results),axis = 0)/n_sim

def get_probability_breakpoint(n_seg,p_new_event,n_sim=1000):
	results = np.zeros((n_sim,n_seg+1))
	for i in range(n_sim):
		events = sample_events([{'start':0,'len':n_seg}],p_new_event)
		for e in events:
			results[i,e.start] = 1
			results[i,e.start+e.length] = 1
	return np.sum(results,axis = 0)/n_sim

def hist_len_events(n_seg,p_new_event,n_sim=1000):
	p = np.zeros(n_seg+1)
	for i in range(n_sim):
		events = sample_events([{'start':0,'len':n_seg}],p_new_event)
		for e in events:
			p[e.length] +=1
	p = p/np.sum(p)
	print('mean length:', np.sum(np.multiply(p,np.arange(n_seg+1))))
	return p

p_new_event = 0.3
number_segments = 100

n_sim = 1000000
p = get_probability_breakpoint(number_segments,p_new_event,n_sim)
#h = hist_len_events(number_segments,p_new_event,n_sim)
plt.plot(p)
plt.show()

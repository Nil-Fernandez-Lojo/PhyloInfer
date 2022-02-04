import numpy as np
from tree.util import update_reagions_available_event,N_bkp,N_g,C_db

def get_sol_comb_problem(n,d):
	"""
	Get list of all the solution of the following combinatorial problem:
	{x:x in N_0^d, sum(x)=n}

	#TODO: this is not optimal, should cache the solutions
	"""
	if (d == 0) or (d>n):
		return []
	elif d == 1:
		return [[n]]
	else:
		sol = []
		for i in range(1,n-d+2):
			partial_sol = get_sol_comb_problem(n-i,d-1)
			for x in partial_sol:
				x.append(i)
			sol.extend(partial_sol)
		return sol

def sample_events(regions_available,p_new_event):
	"""
	TODO change description
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
	if len(regions_available) == 0:
		return []

	K = 0
	possible_breakpoints_start = []
	possible_breakpoints_end = []
	for region in regions_available:
		K+= region['len']
		possible_breakpoints_start.extend(list(range(region['start']-1,region['start']+region['len']-1)))
	possible_breakpoints_end = [b+1 for b in possible_breakpoints_start]

	p_events = p_new_event**np.arange(K+1)
	p_events = p_events/np.sum(p_events)
	n_events = np.random.choice(K+1,p=p_events)
	#print('n_events', n_events)
	if n_events == 0:
		return []
	N_a = np.zeros(n_events) #number of sets of events given the number of breakpoints
	for a in range(n_events):
		if a < 2*n_events - (K+1):
			N_a[a] = 0
		else:
			N_a[a] = N_bkp(n_events,K,a)*N_g(n_events,a)*C_db(n_events,a)
	p_a = N_a/np.sum(N_a)
	a = np.random.choice(n_events,p=p_a)
	b = 2*n_events - a #number of breakpoints
	d = n_events-a #number of super events
	#print('a:', a, ', b:', b, ' d:',d)

	list_sol = get_sol_comb_problem(n_events,d)
	sol = list_sol[np.random.randint(len(list_sol))]
	#print('sol:', sol)

	b_list_idx = np.random.choice(K+1, size=b, replace=False)
	b_list_idx = np.sort(b_list_idx)
	#print('b_list_idx:',b_list_idx)
	
	b_i = 0
	list_events = []
	for super_event_len in sol:
		gain = np.random.randint(2)
		for i in range(super_event_len):
			start = b_list_idx[b_i]
			end = b_list_idx[b_i+1]
			segments = []
			i = 0
			for region in regions_available:
				for s in range(region['start'], region['start'] + region['len']):
					if i>= start and i<end:
						segments.append(s)
					i+=1
			list_events.append(Event(segments,gain))
			b_i +=1
			gain = 0 if gain == 1 else 1
		b_i +=1
	return list_events

	# events = []
	# while (np.random.binomial(1, p_new_event) and len(regions_available)>0):
	# 	if len(regions_available) == 1:
	# 		region_idx = 0
	# 	else:
	# 		p_region = np.array([r['len']*(r['len']+1) for r in regions_available])
	# 		p_region =p_region/np.sum(p_region)
	# 		region_idx = np.argmax(np.random.multinomial(1,p_region))

	# 	# TODO: include the option of adding a weight to each length
	# 	region = regions_available[region_idx]
	# 	p_start = np.arange(region['len'],0,-1)/(region['len']*(region['len']+1)/2)
	# 	rel_start = np.argmax(np.random.multinomial(1,p_start)) 
	# 	length = np.random.randint(1,region['len']-rel_start+1)
	# 	start = region['start'] + rel_start
	# 	gain = np.random.randint(2)
	# 	event = Event(start,length,gain)
	# 	regions_available = update_reagions_available_event(regions_available,event,region_idx)
	# 	events.append(event)
	# return events

class Event():
	"""
	Class used to represent single copy number gain or loss. 
 	
 	...

    Attributes
    ----------
    gain: int
    	represents if the event is a gain (gain = 1) or a loss (gain = 0)
	"""
	def __init__(self,segments,gain):
		assert(gain in [0,1]), 'gain must be either 0 (loss) or 1 (gain)'
		self.segments = segments
		self.gain = gain

	def __str__(self):
		return str(self.segments)+" "+str(self.gain)

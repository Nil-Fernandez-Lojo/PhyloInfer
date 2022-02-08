import numpy as np
from tree.util import N_bkp,N_g,C_db

# def get_sol_comb_problem(n,d):
# 	"""
# 	Get list of all the solution of the following combinatorial problem:
# 	{x:x in N_0^d, sum(x)=n}

# 	#TODO: this is not optimal, should cache the solutions
# 	"""
# 	if (d == 0) or (d>n):
# 		return []
# 	elif d == 1:
# 		return [[n]]
# 	else:
# 		sol = []
# 		for i in range(1,n-d+2):
# 			partial_sol = get_sol_comb_problem(n-i,d-1)
# 			for x in partial_sol:
# 				x.append(i)
# 			sol.extend(partial_sol)
# 		return sol

def sample_events(regions_available,p_new_event=0,n_events = None):
	"""
	TODO change description
	...

	Parameters 
    ----------
    regions_available: list of int:
		list of segments that have not been removed
    p_new_event: float (in [0,1])
    	probability of adding a new event at each iteration

    Output
    ------
  	List of sampled events of the class Event
	"""

	K =len(regions_available)
	if K == 0:
		return []

	if n_events is None:
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

	fixed_events = np.random.choice(np.arange(1,n_events),size=a, replace=False)

	b_list_idx = np.random.choice(K+1, size=b, replace=False)
	b_list_idx = np.sort(b_list_idx)
	#print('b_list_idx:',b_list_idx)
	# print("n_events",n_events)
	# print("a",a)
	# print("fixed_events",fixed_events)
	# print("b_list_idx",b_list_idx)
	b_i = 0
	list_events = []
	for i in range(n_events):
		if i == 0:
			gain = np.random.randint(2)
		else:
			if i in fixed_events:
				gain = 0 if list_events[-1].gain == 1 else 1
				b_i+=1
			else:
				gain = np.random.randint(2)
				b_i+=2
		segments = regions_available[b_list_idx[b_i]:b_list_idx[b_i+1]]		
		list_events.append(Event(segments,gain))

	return list_events

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

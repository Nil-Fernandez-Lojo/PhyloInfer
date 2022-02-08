from tree.tree import Tree
from tree.event import sample_events
from tree.node import get_regions_available_profile

import numpy as np
import copy

################ Utility functions ################

def get_regions_new_event(regions_available,events):
	#TODO: improve implementation e.g. last for loop
	if len(events) == 0:
		return [{"segments": regions_available, 
			"previous_dir": None, 
			"next_dir":None}]

	first_event = events[0]
	x = regions_available[regions_available<first_event.segments[0]]
	#event covers first segment available
	if len(x) == 0:
		potential_region_events = []
	else:
		potential_region_events = [{"segments": x, "previous_dir": None, "next_dir":first_event.gain}]
	for i in range(len(events)):
		next_seg = regions_available[regions_available>events[i].segments[-1]]
		previous_dir = events[i].gain
		if i == len(events)-1:
			x = next_seg
			next_dir = None
		else:
			prev_seg = regions_available[regions_available<events[i+1].segments[0]]
			x = list(set(prev_seg) & set(next_seg))
			#if adjacent events (double breakpoint), no region for new event between those 2 events
			if len(x) > 0:
				x = np.sort(x)
			next_dir = events[i+1].gain
		#if adjacent events (double breakpoint), no region for new event between those 2 events
		if len(x) > 0:
			potential_region_events.append({"segments": x, 
				"previous_dir": previous_dir, 
				"next_dir":next_dir})

	#discard ones of length 1 or 2 if connected events of opposite gains
	potential_region_events_filtered = []
	for region in potential_region_events:
		if ((len(region['segments']) > 2) or 
			(region['previous_dir'] is None) or 
			(region['next_dir'] is None) or 
			(region['previous_dir'] == region['next_dir'])):
				potential_region_events_filtered.append(region)

	return potential_region_events_filtered

def add_event_correct_place(events,new_event):
	#we do this to preserve ordering
	for i,old_event in enumerate(events):
		if new_event.segments[-1]<old_event.segments[0]:		
			events.insert(i,new_event)
			return

	if new_event.segments[0] > events[-1].segments[-1]:
		events.insert(len(events),new_event)

def p_add_event(tree,node,event):
	n_nodes = len(tree.nodes)
	if node.parent is None:
		regions_available = np.arange(tree.config['number_segments'])
	else:
		regions_available = get_regions_available_profile(node.parent.get_profile())

	K = len(regions_available)
	print("node.events", [str(ev) for ev in node.events])
	if len(node.events) == 0:
		print("K", K, "n_nodes", n_nodes)
		print(-np.log(K*(K+1)*n_nodes))
		exit()
		return -np.log(K*(K+1)*n_nodes)
	else:
		potential_region_events = get_regions_new_event(regions_available,node.events)
		print("potential_region_events",potential_region_events)
		#print("region",region)
	exit()

################ general move function ################

def move(tree,move_type):
	tree_modified = copy.deepcopy(tree)
	if move_type == 'prune_and_reatach':
		info = prune_and_reatach(tree_modified)
	elif move_type == 'swap_events_2_nodes':
		info = swap_events_2_nodes(tree_modified)
	elif move_type == 'add_event':
		info = add_event(tree_modified)
	elif move_type == 'remove_event':
		info = remove_event(tree_modified)
	elif move_type == 'modify_event':
		info = modify_event(tree_modified)
	elif move_type == 'modify_sample_attachments':
		info = modify_sample_attachments(tree_modified)
	tree_modified._update_profiles()
	return tree_modified,info

################ different moves ################

def prune_and_reatach(tree):
	#TODO
	pass

def swap_events_2_nodes(tree):
	nodes_idx = np.random.choice(len(tree.nodes), size=2, replace=False)
	node_0 = tree.nodes[nodes_idx[0]]
	node_1 = tree.nodes[nodes_idx[1]]
	additional_info = {"node_0.id_":node_0.id_, "node_1.id_":node_1.id_}
	#print("we swap the events of:",node_0.id_, node_1.id_)

	events_0 = node_0.events
	node_0.events = node_1.events
	node_1.events = events_0
	return additional_info

def add_event(tree):
	#TODO: should improve this move (move by itself and maybe just implementation)
	node = tree.nodes[np.random.choice(len(tree.nodes))]
	additional_info = {"node.id_":node.id_,'sucess':False}
	#Root node
	if node.parent is None:
		regions_available = np.arange(tree.config['number_segments'])
	else:
		regions_available = get_regions_available_profile(node.parent.get_profile())
	print("node id",node.id_)
	print("regions_available",regions_available)
	print("events:")
	for ev in node.events:
		print(ev)
	print()

	if len(node.events) == 0:
		if len(regions_available) == 0:
			additional_info['reason'] = "No regions available (and 0 events)"
			return additional_info
		node.events = sample_events(regions_available,n_events = 1)
		print('new event', str(node.events[0]))
		additional_info['event'] = node.events[0]
		additional_info["sucess"] = True
		return additional_info
	else:
		potential_region_events = get_regions_new_event(regions_available,node.events)
		print('potential_region_events')
		print(potential_region_events)
		
		if len(potential_region_events) == 0:
			additional_info['reason'] = "No regions available"
			return additional_info

		region = potential_region_events[np.random.randint(len(potential_region_events))]
		print('region selected', region)
		
		if (region['previous_dir'] is None) and (region['next_dir'] is None):
			gain = np.random.randint(2)
		elif region['previous_dir'] is None:
			if len(region['segments']) == 1:
				gain = 1 if region['next_dir'] == 0 else 1
			else:
				gain = np.random.randint(2)
				if gain == region['next_dir']:
					region['segments'] = region['segments'][:-1]
		elif region['next_dir'] is None:
			if len(region['segments']) == 1:
				gain = 1 if region['previous_dir'] == 0 else 1
			else:
				gain = np.random.randint(2)
				if gain == region['previous_dir']:
					region['segments'] = region['segments'][1:]
		else:
			if (len(region['segments']) <= 2):
				gain = 1 if region['previous_dir'] == 0 else 1
			else:
				gain = np.random.randint(2)
				if gain == region['previous_dir']:
					region['segments'] = region['segments'][1:]
				else:
					region['segments'] = region['segments'][:-1]
		
		new_event = sample_events(region['segments'],n_events = 1)[0]
		new_event.gain = gain
		print('new event', new_event)	
		add_event_correct_place(node.events,new_event)
		additional_info['sucess'] = True
		additional_info['event'] = new_event
		return additional_info

def remove_event(tree):
	n_events_node = [len(node.events) for node in tree.nodes]
	tot_events = np.sum(n_events_node)
	if tot_events == 0:
		return {"sucess": False,"reason":"There are no events in the tree -> we can't remove any"}
	p = np.array(n_events_node)/tot_events
	node = tree.nodes[np.random.choice(len(tree.nodes),p=p)]
	event = node.events.pop(np.random.randint(len(node.events)))
	print('event removed from node',node.id_)
	info = {"node.id_": node.id_,"event":event}
	return info

def modify_event(tree):
	#TODO: finish, not needed in theory
	n_events_node = [len(node.events) for node in tree.nodes]
	p = np.array(n_events_node)/np.sum(n_events_node)
	node = tree.nodes[np.random.choice(len(tree.nodes),p=p)]
	event = node.events[np.random.randint(len(node.events))]
	#choose if extension or reduction
	pass

def modify_sample_attachments(tree):
	sample_idx = np.random.randint(len(tree.samples))
	sample = tree.samples[sample_idx]
	current_node = sample.node
	
	#TODO: I should find a more elegant way to do this 
	other_nodes = []
	for node in tree.nodes:
		if node.id_ != current_node.id_:
			other_nodes.append(node)
	assert(len(tree.nodes) == len(other_nodes)+1)
	new_node = other_nodes[np.random.randint(len(other_nodes))]
	
	current_node.samples.remove(sample)
	sample.node = new_node
	new_node.samples.append(sample)
	return None

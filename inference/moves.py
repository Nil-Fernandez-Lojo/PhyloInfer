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
		if ((len(region['segments']) > 1) or 
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
	if len(node.events) == 0:
		#print("K", K, "n_nodes", n_nodes)
		return -np.log(K*(K+1)*n_nodes)
	else:
		potential_region_events = get_regions_new_event(regions_available,node.events)
		# if node.parent is not None:
		# 	print("parent profile",node.parent.get_profile())
		# print("potential_region_events",potential_region_events)
		# print()
		for region in potential_region_events:
			if set(event.segments).issubset(set(region["segments"])):
				region_event = region
				break

		gain_fixed = False 
		len_region = len(region_event['segments'])
		if (region_event['previous_dir'] is None) and (region_event['next_dir'] is not None):
			if len_region == 1:
				gain_fixed = True
			elif event.gain == region_event['next_dir']:
				len_region -= 1
		elif (region_event['next_dir'] is None) and (region_event['previous_dir'] is not None):
			if len_region == 1:
				gain_fixed = True
			elif event.gain == region_event['previous_dir']:
				len_region -= 1
		else:
			if (len_region == 1) or ((len_region == 2) and (region_event['previous_dir'] == region_event['next_dir'])):
				gain_fixed = True
			
			if event.gain == region_event['previous_dir']:
				len_region -= 1
			if event.gain == region_event['next_dir']:
				len_region -= 1

		gain_factor = 0 if gain_fixed else -np.log(2)
		K = len(region_event['segments'])
		#print("region_event",region_event)
		return -np.log(n_nodes)-np.log(len(potential_region_events))+gain_factor-np.log(K*(K+1))

def get_root_nodes_affected_by_move(tree,info,include_sample = False):
	if info["move_type"] in ['add_event', 'remove_event', 'modify_event']:
		node_ids = [info["node.id_"]]
	elif info["move_type"] == 'prune_and_reattach':
		node_ids = [info["root_subtree.id_"]]
	elif info["move_type"] == 'swap_events_2_nodes':
		node_ids = [info["node_0.id_"], info["node_1.id_"]]
	elif info["move_type"] == 'modify_sample_attachments':
		if include_sample:
			node_ids = ["old_node.id_","new_node.id_"]
		else:
			node_ids = []

	root_nodes_to_apply_updates = []
	for i in range(len(tree.nodes)):
		if tree.nodes[i].id_ in node_ids:
			root_nodes_to_apply_updates.append(tree.nodes[i])
			if len(root_nodes_to_apply_updates) == len(node_ids):
				break
	return root_nodes_to_apply_updates

def update_tree_after_move(tree,info):
	root_nodes_to_apply_updates = get_root_nodes_affected_by_move(tree,info,include_sample = False)
	for node in root_nodes_to_apply_updates:
		tree._update_profiles(node)
		tree.update_events(node)	

################ general move function ################
def move(tree,move_type,
	node_id = None,
	node_1_id = None,
	event_to_add = None,
	event_idx_to_remove = None,
	sample_idx = None):
	tree_modified = copy.deepcopy(tree)
	if move_type == 'prune_and_reattach':
		info = prune_and_reattach(tree_modified)
	elif move_type == 'swap_events_2_nodes':
		info = swap_events_2_nodes(tree_modified,node_0_id = node_id,node_1_id = node_1_id)
	elif move_type == 'add_event':
		info = add_event(tree_modified,node_id = node_id,new_event = event_to_add)
	elif move_type == 'remove_event':
		info = remove_event(tree_modified,node_id = node_id, event_idx = event_idx_to_remove)
	elif move_type == 'modify_event':
		info = modify_event(tree_modified)
	elif move_type == 'modify_sample_attachments':
		info = modify_sample_attachments(tree_modified,sample_idx = sample_idx,new_node_id = node_id)
	info['move_type'] = move_type
	update_tree_after_move(tree_modified,info)
	
	return tree_modified,info

################ different moves ################
def prune_and_reattach(tree, root_subtree_idx = None, new_parent_subtree_id = None):
	#TODO: we could end up with the same tree if we prune and reattach to the same point, I have to change that
	#TODO: add preference for smaller subtrees like in SCICONE

	if root_subtree_idx is None:
		#TODO: change this not clean
		idx = np.random.choice(len(tree.nodes),replace = False, size = 2)
		if tree.nodes[idx[0]].parent is not None:
			root_subtree = tree.nodes[idx[0]]
		else:
			root_subtree = tree.nodes[idx[1]]
	else:
		root_subtree = tree.nodes[root_subtree_idx]

	if new_parent_subtree_id is None:
		subtree_nodes_id = tree.get_children_id(root_subtree)
		subtree_nodes_id.append(root_subtree.id_)
		remaining_nodes = []
		for node in tree.nodes:
			if node.id_ not in subtree_nodes_id:
				remaining_nodes.append(node)
		new_parent_subtree = remaining_nodes[np.random.choice(len(remaining_nodes))]
	else:
		new_parent_subtree = tree.nodes[new_parent_subtree_id]

	for n in root_subtree.parent.children:
		if n == root_subtree:
			root_subtree.parent.children.remove(n)
			break

	root_subtree.parent = new_parent_subtree
	new_parent_subtree.children.append(root_subtree)
	additional_info = {"root_subtree.id_":root_subtree.id_, "new_parent_subtree.id_":new_parent_subtree.id_,'success':True}
	return additional_info

def swap_events_2_nodes(tree,node_0_id = None,node_1_id = None):
	if (node_0_id is None) and (node_1_id is None):
		nodes_idx = np.random.choice(len(tree.nodes), size=2, replace=False)
		node_0 = tree.nodes[nodes_idx[0]]
		node_1 = tree.nodes[nodes_idx[1]]
	else:
		for node in tree.nodes:
			if node.id_ == node_0_id:
				node_0 = node
			elif node.id_ == node_1_id:
				node_1 = node

	additional_info = {"node_0.id_":node_0.id_, "node_1.id_":node_1.id_}
	#print("we swap the events of:",node_0.id_, node_1.id_)

	events_0 = node_0.events
	node_0.events = node_1.events
	node_1.events = events_0
	return additional_info

def add_event(tree,node_id = None,new_event = None):
	#TODO: should improve this move (move by itself and maybe just implementation)
	if node_id is None:
		node = tree.nodes[np.random.choice(len(tree.nodes))]
	else:
		for n in tree.nodes:
			if n.id_ == node_id:
				node = n

	if new_event is None:
		additional_info = {"node.id_":node.id_,'success':False}
		#Root node
		if node.parent is None:
			regions_available = np.arange(tree.config['number_segments'])
		else:
			regions_available = get_regions_available_profile(node.parent.get_profile())
		
		#print("node id",node.id_)
		#print("regions_available",regions_available)
		#print("events:")
		#for ev in node.events:
		#	print(ev)
		#print()

		if len(node.events) == 0:
			if len(regions_available) == 0:
				additional_info['reason'] = "No regions available (and 0 events)"
				return additional_info
			node.events = sample_events(regions_available,n_events = 1)
			#print('new event', str(node.events[0]))
			additional_info['event'] = node.events[0]
			additional_info["success"] = True
			return additional_info
		else:
			potential_region_events = get_regions_new_event(regions_available,node.events)
			#print('potential_region_events')
			#print(potential_region_events)
			
			if len(potential_region_events) == 0:
				additional_info['reason'] = "No regions available"
				return additional_info

			region = potential_region_events[np.random.randint(len(potential_region_events))]
			#print('region selected', region)
			
			if (region['previous_dir'] is None) and (region['next_dir'] is None):
				gain = np.random.randint(2)
			elif region['previous_dir'] is None:
				if len(region['segments']) == 1:
					gain = 1 if region['next_dir'] == 0 else 0
				else:
					gain = np.random.randint(2)
					if gain == region['next_dir']:
						region['segments'] = region['segments'][:-1]
			elif region['next_dir'] is None:
				if len(region['segments']) == 1:
					gain = 1 if region['previous_dir'] == 0 else 0
				else:
					gain = np.random.randint(2)
					if gain == region['previous_dir']:
						region['segments'] = region['segments'][1:]
			else:
				if (len(region['segments']) == 1):
					assert(region['previous_dir'] == region['next_dir'])
					gain = 1 if region['previous_dir'] == 0 else 0
				elif (len(region['segments']) == 2) and (region['previous_dir'] == region['next_dir']):
					gain = 1 if region['previous_dir'] == 0 else 0
				else:
					gain = np.random.randint(2)
				
				if gain == region['previous_dir']:
					region['segments'] = region['segments'][1:]
				if gain == region['next_dir']:
					region['segments'] = region['segments'][:-1]
			
			new_event = sample_events(region['segments'],n_events = 1)[0]
			new_event.gain = gain
			#print('new event', new_event)	
		add_event_correct_place(node.events,new_event)
		additional_info['success'] = True
		additional_info['event'] = new_event
		return additional_info

def remove_event(tree,node_id = None, event_idx = None):
	if node_id is None:
		n_events_node = [len(node.events) for node in tree.nodes]
		tot_events = np.sum(n_events_node)
		if tot_events == 0:
			return {"success": False,"reason":"There are no events in the tree -> we can't remove any"}
		p = np.array(n_events_node)/tot_events
		node = tree.nodes[np.random.choice(len(tree.nodes),p=p)]
	else:
		for n in tree.nodes:
			if n.id_ == node_id:
				node = n
	
	if event_idx is None:
		event = node.events.pop(np.random.randint(len(node.events)))
	else:
		event = node.events.pop(event_idx) 
	#print('event removed from node',node.id_)
	return {"success": True,"node.id_": node.id_,"event":event}

def modify_event(tree):
	#TODO: finish, not needed in theory
	n_events_node = [len(node.events) for node in tree.nodes]
	p = np.array(n_events_node)/np.sum(n_events_node)
	node = tree.nodes[np.random.choice(len(tree.nodes),p=p)]
	event = node.events[np.random.randint(len(node.events))]
	#choose if extension or reduction
	pass

def modify_sample_attachments(tree,sample_idx = None,new_node_id = None):
	if sample_idx is None:
		sample_idx = np.random.randint(len(tree.samples))

	sample = tree.samples[sample_idx]
	current_node = sample.node

	if new_node_id is None:
		#TODO: I should find a more elegant way to do this 
		other_nodes = []
		for node in tree.nodes:
			if node.id_ != current_node.id_:
				other_nodes.append(node)
		assert(len(tree.nodes) == len(other_nodes)+1)
		new_node = other_nodes[np.random.randint(len(other_nodes))]
	else:
		for n in tree.nodes:
			if n.id_ == node_id:
				new_node = n
	
	current_node.samples.remove(sample)
	sample.node = new_node
	new_node.samples.append(sample)
	return {"old_node.id_":current_node.id_,"new_node.id_":new_node.id_,"success":True}

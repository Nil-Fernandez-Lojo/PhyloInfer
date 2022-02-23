from tree.event import sample_events
from tree.node import get_regions_available_profile,Node

import numpy as np
import copy


# Utility functions
def get_regions_new_event(regions_available, events):
    # TODO: improve implementation e.g. last for loop
    if len(events) == 0:
        return [{"segments": regions_available,
                 "previous_dir": None,
                 "next_dir": None}]

    first_event = events[0]
    x = regions_available[regions_available < first_event.segments[0]]
    # event covers first segment available
    if len(x) == 0:
        potential_region_events = []
    else:
        potential_region_events = [{"segments": x, "previous_dir": None, "next_dir": first_event.gain}]
    for i in range(len(events)):
        next_seg = regions_available[regions_available > events[i].segments[-1]]
        previous_dir = events[i].gain
        if i == len(events) - 1:
            x = next_seg
            next_dir = None
        else:
            prev_seg = regions_available[regions_available < events[i + 1].segments[0]]
            x = list(set(prev_seg) & set(next_seg))
            # if adjacent events (double breakpoint), no region for new event between those 2 events
            if len(x) > 0:
                x = np.sort(x)
            next_dir = events[i + 1].gain
        # if adjacent events (double breakpoint), no region for new event between those 2 events
        if len(x) > 0:
            potential_region_events.append({"segments": x,
                                            "previous_dir": previous_dir,
                                            "next_dir": next_dir})

    # discard ones of length 1 or 2 if connected events of opposite gains
    potential_region_events_filtered = []
    for region in potential_region_events:
        if ((len(region['segments']) > 1) or
                (region['previous_dir'] is None) or
                (region['next_dir'] is None) or
                (region['previous_dir'] == region['next_dir'])):
            potential_region_events_filtered.append(region)

    return potential_region_events_filtered


def add_event_correct_place(events, new_event):
    # we do this to preserve ordering
    for i, old_event in enumerate(events):
        if new_event.segments[-1] < old_event.segments[0]:
            events.insert(i, new_event)
            return

    if new_event.segments[0] > events[-1].segments[-1]:
        events.insert(len(events), new_event)


def p_add_event(tree, node, event):
    n_nodes = len(tree.nodes)
    if node.parent is None:
        regions_available = np.arange(tree.config['number_segments'])
    else:
        regions_available = get_regions_available_profile(node.parent.get_profile())

    k = len(regions_available)
    if len(node.events) == 0:
        # print("K", K, "n_nodes", n_nodes)
        return -np.log(k * (k + 1) * n_nodes)
    else:
        potential_region_events = get_regions_new_event(regions_available, node.events)
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
        k = len(region_event['segments'])
        # print("region_event",region_event)
        return -np.log(n_nodes) - np.log(len(potential_region_events)) + gain_factor - np.log(k * (k + 1))


def get_root_nodes_affected_by_move(tree, info, include_sample=False):
    # TODO: if 2, we should check if one is the descendant of the other
    if info["move_type"] in ['add_event', 'remove_event', 'modify_event']:
        node_ids = [info["node.id_"]]
    elif info["move_type"] == 'prune_and_reattach':
        node_ids = [info["root_subtree.id_"]]
    elif info["move_type"] == 'swap_events_2_nodes':
        node_ids = [info["node_0.id_"], info["node_1.id_"]]
    elif info["move_type"] == 'modify_sample_attachments':
        if include_sample:
            node_ids = [info["old_node.id_"], info["new_node.id_"]]
        else:
            node_ids = []
    elif info["move_type"] in ['split_node', "merge_nodes"]:
        node_ids = [info["parent_node.id_"]]


    root_nodes_to_apply_updates = []
    for i in range(len(tree.nodes)):
        if tree.nodes[i].id_ in node_ids:
            root_nodes_to_apply_updates.append(tree.nodes[i])
            if len(root_nodes_to_apply_updates) == len(node_ids):
                break
    return root_nodes_to_apply_updates


def update_tree_after_move(tree, info):
    root_nodes_to_apply_updates = get_root_nodes_affected_by_move(tree, info, include_sample=False)
    for node in root_nodes_to_apply_updates:
        tree._update_profiles(node)
        tree.update_events(node)

def can_be_merged(parent,child):
    if (len(parent.events) == 0) or (len(child.events) == 0):
        return True

    #Check if overlapping events
    seg_events_parent = []
    for event in parent.events:
        seg_events_parent.extend(event.segments)
    seg_events_parent = set(seg_events_parent)

    seg_events_child = []
    for event in child.events:
        seg_events_child.extend(event.segments)
    seg_events_child = set(seg_events_child)

    if len(seg_events_parent.intersection(seg_events_child)) != 0:
        return False

    # TODO: I do not like this because ideally adjacent events would be merged but then if we do this we can't
    # compute the log proposal anymore. Moreover in the previous lines of code if a gain cancels a loss, we can't
    # merge them either, which ideally we should be able to
    grand_parent_profile = parent.get_parent_profile()
    regions_available = get_regions_available_profile(grand_parent_profile).tolist()

    list_events = parent.events + child.events
    list_events.sort(key= lambda x:x.segments[0])
    for i in range(len(list_events)-1):
        if  list_events[i].gain == list_events[i+1].gain:
            end_segment = list_events[i].segments[-1]
            start_segment = list_events[i+1].segments[0]
            idx = regions_available.index(end_segment)
            if regions_available[idx] + 1 == start_segment:
                return False

    return True

# General move function
def move(tree, move_type,
         node_id=None,
         node_1_id=None,
         event_to_add=None,
         event_idx_to_remove=None,
         sample_idx=None):
    tree_modified = tree.get_copy()
    if move_type == 'prune_and_reattach':
        info = prune_and_reattach(tree_modified)
    elif move_type == 'swap_events_2_nodes':
        info = swap_events_2_nodes(tree_modified, node_0_id=node_id, node_1_id=node_1_id)
    elif move_type == 'add_event':
        info = add_event(tree_modified, node_id=node_id, new_event=event_to_add)
    elif move_type == 'remove_event':
        info = remove_event(tree_modified, node_id=node_id, event_idx=event_idx_to_remove)
    elif move_type == 'modify_event':
        info = modify_event(tree_modified)
    elif move_type == 'modify_sample_attachments':
        info = modify_sample_attachments(tree_modified, sample_idx=sample_idx, new_node_id=node_id)
    elif move_type == 'split_node':
        info = split_node(tree_modified,node_id=node_id)
    elif move_type == 'merge_nodes':
        info = merge_nodes(tree_modified,parent_node_id=node_id,child_node_id=node_1_id)

    info['move_type'] = move_type
    if info["success"]:
        update_tree_after_move(tree_modified, info)

    return tree_modified, info


# different moves
def prune_and_reattach(tree, root_subtree_idx=None, new_parent_subtree_id=None):
    # TODO: if tree of size 1 or 2, there will be an error
    # TODO: add preference for smaller subtrees like in SCICONE

    if root_subtree_idx is None:
        nodes = copy.copy(tree.nodes)
        nodes.remove(tree.root)
        if len(tree.root.children) == 1:
            nodes.remove(tree.root.children[0])
        root_subtree = nodes[np.random.choice(len(nodes))]
    else:
        root_subtree = tree.nodes[root_subtree_idx]

    if new_parent_subtree_id is None:
        nodes_id_forbidden = tree.get_children_id(root_subtree)
        nodes_id_forbidden.append(root_subtree.id_)
        nodes_id_forbidden.append(root_subtree.parent.id_)
        remaining_nodes = []
        for node in tree.nodes:
            if node.id_ not in nodes_id_forbidden:
                remaining_nodes.append(node)
        new_parent_subtree = remaining_nodes[np.random.choice(len(remaining_nodes))]
    else:
        new_parent_subtree = tree.nodes[new_parent_subtree_id]

    for n in root_subtree.parent.children:
        if n == root_subtree:
            root_subtree.parent.children.remove(n)
            break

    new_parent_subtree.add_child(root_subtree)
    additional_info = {"root_subtree.id_": root_subtree.id_, "new_parent_subtree.id_": new_parent_subtree.id_,
                       'success': True}
    return additional_info


def swap_events_2_nodes(tree, node_0_id=None, node_1_id=None):
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

    additional_info = {"node_0.id_": node_0.id_, "node_1.id_": node_1.id_}
    # print("we swap the events of:",node_0.id_, node_1.id_)

    events_0 = node_0.events
    node_0.events = node_1.events
    node_1.events = events_0
    return additional_info


def add_event(tree, node_id=None, new_event=None):
    # TODO: should improve this move (move by itself and maybe just implementation)
    if node_id is None:
        node = tree.nodes[np.random.choice(len(tree.nodes))]
    else:
        for n in tree.nodes:
            if n.id_ == node_id:
                node = n

    if new_event is None:
        additional_info = {"node.id_": node.id_, 'success': False}
        # Root node
        if node.parent is None:
            regions_available = np.arange(tree.config['number_segments'])
        else:
            regions_available = get_regions_available_profile(node.parent.get_profile())

        # print("node id",node.id_)
        # print("regions_available",regions_available)
        # print("events:")
        # for ev in node.events:
        # print(ev)
        # print()

        if len(node.events) == 0:
            if len(regions_available) == 0:
                additional_info['reason'] = "No regions available (and 0 events)"
                return additional_info
            node.events = sample_events(regions_available, n_events=1)
            # print('new event', str(node.events[0]))
            additional_info['event'] = node.events[0]
            additional_info["success"] = True
            return additional_info
        else:
            potential_region_events = get_regions_new_event(regions_available, node.events)
            # print('potential_region_events')
            # print(potential_region_events)

            if len(potential_region_events) == 0:
                additional_info['reason'] = "No regions available"
                return additional_info

            region = potential_region_events[np.random.randint(len(potential_region_events))]
            # print('region selected', region)

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
                if len(region['segments']) == 1:
                    assert (region['previous_dir'] == region['next_dir'])
                    gain = 1 if region['previous_dir'] == 0 else 0
                elif (len(region['segments']) == 2) and (region['previous_dir'] == region['next_dir']):
                    gain = 1 if region['previous_dir'] == 0 else 0
                else:
                    gain = np.random.randint(2)

                if gain == region['previous_dir']:
                    region['segments'] = region['segments'][1:]
                if gain == region['next_dir']:
                    region['segments'] = region['segments'][:-1]

            new_event = sample_events(region['segments'], n_events=1)[0]
            new_event.gain = gain
        # print('new event', new_event)
        add_event_correct_place(node.events, new_event)
        additional_info['success'] = True
        additional_info['event'] = new_event
        return additional_info


def remove_event(tree, node_id=None, event_idx=None):
    if node_id is None:
        n_events_node = [len(node.events) for node in tree.nodes]
        tot_events = np.sum(n_events_node)
        if tot_events == 0:
            return {"success": False, "reason": "There are no events in the tree -> we can't remove any"}
        p = np.array(n_events_node) / tot_events
        node = tree.nodes[np.random.choice(len(tree.nodes), p=p)]
    else:
        for n in tree.nodes:
            if n.id_ == node_id:
                node = n

    if event_idx is None:
        event = node.events.pop(np.random.randint(len(node.events)))
    else:
        event = node.events.pop(event_idx)
    # print('event removed from node',node.id_)
    return {"success": True, "node.id_": node.id_, "event": event}


def event_can_be_extended(node, event_idx):
    # TODO: TEST

    parent_profile = node.get_parent_profile()
    regions_available = list(get_regions_available_profile(parent_profile))
    index_start = regions_available.index(node.events[event_idx].segments[0])
    index_end = regions_available.index(node.events[event_idx].segments[-1])

    possible_directions_extensions = []

    # check if can be extended on the left
    if index_start != 0:
        #first event
        if event_idx == 0:
            possible_directions_extensions.append('left')
        #previous event opposite direction
        elif node.events[event_idx-1].gain != node.events[event_idx].gain:
            if node.events[event_idx - 1].segments[-1] != regions_available[index_start-1]:
                possible_directions_extensions.append('left')
        # previous event same direction
        else:
            if node.events[event_idx - 1].segments[-1] != regions_available[index_start-2]:
                possible_directions_extensions.append('left')

    # check if can be extended on the right
    if index_end != len(regions_available)-1:
        # last event
        if event_idx == len(node.events)-1:
            possible_directions_extensions.append('right')
        # next event opposite direction
        elif node.events[event_idx+1].gain != node.events[event_idx].gain:
            if node.events[event_idx + 1].segments[0] != regions_available[index_end +1]:
                possible_directions_extensions.append('right')
        # next event same direction
        else:
            if node.events[event_idx + 1].segments[0] != regions_available[index_end + 2]:
                possible_directions_extensions.append('right')



    segment_extension = dict()
    if 'left' in possible_directions_extensions:
        segment_extension['left'] = regions_available[index_start-1]
    if 'right' in possible_directions_extensions:
        segment_extension['right'] = regions_available[index_end+1]

    return possible_directions_extensions,segment_extension

def get_possible_modifications_event(node, event_idx):
    possible_modifications = []
    possible_directions_extensions,segment_extension = event_can_be_extended(node, event_idx)
    # can not be shortened
    if len(node.events[event_idx].segments) > 1:
        possible_modifications.append("reduction")
    if len(possible_directions_extensions)>0:
        possible_modifications.append("extension")
    return possible_modifications,possible_directions_extensions,segment_extension

def modify_event(tree):
    # TODO: TEST
    n_events_node = [len(node.events) for node in tree.nodes]
    p = np.array(n_events_node) / np.sum(n_events_node)
    node = tree.nodes[np.random.choice(len(tree.nodes), p=p)]
    event_idx = np.random.randint(len(node.events))
    event = node.events[event_idx]

    (possible_modifications,
     possible_directions_extensions,
     segment_extension) = get_possible_modifications_event(node, event_idx)

    if len(possible_modifications) == 0:
        return {"success": False, "reason": "selected event could not be modified"}

    modification = np.random.choice(possible_modifications)

    if modification == "reduction":
        direction = np.random.choice(["left", "right"])
        if direction == "left":
            event.segments = event.segments[1:]
        else:
            event.segments = event.segments[:-1]
    else:
        direction = np.random.choice(possible_directions_extensions)
        if direction == "left":
            event.segments = np.insert(event.segments, 0, segment_extension[direction])
        else:
            event.segments = np.insert(event.segments, len(event.segments), segment_extension[direction])

    return {"success": True,
            "node.id_": node.id_,
            'event_idx': event_idx,
            "modification": modification,
            "direction": direction,
            "possible_modifications": possible_modifications}


def modify_sample_attachments(tree, sample_idx=None, new_node_id=None):
    if sample_idx is None:
        sample_idx = np.random.randint(len(tree.samples))

    sample = tree.samples[sample_idx]
    current_node = sample.node

    if new_node_id is None:
        # TODO: I should find a more elegant way to do this
        other_nodes = []
        for node in tree.nodes:
            if node.id_ != current_node.id_:
                other_nodes.append(node)
        assert (len(tree.nodes) == len(other_nodes) + 1)
        new_node = other_nodes[np.random.randint(len(other_nodes))]
    else:
        for n in tree.nodes:
            if n.id_ == new_node_id:
                new_node = n

    current_node.samples.remove(sample)
    sample.node = new_node
    new_node.samples.append(sample)
    return {"old_node.id_": current_node.id_, "new_node.id_": new_node.id_, "success": True}


def split_node(tree,node_id=None):
    if node_id is None:
        idx = np.random.choice(len(tree.nodes))
        node = tree.nodes[idx]
    else:
        for n in tree.nodes:
            if n.id_ == node_id:
                node = n
                break

    tree.node_max_id += 1
    new_node = Node(tree.node_max_id, tree.config)
    if node.parent is None:
        tree.root = new_node
    else:
        node.parent.children.remove(node)
        node.parent.add_child(new_node)

    new_node.add_child(node)

    for event in node.events:
        if np.random.randint(2):
            node.events.remove(event)
            new_node.events.append(event)

    for sample in node.samples:
        if np.random.randint(2):
            node.samples.remove(sample)
            new_node.add_sample(sample)
    tree.nodes.append(new_node)
    return {"child_node.id_": node.id_, "parent_node.id_": new_node.id_, "success": True}

def merge_nodes(tree,parent_node_id=None,child_node_id=None):
    if parent_node_id is None:
        potential_merges = []
        for node in tree.nodes:
            for child in node.children:
                if can_be_merged(node,child):
                    potential_merges.append((node,child))
        if len(potential_merges) == 0:
            return {"success": False, "reason":"No nodes could be merged"}

        merge = potential_merges[np.random.randint(len(potential_merges))]
        parent_node = merge[0]
        child_node = merge[1]
    else:
        parent_node = None
        child_node = None
        for n in tree.nodes:
            if n.id_ == parent_node_id:
                parent_node = n
            elif n.id_ == child_node_id:
                child_node = n
    parent_node.children.remove(child_node)
    tree.nodes.remove(child_node)
    parent_node.events.extend(child_node.events)
    parent_node.events.sort(key = lambda x: x.segments[0])
    for sample in child_node.samples:
        parent_node.add_sample(sample)
    for child_child in child_node.children:
        parent_node.add_child(child_child)

    return {"success": True, "parent_node.id_":parent_node.id_, "child_node.id_":child_node.id_}

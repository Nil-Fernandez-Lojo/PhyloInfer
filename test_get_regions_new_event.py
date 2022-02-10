from tree.event import Event
from tree.node import change_profile_to_events
from tree.util import events_to_vector

from inference.moves import get_regions_new_event
import numpy as np
number_segments = 10
regions_available = np.array([0,1,2,3,4,5,6,7,8,9])
events = [Event([2],1),Event([4],0),Event([6,7],1),Event([8,9],0)]

change_profile = events_to_vector(events,number_segments)
print(change_profile)
events_2 = change_profile_to_events(change_profile,regions_available)
for ev in events_2:
	print(ev)
# regions = get_regions_new_event(regions_available,events)
# for r in regions:
# 	print(r)
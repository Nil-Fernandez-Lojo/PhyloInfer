from tree.event import Event
from inference.moves import get_regions_new_event
import numpy as np

regions_available = np.array([0,2,3,4,5,6,7,8,9])
events = [Event([0,2],1),Event([4],0),Event([6,7],1)]
regions = get_regions_new_event(regions_available,events)
for r in regions:
	print(r)
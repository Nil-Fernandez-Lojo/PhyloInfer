import numpy as np
from tree.util import n_bkp, n_g, c_db

def sample_events(regions_available, p_new_event=0, n_events=None):
    k = len(regions_available)
    if k == 0:
        return []

    if n_events is None:
        p_events = p_new_event ** np.arange(k + 1)
        p_events = p_events / np.sum(p_events)
        n_events = np.random.choice(k + 1, p=p_events)
    # print('n_events', n_events)
    if n_events == 0:
        return []

    n_a = np.zeros(n_events)  # number of sets of events given the number of breakpoints
    for a in range(n_events):
        if a < 2 * n_events - (k + 1):
            n_a[a] = 0
        else:
            n_a[a] = n_bkp(n_events, k, a) * n_g(n_events, a) * c_db(n_events, a)
    p_a = n_a / np.sum(n_a)
    a = np.random.choice(n_events, p=p_a)
    b = 2 * n_events - a  # number of breakpoints

    fixed_events = np.random.choice(np.arange(1, n_events), size=a, replace=False)

    b_list_idx = np.random.choice(k + 1, size=b, replace=False)
    b_list_idx = np.sort(b_list_idx)
    # print('b_list_idx:',b_list_idx)
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
                b_i += 1
            else:
                gain = np.random.randint(2)
                b_i += 2
        segments = regions_available[b_list_idx[b_i]:b_list_idx[b_i + 1]]
        list_events.append(Event(segments, gain))

    return list_events


class Event:
    def __init__(self, segments, gain):
        assert (gain in [0, 1]), 'gain must be either 0 (loss) or 1 (gain)'
        self.segments = segments
        self.gain = gain

    def __str__(self):
        return str(self.segments) + " " + str(self.gain)
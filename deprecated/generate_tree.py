import numpy as np
# Script to sample an undirected labeled tree uniformly at random
# It returns the list of directed edges

def generate_prufer_sequence(n):
	return np.random.randint(0, high=n+1, size=n)

def decode_prufer(p):
    p = list(p)
    vertices = set(range(len(p) + 2))
    for (i, u) in enumerate(p):
        v = min(vertices.difference(p[i:]))
        vertices.remove(v)
        yield u, v
    yield tuple(vertices)

def sample_tree(n):
	# TODO: case n<2
	prufer = generate_prufer_sequence(n-2)
	edges = decode_prufer(prufer)
	root = np.random.randint(n)
	
print(list(decode_prufer([3,3,3,4])))
	


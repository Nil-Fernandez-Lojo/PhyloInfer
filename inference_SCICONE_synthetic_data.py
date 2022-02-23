import pandas as pd
import numpy as np
import copy
from loky import get_reusable_executor

from tree.sample import Sample
from tree.tree import Tree
from infer_tree import get_moves_setup,mcmc,mc3,annealed_smc


inference_algorithm = "mcmc" # must be asmc or mcmc

path_to_save_trees = 'trees_simulation.txt'
path_reads_per_bin = "SCICONE_synthetic_data/reads_per_bin.csv"
path_segment_size = "SCICONE_synthetic_data/segmented_region_sizes.csv"
setup_simulation = "size_tree_NOT_known"

n_cycles = 10000
n_iter_cycle = 100
#list_beta = [1]*10
# list_beta = [1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10]
list_beta = [1, 1 / 2, 1 / 2, 1 / 4]
#list_beta = [1]
iter_equal_move_accepted = True

n_particles = 20
n_iter_smc = 1000
epsilon_ress = 0.75
alpha = 0.999 # not used at the moment

p_new_event = 0.1
k_n_nodes = 0.0001

verbose = 1

move_weights = {"swap_events_2_nodes": 1, "add_event": 1, "remove_event": 1, "remove_then_add_event": 1,
                "modify_sample_attachments": 1, "prune_and_reattach": 1, "split_node": 1, "merge_nodes": 1,
                "modify_event": 1}

#bins_per_segment = pd.read_csv(path_segment_size, header=None).to_numpy(dtype=int)[:,0]
#34-46 209-219 250-301 506-516 596-607
bins_per_segment = np.array([34, 13, 162, 11,30, 52,204,11,79,12,392])
print(np.cumsum(bins_per_segment))
reads_per_bin = pd.read_csv(path_reads_per_bin, header=None).to_numpy(dtype=int)



n_segments = len(bins_per_segment)
(n_cells, n_bins) = reads_per_bin.shape
reads_per_segment= np.zeros((n_cells,n_segments),dtype=int)
bin_start = 0

for i,seg_size in enumerate(bins_per_segment):
    reads_per_segment[:,i] = np.sum(reads_per_bin[:,bin_start:(bin_start+seg_size)],axis=1)
    bin_start += seg_size

# super_segment_length = [8,1,4,4,2,5,1,4]#in number of segments
# n_super_segments = len(super_segment_length)
# reads_per_super_segment = np.zeros((n_cells,n_super_segments),dtype=int)
# bins_per_super_segment = np.zeros(n_super_segments)
# bin_start = 0
# for i,seg_size in enumerate(super_segment_length):
#     reads_per_super_segment[:,i] = np.sum(reads_per_segment[:,bin_start:(bin_start+seg_size)],axis=1)
#     bins_per_super_segment[i] = np.sum(bins_per_segment[bin_start:(bin_start+seg_size)])
#     bin_start += seg_size


config = {'number_segments': n_segments,
          'p_new_event': p_new_event,
          'length_segments': bins_per_segment,
          "k_n_nodes": k_n_nodes}

samples = []
for i,read_count in enumerate(reads_per_segment):
    samples.append(Sample(i, config, read_count=read_count))


moves = get_moves_setup(setup_simulation)
number_nodes = 3
if (inference_algorithm == 'mcmc'):
    tree_start = Tree(number_nodes, config)
    tree_start.randomly_assign_samples(samples)
else:
    tree_start = []
    for i in range(n_particles):
        t = Tree(number_nodes, config)
        t.randomly_assign_samples(copy.deepcopy(samples))
        tree_start.append(t)

if inference_algorithm == 'mcmc':
    if len(list_beta) == 1:
        list_trees = mcmc(tree_start,
                          moves,
                          move_weights,
                          n_iter_cycle,
                          iter_equal_move_accepted=iter_equal_move_accepted,
                          verbose=verbose)

    else:
        list_trees = mc3(moves,
                         tree_start,
                         move_weights,
                         n_iter_cycle,
                         n_cycles,
                         list_beta,
                         iter_equal_move_accepted=iter_equal_move_accepted,
                         verbose=verbose,
                         path_to_save_trees = path_to_save_trees)
else:
    (weights,trees) = annealed_smc(moves,
                                   move_weights,
                                   n_particles,
                                   n_iter_smc,
                                   tree_start,
                                   epsilon_ress,
                                   alpha,
                                   verbose=verbose)
    list_trees = trees[-1]

best_post = list_trees[0].get_log_posterior(update=False)
best_tree = list_trees[0]

for i, t in enumerate(list_trees):
    post = t.get_log_posterior(update=False)
    print("Posterior distributions")
    print(i, post)
    if post > best_post:
        best_post = post
        best_tree = t
print("best_tree, post:", best_post)
best_tree.print(add_samples=False)




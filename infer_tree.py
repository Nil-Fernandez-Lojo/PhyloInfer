from tree.tree import Tree
from inference.moves import move, p_add_event,get_root_nodes_affected_by_move,can_be_merged,get_possible_modifications_event
from inference.util import ress,nextAnnealingParameter
import numpy as np
import copy
import math
from loky import get_reusable_executor
import time

# TODO: use random number generator of numpy instead!
# TODO: use one random number generator for each core/iteration so determistic result when simulation on multiple cores
np.random.seed(2020)
MAX_WORKERS = 10
# SIDE FUNCTIONS
def compute_log_proposal_ratio(move_type,
                               move_weights,
                               info,
                               tree,
                               new_tree):
    if move_type in ["swap_events_2_nodes", "modify_sample_attachments", "prune_and_reattach"]:
        # TODO check that prune_and_reattach is symmetric (they say it in SCICONE) and I am pretty sure it is but need
        # to prove it
        log_proposal_ratio = 0
    elif move_type == 'modify_event':
        node = new_tree.get_node_from_id(info["node.id_"])
        (possible_modifications_new_tree,
         possible_directions_extensions,
         segment_extension) = get_possible_modifications_event(node, (info["event_idx"]))
        if len(possible_modifications_new_tree) == len(info['possible_modifications']):
            log_proposal_ratio = 0
        elif len(info['possible_modifications']) == 2:
            log_proposal_ratio = 0-(-np.log(2))
        else:
            log_proposal_ratio = -np.log(2)

    elif move_type in ["merge_nodes", "split_node"]:
        number_potential_merges = 0
        if move_type == "merge_nodes":
            tree_to_merge = tree
            tree_to_split = new_tree
            node_id_split = info["parent_node.id_"]
        else:
            tree_to_merge = new_tree
            tree_to_split = tree
            node_id_split = info["child_node.id_"]
        # TODO this was already computed in moves.merge_nodes...
        # compute log_p_merge_node
        for node in tree_to_merge.nodes:
            for child in node.children:
                if can_be_merged(node, child):
                    number_potential_merges +=1
        if number_potential_merges == 0:
            print("there are 0 number_potential_merges, this is an error")
            print(info)
            print("tree")
            print(tree)
            print("new tree")
            print(new_tree)
            exit()
        log_p_merge_node = - np.log(number_potential_merges) + np.log(move_weights["merge_nodes"])

        # compute log_p_split_nodes
        node_split = tree_to_split.get_node_from_id(node_id_split)
        if node_split is None:
            print(info)
            print("tree")
            print(tree)
            print("nodes id", [node.id_ for node in tree.nodes])
            print("new tree")
            print(new_tree)
        n_events = len(node_split.events)
        n_samples = len(node_split.samples)
        log_p_split_nodes = (-np.log(len(tree_to_split.nodes))-np.log(2)*(n_events+n_samples) +
                             np.log(move_weights["split_node"]))

        if move_type == "merge_nodes":
            log_proposal_ratio = log_p_split_nodes - log_p_merge_node
        else:
            log_proposal_ratio = log_p_merge_node - log_p_split_nodes

    else:
        # move_type in ["add_event", "remove_event"]
        event = info["event"]
        if move_type == "add_event":
            node_to_add_event = tree.get_node_from_id(info["node.id_"])
        else:
            node_to_add_event = new_tree.get_node_from_id(info["node.id_"])

        # for j in range(len(tree.nodes)):
        #     if tree.nodes[j].id_ == info["node.id_"]:
        #         if move_type == "add_event":
        #             node_to_add_event = tree.nodes[j]
        #         else:
        #             node_to_add_event = new_tree.nodes[j]
        #         break

        if move_type == "add_event":
            log_p_add_event = p_add_event(tree, node_to_add_event, event)
            log_p_remove_event = - np.log(new_tree.get_number_events())

            log_p_forw_mv = np.log(move_weights[move_type]) + log_p_add_event
            log_p_back_mv = np.log(move_weights["remove_event"]) + log_p_remove_event
        # print("log_p_add_event", log_p_add_event,"log_p_remove_event", log_p_remove_event)
        else:
            log_p_add_event = p_add_event(new_tree, node_to_add_event, event)
            log_p_remove_event = - np.log(tree.get_number_events())

            log_p_forw_mv = np.log(move_weights[move_type]) + log_p_remove_event
            log_p_back_mv = np.log(move_weights["add_event"]) + log_p_add_event

        # print("log_prior_ratio",new_tree.get_log_prior()-tree.get_log_prior())
        # print("log_likelihood_ratio",new_tree.get_log_likelihood()-tree.get_log_likelihood())
        # print("log_proposal_ratio",log_p_back_mv-log_p_forw_mv )
        log_proposal_ratio = log_p_back_mv - log_p_forw_mv

    return log_proposal_ratio


def compute_probability_acceptance(move_type,
                                   move_weights,
                                   info,
                                   tree,
                                   new_tree,
                                   beta,
                                   annealing_on_prior):
    if not info["success"]:
        p_accept = 0
    else:
        if move_type == "modify_sample_attachments":
            log_prior_ratio = 0
        else:
            # TODO: for prune_and_reattach it is either 0 or -inf, we do not have to recompute everything
            root_nodes_affected_by_move = get_root_nodes_affected_by_move(new_tree, info, include_sample=False)
            log_prior_ratio = (new_tree.get_log_prior(root_nodes_to_update=root_nodes_affected_by_move)
                              - tree.get_log_prior(root_nodes_to_update=None))
        if math.isinf(log_prior_ratio):
            return 0

        log_proposal_ratio = compute_log_proposal_ratio(move_type,
                                                        move_weights,
                                                        info,
                                                        tree,
                                                        new_tree)
        root_nodes_affected_by_move = get_root_nodes_affected_by_move(new_tree, info, include_sample=True)

        log_likelihood_ratio = (new_tree.get_log_likelihood(root_nodes_to_update=root_nodes_affected_by_move)
                                - tree.get_log_likelihood(root_nodes_to_update=None))

        log_likelihood_ratio *= beta
        if annealing_on_prior:
            log_prior_ratio *= beta
        if (log_prior_ratio + log_likelihood_ratio + log_proposal_ratio) > 0:
            p_accept = 1
        else:
            p_accept = np.exp(log_prior_ratio + log_likelihood_ratio + log_proposal_ratio)

    return p_accept


def select_move_type(moves,move_weights,tree):
    # Get moves_allowed at this iteration
    moves_allowed = []
    if "add_event" in moves:
        moves_allowed.append("add_event")
    if len(tree.nodes) > 2:
        if "prune_and_reattach" in moves:
            moves_allowed.append("prune_and_reattach")
    if tree.get_number_events() != 0:
        if "remove_event" in moves:
            moves_allowed.append("remove_event")
        if "remove_then_add_event" in moves:
            moves_allowed.append("remove_then_add_event")
        if "modify_event" in moves:
            moves_allowed.append("modify_event")
    if len(tree.nodes) > 1:
        if "swap_events_2_nodes" in moves:
            moves_allowed.append(
                "swap_events_2_nodes")  # TODO: we should not allow to swap 2 nodes with the same events
        if "merge_nodes" in moves:
            moves_allowed.append("merge_nodes")
        if "modify_sample_attachments" in moves:
               moves_allowed.append("modify_sample_attachments")
    if "split_node" in moves:
        moves_allowed.append("split_node")

    # Select move type
    p = [move_weights[mv] for mv in moves_allowed]
    p = np.array(p) / np.sum(p)
    move_type = np.random.choice(moves_allowed, p=p)
    return move_type

def mh_move(tree,
            move_type,
            move_weights,
            beta,
            annealing_on_prior = True):
    # Do move
    new_tree, info = move(tree, move_type)

    # Compute probability of accepting move
    p_accept = compute_probability_acceptance(move_type,
                                              move_weights,
                                              info,
                                              tree,
                                              new_tree,
                                              beta,
                                              annealing_on_prior)

    # Decision to accept it
    if np.random.choice(2, p=[1 - p_accept, p_accept]):
        accepted = True
    else:
        accepted = False

    return accepted, new_tree, info

# SINGLE CHAIN MCMC
def mcmc(tree_start,
         moves,
         move_weights,
         n_samples,
         beta=1,
         iter_equal_move_accepted=False,
         verbose=1,
         max_proposed_moves=10 ** 6):
    list_trees = [tree_start]
    tree = tree_start
    i = 0

    if iter_equal_move_accepted:
        max_proposed_moves = n_samples

    while (i < max_proposed_moves) and len(list_trees) < n_samples:
        # select move type
        move_type = select_move_type(moves,move_weights,tree)
        (accepted, new_tree, info) = mh_move(tree, move_type, move_weights, beta)
        if accepted:
            list_trees.append(new_tree)
            tree = new_tree
            #print("number of nodes: ", len(new_tree.nodes))
            #if (move_type in ["split_node", "merge_nodes"]):
            #    print("move accepted", move_type,"number of nodes", len(tree.nodes))
            if verbose > 1:
                print(i, "move_type", move_type, info)
                #print(new_tree)
        i += 1
    return list_trees


# MC3
def mc3(moves,
        tree_start,
        move_weights,
        n_iter_cycle,
        n_cycles,
        list_beta,
        iter_equal_move_accepted = True,
        verbose=1,
        path_to_save_trees=None):

    if path_to_save_trees is not None:
        tree_start.to_file(path_to_save_trees,'w',0)
    list_trees = []
    tree_start_chain = dict()
    for j in range(len(list_beta)):
        tree_start_chain[j] = tree_start
    best_post = float("-inf")
    executor = get_reusable_executor(max_workers=min(MAX_WORKERS,len(list_beta)))
    for i in range(n_cycles):
        print("cycle:", i, "out of", n_cycles)
        if verbose>0:
            time_start = time.time()
        if MAX_WORKERS == 1:
            list_trees_cycles = []
            for j in range(len(list_beta)):
                list_trees_cycles.append(mcmc(tree_start_chain[j],
                                              moves,
                                              move_weights,
                                              n_iter_cycle,
                                              list_beta[j],
                                              iter_equal_move_accepted,
                                              verbose))
        else:
            jobs = []
            for j in range(len(list_beta)):
                jobs.append((tree_start_chain[j],
                             moves,
                             move_weights,
                             n_iter_cycle,
                             list_beta[j],
                             iter_equal_move_accepted,
                             verbose))
            list_trees_cycles = list(executor.map(lambda p: mcmc(*p), jobs, chunksize=1))

        list_trees.extend(list_trees_cycles[0][1:])
        for j in range(len(list_beta)):
            tree_start_chain[j] = list_trees_cycles[j][-1]

        if verbose > 0:
            print('n samples', len(list_trees))
            for k, t in enumerate(list_trees_cycles[0]):
                post = t.get_log_posterior(update=False)
                if post > best_post:
                    best_post = post
            print("best_post", best_post)
        # choose 2 chains at random
        (beta_1_idx, beta_2_idx) = np.random.choice(len(list_beta), size=2, replace=False)
        beta_1 = list_beta[beta_1_idx]
        beta_2 = list_beta[beta_2_idx]

        p_exchange = min(1, np.exp((beta_1 - beta_2) * (
                tree_start_chain[beta_2_idx].get_log_posterior(update=False) -
                tree_start_chain[beta_1_idx].get_log_posterior(update=False))))
        if np.random.choice(2,p=[1-p_exchange,p_exchange]):
            temp = tree_start_chain[beta_1_idx]
            tree_start_chain[beta_1_idx] = tree_start_chain[beta_2_idx]
            tree_start_chain[beta_2_idx] = temp
            if beta_1_idx == 0:
                list_trees.append(tree_start_chain[beta_1_idx])
            elif beta_2_idx == 0:
                list_trees.append(tree_start_chain[beta_2_idx])

        if path_to_save_trees is not None:
            tree_start_chain[0].to_file(path_to_save_trees, 'a',i+1)
        if verbose>0:
            print('time taken by cycle', time.time() - time_start)

    return list_trees


def asmc_iteration(moves,move_weights,tree,phi_r,phi_r_minus_1):
    move_type = select_move_type(moves, move_weights, tree)
    (accepted, new_tree, info) = mh_move(tree, move_type, move_weights, phi_r, annealing_on_prior=False)
    log_w = tree.get_log_likelihood() * (phi_r - phi_r_minus_1)
    if accepted:
        return new_tree,log_w
    else:
        return tree, log_w

def annealed_smc(moves,
                 move_weights,
                 n_particles,
                 n_iter,
                 trees_start,
                 epsilon_ress,
                 alpha,
                 verbose=1):
    executor = get_reusable_executor(max_workers=MAX_WORKERS)

    # n_moves_start = 10
    # for j in range(n_moves_start):
    #     jobs = []
    #     for i in range(n_particles):
    #         jobs.append((moves, move_weights, trees_start[i], 1, 0))
    #     # print('time before extension', time.time()-time_start_iteration)
    #     results = list(executor.map(lambda p: asmc_iteration(*p), jobs, chunksize=round(n_particles / 10)))
    #     # print('time after extension', time.time()-time_start_iteration)
    #     for i in range(n_particles):
    #         trees_start[i] = results[i][0]


    trees = trees_start
    list_trees = [trees]
    weights = np.ones((n_iter,n_particles))
    log_w = np.zeros(n_particles)
    W = np.ones(n_particles)/n_particles

    #phi_r = 0.01
    for r in range(1,n_iter+1):
        # time_start_iteration = time.time()
        if verbose>0:
            print('iteration', r, 'out of',n_iter)
        phi_r_minus_1 = ((r-1)/n_iter)**3
        phi_r = (r/n_iter)**3
        # log_likelihood = np.array([t.get_log_likelihood(root_nodes_to_update=None) for t in trees])
        # phi_r_minus_1 = phi_r
        # phi_r = nextAnnealingParameter(W,log_likelihood,phi_r_minus_1,alpha)
        # print('phi_r',phi_r,'phi_r_minus_1',phi_r_minus_1)
        jobs = []
        for i in range(n_particles):
            jobs.append((moves, move_weights, trees[i], phi_r, phi_r_minus_1))
        #print('time before extension', time.time()-time_start_iteration)
        results = list(executor.map(lambda p: asmc_iteration(*p), jobs, chunksize=round(n_particles/10)))
        #print('time after extension', time.time()-time_start_iteration)
        for i in range(n_particles):
            trees[i] = results[i][0]
            log_w[i] = log_w[i]+results[i][1]
        #print('time after unpacking', time.time()-time_start_iteration)
        # for i in range(n_particles):
        #     tree = trees[i]
        #     (new_tree,w_i) = asmc_iteration(moves, move_weights, tree, phi_r, phi_r_minus_1)
        #     w[i] = w_i
        #     trees[i] = new_tree
        # this line is used to avoid an overflow
        log_w = log_w-np.max(log_w)
        w = np.exp(log_w)
        W = w/np.sum(w)
        if r < n_iter and ress(W)< epsilon_ress:
            particule_idx = np.random.choice(n_particles, size=n_particles, p=W)
            temp = copy.copy(trees)
            for part,old_part in enumerate(particule_idx):
                trees[part] = temp[old_part]
            log_w = np.zeros(n_particles)
        #print('time after resampling', time.time()-time_start_iteration)
        list_trees.append(trees)
        weights[r-1,:] = W
        #print('time after archiving', time.time()-time_start_iteration)


        best_post = float('-inf')
        for tree in trees:
            post = tree.get_log_posterior(update=False)
            if post > best_post:
                best_post = post
        print('best posterior', best_post)
        print("ress",ress(W))
        #print('time after printing best post', time.time()-time_start_iteration)


    return weights,list_trees

def get_moves_setup(setup_simulation):
    if setup_simulation == "tree_known":
        return ["modify_sample_attachments"]
    elif setup_simulation == "tree_topology_and_sample_assignments_known":
        return ["add_event", "remove_event","modify_event"]
    elif setup_simulation == "tree_topology_known":
        return ["modify_sample_attachments", "add_event", "remove_event", "modify_event"]
    elif setup_simulation == "size_tree_known":
        return ["modify_sample_attachments", "add_event", "remove_event", "prune_and_reattach", "modify_event"]
    else:
        # setup_simulation == "size_tree_NOT_known"
        return ["modify_sample_attachments", "add_event", "remove_event", "modify_event", "prune_and_reattach",
                "split_node", "merge_nodes"]

def get_random_tree_start(setup_simulation,tree):
    if setup_simulation == "tree_known":
        tree_start = tree.get_copy()
        tree_start.remove_samples()
        samples = tree.get_samples_unassigned_copy()
        tree_start.randomly_assign_samples(samples)

    elif setup_simulation == "tree_topology_and_sample_assignments_known":
        tree_start = tree.get_copy()
        tree_start.remove_events()

    elif setup_simulation == "tree_topology_known":
        tree_start = tree.get_copy()
        tree_start.remove_samples()
        tree_start.remove_events()
        samples = tree.get_samples_unassigned_copy()
        tree_start.randomly_assign_samples(samples)

    elif setup_simulation == "size_tree_known":
        tree_start = Tree(len(tree.nodes), tree.config)
        samples = tree.get_samples_unassigned_copy()
        tree_start.randomly_assign_samples(samples)

    else:
        # setup_simulation == "size_tree_NOT_known"
        # TODO sample random number of nodes should not be here
        samples = tree.get_samples_unassigned_copy()
        n_samples = len(samples)

        # TODO use random number of nodes at the beginning
        number_nodes = np.random.geometric((1/2)**(n_samples*tree.config["k_n_nodes"]))
        number_nodes = 3

        tree_start = Tree(number_nodes, tree.config)
        tree_start.randomly_assign_samples(samples)

    return tree_start

# SIMULATION
def simulation_mcmc(number_nodes,
               n_reads_sample,
               config,
               move_weights,
               setup_simulation,
               n_cycles,
               n_iter_cycle,
               list_beta,
               tree=None,
               iter_equal_move_accepted = False,
               verbose=1):
    assert(setup_simulation in ["tree_known",
                                "tree_topology_and_sample_assignments_known",
                                "tree_topology_known",
                                "size_tree_known",
                                "size_tree_NOT_known"])

    if tree is None:
        tree = Tree(number_nodes, config)  # generates a tree with random topology
        tree.generate_events()  # generates a tree with random topology
        tree.generate_samples(n_reads_sample)


    moves = get_moves_setup(setup_simulation)
    tree_start = get_random_tree_start(setup_simulation, tree)

    print("tree simulation, post", tree.get_log_posterior())
    print(tree)
    print()

    print("tree start, post", tree_start.get_log_posterior())
    print(tree_start)
    print()

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
                         verbose=verbose)

    best_post = list_trees[0].get_log_posterior(update=False)
    best_tree = list_trees[0]

    for i, t in enumerate(list_trees):
        post = t.get_log_posterior(update=False)
        print("Posterior distributions")
        print(i, post)
        if post > best_post:
            best_post = post
            best_tree = t
    print("tree_start")
    print(tree_start)
    print()
    print("tree simulation, post", tree.get_log_posterior())
    print(tree)
    print()
    print("best_tree, post:", best_post)
    print(best_tree)

def simulation_asmc(number_nodes,
                    n_reads_sample,
                    config,
                    move_weights,
                    setup_simulation,
                    n_particles,
                    n_iter,
                    epsilon_ress,
                    alpha,
                    tree=None,
                    verbose=1):

    assert (setup_simulation in ["tree_known",
                                 "tree_topology_and_sample_assignments_known",
                                 "tree_topology_known",
                                 "size_tree_known",
                                 "size_tree_NOT_known"])

    if tree is None:
        tree = Tree(number_nodes, config)  # generates a tree with random topology
        tree.generate_events()  # generates a tree with random topology
        tree.generate_samples(n_reads_sample)

    moves = get_moves_setup(setup_simulation)
    executor = get_reusable_executor(max_workers=MAX_WORKERS)
    trees_start = list(executor.map(lambda p: get_random_tree_start(*p), [(setup_simulation, tree)]*n_particles, chunksize=round(n_particles / 10)))

    (weights,trees) = annealed_smc(moves,
                                   move_weights,
                                   n_particles,
                                   n_iter,
                                   trees_start,
                                   epsilon_ress,
                                   alpha,
                                   verbose=verbose)

    best_post = float('-inf')
    best_tree = None
    for t in trees[-1]:
        post = t.get_log_posterior(update=False)
        if post > best_post:
            best_post = post
            best_tree = t
    print('tree simulation and its posterior', tree.get_log_posterior())
    print(tree)
    print('best tree inference and its posterior', best_post)
    print(best_tree)
    print("weights")
    print(weights[-1])


def main():
    verbose = 0

    path_tree_load = "trees_test/tree_test_4.json"

    setup_simulation = "size_tree_NOT_known"
    # setup_simulation = "size_tree_known"
    # setup_simulation = "tree_topology_known"
    # setup_simulation = "tree_topology_and_sample_assignments_known"
    # setup_simulation = "tree_known"

    inference_algorithm = "mcmc" # must be asmc or mcmc

    #if random tree simulation, its parameters are
    number_nodes = 7
    number_samples = 20
    n_reads_sample = 10000 * np.ones(number_samples)
    number_segments = 4
    config = {'number_segments': number_segments,
              'p_new_event': 0.1,
              'length_segments':  np.ones(number_segments),
              "k_n_nodes": 1}

    move_weights = {"swap_events_2_nodes": 1, "add_event": 1, "remove_event": 1, "remove_then_add_event": 1,
                    "modify_sample_attachments": 1, "prune_and_reattach": 1, "split_node": 1, "merge_nodes": 1,
                    "modify_event":1}

    # MC3 parameters
    n_cycles = 200
    n_iter_cycle = 100
    #list_beta = [1,1/2,1/3,1/4]
    list_beta = [1,1,1/2,1/2,1/4,1/4,1,1,1/2,1/2,1/4,1/4]
    #list_beta = [1, 1 / 1.5, 1 / 2, 1 / 2.5, 1/3, 1/3.5, 1/4,1/4.5,1/5, 1/5.5]
    #list_beta = [1]
    iter_equal_move_accepted = True

    # asmc parameters
    n_particles = 1000
    n_iter = 10
    epsilon_ress = 0.75
    beta = 3
    alpha = 1-10**(-beta)

    tree = Tree(config=config, path_tree_load=path_tree_load)
    number_nodes = len(tree.nodes)

    if inference_algorithm == 'mcmc':
        simulation_mcmc(number_nodes,
                   n_reads_sample,
                   config,
                   move_weights,
                   setup_simulation,
                   n_cycles,
                   n_iter_cycle,
                   list_beta,
                   tree=tree,
                   iter_equal_move_accepted=iter_equal_move_accepted,
                   verbose=verbose)
    else:
        simulation_asmc(number_nodes,
                        n_reads_sample,
                        config,
                        move_weights,
                        setup_simulation,
                        n_particles,
                        n_iter,
                        epsilon_ress,
                        alpha,
                        tree=tree)
if __name__ == "__main__":
    main()

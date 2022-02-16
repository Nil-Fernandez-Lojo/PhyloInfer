from tree.tree import Tree
from inference.moves import move, p_add_event
import numpy as np
import copy
import math
from tree.node import get_regions_available_profile

# TODO: use random number generator of numpy instead!
np.random.seed(2020)


# TODO: True setup

# SIDE FUNCTIONS
def compute_log_proposal_ratio(move_type,
                               move_weights,
                               info,
                               tree,
                               new_tree):
    if move_type in ["swap_events_2_nodes", "modify_sample_attachments", "prune_and_reattach"]:
        log_proposal_ratio = 0
    else:
        event = info["event"]
        for j in range(len(tree.nodes)):
            if tree.nodes[j].id_ == info["node.id_"]:
                if move_type == "add_event":
                    node_to_add_event = tree.nodes[j]
                else:
                    node_to_add_event = new_tree.nodes[j]

                break
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
                                   beta):
    if not info["success"]:
        p_accept = 0
    else:
        if move_type == "modify_sample_attachments":
            log_prior_ratio = 0
        else:
            # TODO: for prune_and_reattach it is either 0 or -inf, we do not have to recompute everything
            log_prior_ratio = new_tree.get_log_prior() - tree.get_log_prior()
        # print("log_prior_ratio",log_prior_ratio)
        # print("WTF","new_tree.get_log_prior()-tree.get_log_prior()",new_tree.get_log_prior()-tree.get_log_prior())
        # print("move_type",move_type)
        # print("log_prior_ratio", log_prior_ratio)
        # print("tree")
        # print(tree)
        # print("new_tree")
        # print(new_tree)
        if math.isinf(log_prior_ratio):
            return 0

        log_proposal_ratio = compute_log_proposal_ratio(move_type,
                                                        move_weights,
                                                        info,
                                                        tree,
                                                        new_tree)

        # print("new_tree")
        # print(new_tree)
        # print("new_tree.get_log_prior()",new_tree.get_log_prior(),"tree.get_log_prior()",tree.get_log_prior())
        # print("new_tree.get_log_prior()-tree.get_log_prior()",new_tree.get_log_prior()-tree.get_log_prior())
        # print("log_prior_ratio",log_prior_ratio)
        log_likelihood_ratio = new_tree.get_log_likelihood() - tree.get_log_likelihood()
        p_accept = min(1, np.exp(beta * (log_prior_ratio + log_likelihood_ratio) + log_proposal_ratio))

    return p_accept


# SINGLE CHAIN MCMC
def mcmc(tree_start, moves, move_weights, n_samples, max_proposed_moves=10 ** 6, beta=1):
    list_trees = [copy.deepcopy(tree_start)]
    tree = tree_start
    i = 0
    while (i < max_proposed_moves) and len(list_trees) < n_samples:
        # Get moves_allowed at this iteration
        moves_allowed = []
        if "add_event" in moves:
            moves_allowed.append("add_event")
        if "swap_events_2_nodes" in moves:
            moves_allowed.append(
                "swap_events_2_nodes")  # TODO: we should not allow to swap 2 nodes with the same events
        if "modify_sample_attachments" in moves:
            moves_allowed.append("modify_sample_attachments")
        if "prune_and_reattach" in moves:
            moves_allowed.append("prune_and_reattach")
        if tree.get_number_events() != 0:
            if "remove_event" in moves:
                moves_allowed.append("remove_event")
            if "remove_then_add_event" in moves:
                moves_allowed.append("remove_then_add_event")

        # Select move type
        p = [move_weights[mv] for mv in moves_allowed]
        p = np.array(p) / np.sum(p)
        move_type = np.random.choice(moves_allowed, p=p)

        # Do move
        new_tree, info = move(tree, move_type)

        # Compute probability of accepting move
        p_accept = compute_probability_acceptance(move_type,
                                                  move_weights,
                                                  info,
                                                  tree,
                                                  new_tree,
                                                  beta)

        # Decision to accept it
        if np.random.choice(2, p=[1 - p_accept, p_accept]):
            list_trees.append(copy.deepcopy(new_tree))
            tree = new_tree
        # print("n_samples",  len(list_trees), "out of", n_samples)
        i += 1
    return list_trees


# MC3
def mc3(moves, tree_start, move_weights, n_iter_cycle, n_cycles, list_beta):
    # TODO: change definition n_iter_cycle such that number of accepted moves and not number of proposed moves

    list_trees = dict()
    tree_start_chain = dict()
    for beta in list_beta:
        list_trees[beta] = []
        tree_start_chain[beta] = tree_start

    for i in range(n_cycles):
        print("cycle:", i)
        for beta in list_beta:
            print("beta:", beta, "tree start")
            print(tree_start_chain[beta])
            list_trees_cycle = mcmc(tree_start_chain[beta],
                                    moves,
                                    move_weights,
                                    n_iter_cycle,
                                    beta=beta)
            list_trees[beta].extend(copy.deepcopy(list_trees_cycle))
            tree_start_chain[beta] = copy.deepcopy(list_trees_cycle[-1])

        # choose 2 chains at random
        (beta_1, beta_2) = np.random.choice(list_beta, size=2, replace=False)
        print("beta_1", beta_1, "beta_2", beta_2)
        p_exchange = min(1, np.exp((beta_1 - beta_2) * (
                tree_start_chain[beta_2].get_log_posterior() - tree_start_chain[beta_1].get_log_posterior())))
        print("p_exchange", p_exchange)
        print("posterior 1", tree_start_chain[beta_1].get_log_posterior())
        print("posterior 2", tree_start_chain[beta_2].get_log_posterior())
        print('tree_1')
        print(tree_start_chain[beta_1])
        print('tree_2')
        print(tree_start_chain[beta_2])

        temp = tree_start_chain[beta_1]
        tree_start_chain[beta_1] = tree_start_chain[beta_2]
        tree_start_chain[beta_2] = temp
    return list_trees[1]


# SIMULATION
def simulation(number_nodes,
               n_reads_sample,
               config,
               move_weights,
               setup_simulation,
               n_cycles,
               n_iter_cycle,
               list_beta):
    if setup_simulation == "tree_known":
        tree = Tree(number_nodes, config)  # generates a tree with random topology
        tree.generate_events()  # generates a tree with random topology
        tree_start = copy.deepcopy(tree)
        # Assigns randomly a sample to a node and samples the reads per segment from the copy number profile
        tree.generate_samples(
            n_reads_sample)
        read_counts = []
        for node in tree.nodes:
            for sample in node.samples:
                read_counts.append(sample.read_count)
        tree_start.randomly_assign_samples(read_counts)

        moves = ["modify_sample_attachments"]

    elif setup_simulation == "tree_topology_and_sample_assignments_known":
        tree = Tree(number_nodes, config)  # generates a tree with random topology
        tree.generate_events()  # generates a tree with random topology
        tree.generate_samples(n_reads_sample)

        tree_start = copy.deepcopy(tree)
        for node in tree_start.nodes:
            node.events = []
        tree_start._update_profiles()
        moves = ["add_event", "remove_event"]

    elif setup_simulation == "tree_topology_known":
        tree = Tree(number_nodes, config)  # generates a tree with random topology
        tree_start = copy.deepcopy(tree)
        tree.generate_events()  # generates a tree with random topology
        tree.generate_samples(n_reads_sample)

        read_counts = []
        for node in tree.nodes:
            for sample in node.samples:
                read_counts.append(sample.read_count)
        tree_start.randomly_assign_samples(read_counts)

        moves = ["modify_sample_attachments", "add_event", "remove_event"]

    elif setup_simulation == "size_tree_known":
        tree = Tree(number_nodes, config)  # generates a tree with random topology
        tree.generate_events()  # generates a tree with random topology
        tree.generate_samples(n_reads_sample)
        tree_start = Tree(number_nodes, config)

        read_counts = []
        for node in tree.nodes:
            for sample in node.samples:
                read_counts.append(sample.read_count)
        tree_start.randomly_assign_samples(read_counts)

        moves = ["modify_sample_attachments", "add_event", "remove_event", "prune_and_reattach"]

    print("tree simulation, post", tree.get_log_posterior())
    print(tree)
    print()

    if len(list_beta) == 1:
        list_trees = mcmc(tree_start,
                          moves,
                          move_weights,
                          n_iter_cycle)

    else:
        list_trees = mc3(moves, tree_start, move_weights, n_iter_cycle, n_cycles, list_beta)
    for j, t in enumerate(list_trees):
        correct_tree = True
        for i in range(len(tree_start.nodes)):
            if np.any((tree.nodes[i].get_profile() - t.nodes[i].get_profile()) != 0):
                correct_tree = False
                break
        if correct_tree:
            print("got original tree at MCMC sample", j)
    best_post = list_trees[0].get_log_posterior()
    best_tree = list_trees[0]

    for i, t in enumerate(list_trees):
        post = t.get_log_posterior()
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


def main():
    number_nodes = 5
    number_samples = 40
    n_reads_sample = 10000 * np.ones(number_samples)
    config = {'number_segments': 5, 'p_new_event': 0.6}
    config['length_segments'] = np.ones(config['number_segments'])

    move_weights = {"swap_events_2_nodes": 1, "add_event": 1, "remove_event": 1, "remove_then_add_event": 1,
                    "modify_sample_attachments": 1, "prune_and_reattach": 1}

    # MC3 parameters
    n_cycles = 5
    n_iter_cycle = 1000
    # list_beta = [1,1/2]
    # list_beta = [1, 1 / 1.5, 1 / 2, 1 / 2.5]
    list_beta = [1]

    setup_simulation = "size_tree_known"
    # setup_simulation = "tree_topology_known"
    # setup_simulation = "tree_topology_and_sample_assignments_known"
    # setup_simulation = "tree_known"

    simulation(number_nodes,
               n_reads_sample,
               config,
               move_weights,
               setup_simulation,
               n_cycles,
               n_iter_cycle,
               list_beta)


main()

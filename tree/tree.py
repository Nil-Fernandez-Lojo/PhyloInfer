import numpy as np
import math
from tree.node import Node
from tree.sample import Sample
import json
import copy


# TODO: for now 1 diploid chromosome, need to change that

class Tree:
    """
    Class used to represent a phylogeny

    ...

    Attributes
    ----------
    nodes: list of objects of the class node
        list of nodes
    config: dict
        dictionary containing configuration parameters
    root: Node
        node a the root

    Methods
    -------
    __init__:
        Initialises a (labeled) tree with random topology.
        Parameters (other than self):
            number_nodes: attribute number_nodes
            config: attribute config

    generate_events:
        Samples random CN events at each node (in a single node they cannot be overlapping)
    generate_samples:
        Creates config['n_samples'] samples, assigns them uniformly at random to a node
        then samples the distribution of the coverage of each segment from the CN profile
        of the assigned node.
    get_samples:
        returns the list of samples
    get_log_prior:
        returns the log of the prior of the phylogeny, from the contribution:
        prior tree size
        prior tree topology
        prior events
        prior sample assignation
    get_log_likelihood:
        returns the log of the likelihood of the segment coverage given the CN profile
    get_log_posterior:
        returns the unnormalised log posterior (i.e. up to an additive constant).
        It actually returns the log of the joint distribution
    __str__:
        returns a string representation of the tree
    """

    def __init__(self, number_nodes=None, config=None, path_tree_load=None, random_init=True):
        self.config = config
        self.nodes = []
        self.root = None
        self.samples = []
        if path_tree_load is not None:
            self.load_tree(path_tree_load)
        elif random_init:
            self.random_init(number_nodes)

        self.log_prior = None
        self.log_likelihood = None
        if (random_init) or (path_tree_load is not None):
            self.update_log_likelihood()
            self.update_log_prior()
        self.node_max_id = len(self.nodes)

    def load_tree(self, path_tree_load):
        # TODO change code such that it loads the whole tree
        # TODO check that the tree is valid

        with open(path_tree_load) as json_file:
            tree_json = json.load(json_file)

        # TODO: p_new_event should not be in json file...
        self.config = {'number_segments' : len(tree_json["length_segments"]),
                       'length_segments' : tree_json["length_segments"],
                       'p_new_event': tree_json["p_new_event"],
                       "k_n_nodes": tree_json["k_n_nodes"]}

        self._add_node(tree_json["tree"][0], None)
        self.root = self.nodes[0]

    def _add_node(self,node_dict,parent):
        node = Node(node_dict["id"],
                         self.config,
                         parent=parent,
                         profile=np.array(node_dict["profile"]))

        if parent is not None:
            parent.children.append(node)

        for s in node_dict["samples"]:
            sample = Sample(s["id"], self.config, read_count=s["read_count"])
            node.add_sample(sample)
            self.samples.append(sample)

        self.nodes.append(node)
        for child_dict in node_dict["children"]:
            self._add_node(child_dict,node)

    def random_init(self, number_nodes):
        self.nodes = [Node(i, self.config) for i in range(number_nodes)]

        if number_nodes == 0:
            self.root = None
            return

        self.root = self.nodes[np.random.randint(number_nodes)]
        if number_nodes == 1:
            return
        elif number_nodes == 2:
            self._add_directed_edge((0, 1))
            return
        else:
            prufer = np.random.randint(0, high=number_nodes, size=number_nodes - 2)
            edges = list(decode_prufer(prufer))
            connected_nodes = {self.root.id_}
            while len(edges) > 0:
                for i, edge in enumerate(edges):
                    if edge[0] in connected_nodes:
                        self._add_directed_edge(edge)
                        connected_nodes.add(edge[1])
                        del edges[i]
                        break

                    elif edge[1] in connected_nodes:
                        self._add_directed_edge((edge[1], edge[0]))
                        connected_nodes.add(edge[0])
                        del edges[i]
                        break

    def _add_directed_edge(self, edge):
        self.nodes[edge[0]].add_child(self.nodes[edge[1]])

    def generate_events(self):
        self._dfs(self.root, 'sample_events')
        self.update_log_prior()
        self.update_log_prior()

    def remove_events(self):
        for node in self.nodes:
            node.events = []
        self._update_profiles()
        self.update_log_prior()
        self.update_log_prior()

    def update_events(self, node=None):
        if node is None:
            self._dfs(self.root, 'update_events')
        else:
            self._dfs(node, 'update_events')
        self.update_log_prior()

    def generate_samples(self, n_reads_sample):
        for i in range(len(n_reads_sample)):
            node = self.nodes[np.random.randint(len(self.nodes))]
            sample = Sample(i, self.config)
            node.add_sample(sample)
            sample.generate_read_counts_from_cn(n_reads_sample[i])
            self.samples.append(sample)
        self.update_log_prior()
        self.update_log_likelihood()



    def randomly_assign_samples(self, samples):
        for sample in samples:
            permutation = np.random.permutation(len(self.nodes))
            for j in range(len(self.nodes)):
                node = self.nodes[permutation[j]]
                profile = node.get_profile()
                if not np.any((profile == 0) & (sample.read_count != 0)):
                    break
            node.add_sample(sample)
            self.samples.append(sample)
        self.update_log_prior()
        self.update_log_likelihood()

    def remove_samples(self):
        for node in self.nodes:
            node.samples = []
        self.samples = []
        self.update_log_prior()
        self.update_log_likelihood()

    def get_samples_unassigned_copy(self,copy_log_likelihood=False):
        return [sample.get_copy_unassigned(copy_log_likelihood) for sample in self.samples]

    def get_number_samples(self):
        return len(self.samples)

    def get_number_events(self):
        n = 0
        for node in self.nodes:
            n += len(node.events)
        return n

    def update_log_prior(self,root_nodes_to_update = "root"):
        # TODO:  allow to just update 1 term of log prior
        n_samples = self.get_number_samples()
        tree_size_term = -np.log(2) * n_samples * len(self.nodes)*self.config["k_n_nodes"]  # TODO: Do we really want this?
        #tree_size_term = -np.log(2) * len(self.nodes)*self.config["k_n_nodes"]
        tree_topology_term = - (len(self.nodes) - 1) * np.log(len(self.nodes))

        if root_nodes_to_update is not None:
            if root_nodes_to_update == "root":
                root_nodes_to_update = [self.root]
            for node in root_nodes_to_update:
                # TODO: this assumes that the CN profiles of each node are updated, we should probably change this
                self._dfs(node, 'update_log_prior_events')

        # prior events
        events_term = 0
        for node in self.nodes:
            log_prior_node = node.get_log_prior_events()
            if math.isinf(log_prior_node):
                self.log_prior =  float('-inf')
                return
            else:
                events_term += log_prior_node

        sample_assignation_term = - n_samples * np.log(len(self.nodes))
        self.log_prior = tree_size_term + tree_topology_term + events_term + sample_assignation_term

    def get_log_prior(self,root_nodes_to_update="root"):
        if root_nodes_to_update is not None:
            self.update_log_prior(root_nodes_to_update)
        return self.log_prior

    def update_log_likelihood(self, root_nodes_to_update="root"):
        if root_nodes_to_update == "root":
            root_nodes_to_update = [self.root]

        for node in root_nodes_to_update:
            self._dfs(node, 'update_log_likelihood_samples')

        self.log_likelihood = 0
        for sample in self.samples:
            self.log_likelihood += sample.get_log_likelihood()

    def get_log_likelihood(self, root_nodes_to_update="root"):
        if root_nodes_to_update is not None:
            self.update_log_likelihood(root_nodes_to_update)
        return self.log_likelihood

    def get_log_posterior(self,update=True):
        # TODO: we shoudl be able to use root_nodes_to_update
        # Unnormalised (i.e. up to an additive constant)
        if update:
            self.update_log_prior()
            self.update_log_likelihood()
        return self.get_log_prior(root_nodes_to_update=None) + self.get_log_likelihood(root_nodes_to_update=None)

    def _dfs(self, node, method_name):
        method = getattr(node, method_name)
        method()
        for child in node.children:
            self._dfs(child, method_name)

    def get_children_id(self, node):
        children_id = []
        for child in node.children:
            children_id.append(child.id_)
            children_id.extend(self.get_children_id(child))
        return children_id

    def _update_profiles(self, node="root"):
        if node == "root":
            node = self.root
        self._dfs(node, 'update_profile')

    def get_node_from_id(self,id_):
        # TODO: this should be used in multiple places in the code
        for node in self.nodes:
            if node.id_ == id_:
                return node

    def dfs_str(self,depth, node, string,add_samples=True):
        string += depth * '  ' + '-id: ' + str(node.id_) + ' CN: ' + str(node.get_profile()) + '\n'
        if add_samples:
            for sample in node.samples:
                string += (depth + 1) * '  ' + 'sample: ' + str(sample.read_count) + '\n'
        else:
            string += (depth + 1) * '  ' + ' Number of samples: ' + str(len(node.samples))+ '\n'
        for child in node.children:
            string = self.dfs_str(depth + 1, child, string,add_samples=add_samples)
        return string

    def print(self,add_samples=True):
        self._update_profiles()  # TODO remove
        print(self.dfs_str(0, self.root, '',add_samples=add_samples))

    def to_file(self,path, mode,iteration,add_samples=False):
        with open(path, mode) as f:
            f.write("cycle: "+str(iteration)+" posterior: "+ str(self.get_log_posterior(update=False)))
            f.write(self.dfs_str(0, self.root, '',add_samples=add_samples))
            f.write('\n')

    def _copy_nodes(self, tree_copy,parent_node_copy,current_node):
        node_copy = Node(current_node.id_, self.config)
        node_copy.events = copy.deepcopy(current_node.events)
        node_copy.profile = copy.deepcopy(current_node.profile)
        node_copy.log_prior = current_node.log_prior
        node_copy.p_read =  copy.deepcopy(current_node.p_read)
        if parent_node_copy is not None:
            parent_node_copy.add_child(node_copy)
        tree_copy.nodes.append(node_copy)

        for sample in current_node.samples:
            sample_copy = sample.get_copy_unassigned(copy_log_likelihood=True)
            node_copy.add_sample(sample_copy,update_log_likelihood=False)

        for child in current_node.children:
            self._copy_nodes(tree_copy, node_copy, child)

    def get_copy(self):
        tree_copy = Tree(number_nodes=None, config=self.config, path_tree_load=None, random_init=False)
        self._copy_nodes(tree_copy,None,self.root)
        tree_copy.root = tree_copy.nodes[0]
        tree_copy.node_max_id = self.node_max_id
        tree_copy.log_prior = self.log_prior
        tree_copy.log_likelihood =  self.log_likelihood
        for node in tree_copy.nodes:
            tree_copy.samples.extend(node.samples)
        return tree_copy

    def __str__(self):
        self._update_profiles()  # TODO remove
        return self.dfs_str(0, self.root, '')


def decode_prufer(p):
    """
    Generative function that coverts iteratively a prufer sequence into a list of directed edges
    To get the whole list at once call list(decode_prufer(p))

    ...

    Parameters
    ---------
    p : list of ints
        prufer sequence
    """
    p = list(p)
    vertices = set(range(len(p) + 2))
    for (i, u) in enumerate(p):
        v = min(vertices.difference(p[i:]))
        vertices.remove(v)
        yield u, v
    yield tuple(vertices)

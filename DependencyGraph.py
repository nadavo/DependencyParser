from collections import defaultdict

from global_consts import ROOT


class DependencyGraph:
    def __init__(self, sentence_id, nodes, successors):
        self.sentence_id = sentence_id
        # node_index: node object
        self.nodes_dict = nodes
        # node_index: [indices of children nodes]
        self.successors_dict = successors
        self.__decodeParents__()
        self.__decodeChildren__()
        # (parent_index, child_index): edge obj
        self.edges_dict = {}
        self.generated_edges_dict = {}
        self.__make_edges__()
        # node_index: [indices of all nodes (except ROOT)]
        self.full_successors_dict = defaultdict(list)
        self.__generate_full_successors__()
        self.__register_nodes()

    def __make_edges__(self):
        for node in self.nodes_dict.values():
            if node.get_word() == ROOT:
                continue
            self.edges_dict[node.initial_parent_index, node.get_index()] = Edge(node.initial_parent, node)

    def get_nodes(self):
        return tuple(self.nodes_dict.values())

    def get_edges_set(self):
        return set(self.edges_dict.values())

    def get_edges(self):
        ordered_edges_keys = sorted(self.edges_dict.keys(), key=lambda x: x[1])
        ordered_edges = [self.edges_dict.get(key) for key in ordered_edges_keys]
        return tuple(ordered_edges)

    def get_successors(self):
        return self.successors_dict

    def __decodeChildren__(self):
        for node in self.get_nodes():
            children_indices = self.successors_dict.get(node.get_index())
            if children_indices is None:
                node.children = []
                continue
            children = []
            for index in children_indices:
                children.append(self.nodes_dict.get(index))
            node.children = children

    def __decodeParents__(self):
        for node in self.get_nodes():
            if node.get_pos_tag() == ROOT:
                continue
            node.initial_parent = self.nodes_dict.get(node.initial_parent_index)

    def get_sentence_length(self):
        """ :returns int number of words in the sentence"""
        return len(self.nodes_dict) - 1

    def get_sentence_graph_size(self):
        """ :returns int number of nodes in the graph (including ROOT) """
        return len(self.nodes_dict)

    def get_sentence_indices(self):
        """ :returns (1, ..., n)"""
        return tuple(range(1, self.get_sentence_graph_size()))

    def __generate_full_successors__(self):
        full_graph = defaultdict(list)
        sentence_indices = self.get_sentence_indices()
        for i in range(self.get_sentence_graph_size()):
            full_graph[i] = list(filter(lambda x: x != i, sentence_indices))
        self.full_successors_dict = full_graph

    def get_full_successors(self):
        """ :returns dictionary of all the nodes indices in the graph as keys, each has a value of a list
        of all other nodes indices in the graph except ROOTINDEX (ROOTINDEX appears only as a key)"""
        return self.full_successors_dict

    def get_node(self, index):
        return self.nodes_dict.get(index, None)

    def get_edge(self, source_index, target_index):
        """returns the edge object for an edge key (parent_index, child_index)"""
        key = (source_index, target_index)
        edge = self.edges_dict.get(key, None)
        if edge is None:
            edge = self.generated_edges_dict.get(key)
            if edge is None:
                edge = Edge(self.get_node(key[0]), self.get_node(key[1]))
                self.generated_edges_dict[key] = edge
        return edge

    def __register_nodes(self):
        for node in self.get_nodes():
            node.get_node = self.get_node


class Node:
    def __init__(self, word, index, pos_tag, parent_index=None, parent=None, get_node=None):
        """
        :param word: string
        :param index: int
        :param pos_tag: string
        :param parent_index: int
        :param parent: Node object reference
        """
        self.word = word
        self.pos_tag = pos_tag
        self.index = index
        self.initial_parent_index = parent_index
        self.initial_parent = parent
        self.children = set()
        self.get_node = get_node

    def get_index(self):
        return self.index

    def get_word(self):
        return self.word

    def get_pos_tag(self):
        return self.pos_tag

    def get_neighbor(self, offset):
        return self.get_node(self.index + offset)


class Edge:
    def __init__(self, source_node, target_node, label=None):
        self.source = source_node
        self.target = target_node
        self.label = label
        self.score = 0
        self.key = (source_node.get_index(), target_node.get_index())

    def get_label(self):
        return self.label

    def get_score(self):
        return self.score

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def get_key(self):
        """:returns tuple (source node, target node) """
        return self.source, self.target

    def get_indices_key(self):
        """:returns tuple (source node, target node) """
        return self.source.get_index(), self.target.get_index()


class FullGraph:
    def __init__(self, sentence_id, nodes):
        self.sentence_id = sentence_id
        # node_index: node object
        self.nodes_dict = nodes
        # (parent_index, child_index): edge obj
        self.edges_dict = {}
        self.generated_edges_dict = {}
        self.__make_edges__()
        # node_index: [indices of all nodes (except ROOT)]
        self.full_successors_dict = defaultdict(list)
        self.__generate_full_successors__()
        self.__register_nodes()

    def __make_edges__(self):
        nodes = self.nodes_dict.values()
        for node1 in nodes:
            for node2 in list(filter(lambda x: x != node1, nodes)):
                if node2.get_word() == ROOT:
                    continue
                self.edges_dict[node1.get_index(), node2.get_index()] = Edge(node1, node2)

    def get_nodes(self):
        return tuple(self.nodes_dict.values())

    def get_edges_set(self):
        return set(self.edges_dict.values())

    def get_edges(self):
        ordered_edges_keys = sorted(self.edges_dict.keys(), key=lambda x: x[1])
        ordered_edges = [self.edges_dict.get(key) for key in ordered_edges_keys]
        return tuple(ordered_edges)

    def __decodeParents__(self):
        for node in self.get_nodes():
            if node.get_pos_tag() == ROOT:
                continue
            node.initial_parent = self.nodes_dict.get(node.initial_parent_index)

    def get_sentence_length(self):
        """ :returns int number of words in the sentence"""
        return len(self.nodes_dict) - 1

    def get_sentence_graph_size(self):
        """ :returns int number of nodes in the graph (including ROOT) """
        return len(self.nodes_dict)

    def get_sentence_indices(self):
        """ :returns (1, ..., n)"""
        return tuple(range(1, self.get_sentence_graph_size()))

    def __generate_full_successors__(self):
        full_graph = defaultdict(list)
        sentence_indices = self.get_sentence_indices()
        for i in range(self.get_sentence_graph_size()):
            full_graph[i] = list(filter(lambda x: x != i, sentence_indices))
        self.full_successors_dict = full_graph

    def get_full_successors(self):
        """ :returns dictionary of all the nodes indices in the graph as keys, each has a value of a list
        of all other nodes indices in the graph except ROOTINDEX (ROOTINDEX appears only as a key)"""
        return self.full_successors_dict

    def get_node(self, index):
        return self.nodes_dict.get(index, None)

    def get_edge(self, source_index, target_index):
        """returns the edge object for an edge key (parent_index, child_index)"""
        key = (source_index, target_index)
        edge = self.edges_dict.get(key, None)
        if edge is None:
            edge = self.generated_edges_dict.get(key)
            if edge is None:
                edge = Edge(self.get_node(key[0]), self.get_node(key[1]))
                self.generated_edges_dict[key] = edge
        return edge

    def __register_nodes(self):
        for node in self.get_nodes():
            node.get_node = self.get_node


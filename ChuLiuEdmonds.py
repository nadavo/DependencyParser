from chu_liu import Digraph


class ChuLiuEdmonds:
    def __init__(self, graph, features, weights):
        self.graph = graph
        self.features = features
        self.weights = weights
        # successors dict, edges set
        self.result_successors, self.result_edges = self.__predict__()

    @staticmethod
    def calc_dot_product(features_indices, weights):
        """function to calculate dot product between feature and weights vectors
         by summing up values of feature indices in weights vector"""
        total = 0.0
        for index in features_indices:
            total += weights[index]
        return total

    def __generate_edges_set__(self, successors):
        edges = set()
        for source, targets in successors.items():
            for target in targets:
                edges.add(self.graph.get_edge(source, target))
        return edges

    def __predict__(self):
        MST = Digraph(self.graph.get_full_successors(), self.get_score).mst()
        return MST.successors, self.__generate_edges_set__(MST.successors)

    def get_score(self, head_index, modifier_index):
        edge = self.graph.get_edge(head_index, modifier_index)
        return self.calc_dot_product(self.features.get_features(edge), self.weights)

    def get_result_successors(self):
        return self.result_successors

    def get_result_edges(self):
        return self.result_edges

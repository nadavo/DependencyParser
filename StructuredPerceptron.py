from utils import Timer
from ChuLiuEdmonds import ChuLiuEdmonds
from numpy import abs, average, zeros, full
from collections import defaultdict


class StructuredPerceptron:
    def __init__(self, data, features, N=20, evaluate=None):
        self.data = data
        self.features = features
        self.N = N
        self.T = self.data.get_num_sentences()
        self.weights = full(self.features.getFeaturesVectorLength(), fill_value=0.1, dtype=float)
        self.w_avg = zeros(self.features.getFeaturesVectorLength(), dtype=float)
        self.loss_list = []
        self.evaluate = evaluate
        self.accuracy_list = []
        self.__run__()

    @staticmethod
    def is_inference_incorrect(inferred, true):
        ## Set objects implementation
        return len(true - inferred) != 0

    def calc_inference_diff(self, inferred, true):
        # Sets implementation
        # index: diff value
        diff_dict = defaultdict(float)
        diff_set_pos = true.difference(inferred)
        diff_set_neg = inferred.difference(true)
        for edge in diff_set_pos:
            for index in self.features.get_features(edge):
                diff_dict[index] += 1.0
        for edge in diff_set_neg:
            for index in self.features.get_features(edge):
                diff_dict[index] -= 1.0
        return diff_dict

    def calc_loss(self, diff_dict):
        """function to calculate the loss value over for the given features diff"""
        loss = 0.0
        for value in diff_dict.values():
            loss += abs(value)
        loss /= self.T
        return loss

    def update_weights(self, diff_dict):
        for index, update in diff_dict.items():
            self.weights[index] += update

    def __run__(self):
        main_timer = Timer("Structured Perceptron "+str(self.N)+" Iterations")
        for i in range(self.N):
            iteration_timer = Timer("Perceptron Iteration "+str(i+1))
            loss = 0
            for graph in self.data.get_sentence_graphs():
                true = graph.get_edges_set()
                maximizer = ChuLiuEdmonds(graph, self.features, self.weights)
                inferred = maximizer.get_result_edges()
                if self.is_inference_incorrect(inferred, true):
                    diff = self.calc_inference_diff(inferred, true)
                    self.update_weights(diff)
                    self.w_avg += self.weights
                    loss += self.calc_loss(diff)
            iteration_timer.stop()
            self.loss_list.append(loss)
            print("Loss:", loss)
            if callable(self.evaluate):
                self.accuracy_list.append(self.evaluate(self.data, self.weights))
        main_timer.stop()
        print("Average Loss:", average(self.get_losses()))
        if callable(self.evaluate):
            print("Average Accuracy:", average(self.get_accuracies()))
        self.w_avg = self.w_avg/(self.N * self.T)

    def get_weights(self):
        return self.weights

    def get_avg_weights(self):
        return self.w_avg

    def get_losses(self):
        return tuple(self.loss_list)

    def get_accuracies(self):
        return tuple(self.accuracy_list)

from collections import defaultdict
import numpy as np
import pickle as pk
from utils import Timer
from tabulate import tabulate
from ChuLiuEdmonds import ChuLiuEdmonds
from StructuredPerceptron import StructuredPerceptron


class DependencyParser:
    """Class to implement an Dependency Parsing model as learnt in lectures and tutorials
        Constructor parameters:
            feature_factory - a BasicFeatures or AdvancedFeatures feature factory object created for TaggedDataReader object of training data
            pretrained_weights - option to create model with pretrained weights from cache (used for quick model evaluation)"""
    def __init__(self, feature_factory, pretrained_weights=None):
        self.feature_factory = feature_factory
        self.data = feature_factory.data
        self.cache = self.get_trained_weights_cache_name()
        self.weights = self.__initializeWeights__(pretrained_weights)
        self.train_results = None
        # data.file: sentence_index: word_index: head_index
        self.predictions = {}
        self.correct_tags = defaultdict(int)
        self.wrong_tags = defaultdict(int)
        self.wrong_tag_pairs = defaultdict(int)
        self.wrong_tags_dicts = {}

    def __initializeWeights__(self, pretrained_weights):
        """method to initialize model weights according to pretrained_weights parameter"""
        weights_vector_length = self.feature_factory.getFeaturesVectorLength()
        weights = np.zeros(weights_vector_length, dtype=float)
        if pretrained_weights is True:
            weights = self.load_trained_weights(self.get_trained_weights_cache_name())
        elif pretrained_weights is not None and type(pretrained_weights) is int:
            weights = self.load_trained_weights(self.get_trained_weights_cache_name(pretrained_weights))
        elif pretrained_weights is not None and type(pretrained_weights) is np.ndarray and len(pretrained_weights) == weights_vector_length:
            weights = pretrained_weights
        return weights

    def get_trained_weights_cache_name(self, num_iter=20):
        """method to retrieve cache file name according to model parameters"""
        return "final/cache/data-{}_numIterations-{}_features-{}_numFeatures-{}_cutoff-{}_trained_weights.pkl".format(self.data.get_num_sentences(), num_iter, self.feature_factory.type, self.feature_factory.getFeaturesVectorLength(), self.feature_factory.getCutoffParameter())

    def load_trained_weights(self, file):
        """method to load pretrained weights from a given cache file"""
        with open(file, 'rb') as cache:
            trained = pk.load(cache)
            weights = trained.get('avg_weights')
        return weights

    def get_weights(self):
        return self.weights

    def fit(self, num_iter=20, evaluate_per_iteration=False, save=True):
        """method used for training the model, by passing the loss, gradient calculation functions
        and an initial weights vector to the L-BFGS-B minimizer function.
        returns training results and final weights vector.
        has the option of saving the trained weights and results to a local cache file (pickle)"""
        timer = Timer("Training")
        if evaluate_per_iteration:
            predict_and_evaluate = self.predict_and_evaluate
        else:
            predict_and_evaluate = None
        estimator = StructuredPerceptron(self.data, self.feature_factory, num_iter, predict_and_evaluate)
        self.train_results = {'weights': estimator.get_weights(), 'avg_weights': estimator.get_avg_weights(), 'losses': estimator.get_losses(), 'accuracies': estimator.get_accuracies()}
        self.weights = estimator.get_avg_weights()
        timer.stop()
        if save:
            with open(self.get_trained_weights_cache_name(num_iter), 'wb') as cache:
                pk.dump(self.train_results, cache)

    def transform_to_word_head(self, edges):
        tags = defaultdict(int)
        for edge in edges:
            tags[edge.target.get_index()] = edge.source.get_index()
        return tags

    def predict_and_evaluate(self, data, weights):
        self.predict(data, weights)
        return self.evaluate(data)[1]

    def predict(self, data, weights=None):
        timer = Timer("Inference on "+str(data.file))
        self.predictions[data.file] = {}
        if weights is None:
            weights = self.weights
        for index, graph in data.sentence_graphs.items():
            prediction = ChuLiuEdmonds(graph, self.feature_factory, weights)
            self.predictions[data.file][index] = self.transform_to_word_head(prediction.get_result_edges())
        timer.stop()

    def evaluate(self, data, verbose=False):
        """method to evaluate the model's predictions vs truth over entire dataset
        by accuracy measure and confusion matrix for top 10 wrong tags.
        must be called only after predict method, otherwise no predictions will be available for evaluation"""
        assert data.get_num_sentences() == len(self.predictions.get(data.file, {})), "Predcitions and truth are not the same length!"
        timer = Timer("Evaluation on "+str(data.file))
        micro_accuracy = 0.0
        accuracies = []
        for i in data.get_sentence_ids():
            truth = data.get_sentence_dependencies(i)
            prediction = self.predictions.get(data.file).get(i, None)
            accuracy, correct = self.accuracy(truth, prediction, verbose)
            accuracies.append(accuracy)
            micro_accuracy += correct
        avg = np.mean(accuracies)
        avg_micro_accuracy = micro_accuracy / sum(data.word_dict.values())
        minimum = np.min(accuracies)
        maximum = np.max(accuracies)
        med = np.median(accuracies)
        print("Results for", data.file)
        print("Total Micro-Average Accuracy: {:.4f}".format(avg_micro_accuracy))
        print("Total Average Accuracy: {:.4f}".format(avg))
        print("Minimal Accuracy: {:.4f}".format(minimum))
        print("Maximal Accuracy: {:.4f}".format(maximum))
        print("Median Accuracy: {:.4f}".format(med))
        # self.confusion_table(data.file)
        # self.confusionMatrix(data.file)
        timer.stop()
        return data.file, avg_micro_accuracy, avg, minimum, maximum, med

    def accuracy(self, truth, predictions, verbose=False):
        """function to calculate accuracy for a given sentence and model predictions
        truth and predictions are both dictionaries {word_index: head_index} for the same sentence"""
        assert len(truth) == len(predictions), "Predcitions and truth are not the same length!"
        correct = 0
        for i in truth.keys():
            true_head = truth.get(i)
            predicted_head = predictions.get(i)
            key = (true_head, i)
            subkey = (predicted_head, i)
            if true_head == predicted_head:
                correct += 1
                self.correct_tags[key] += 1
            else:
                self.wrong_tags[key] += 1
                self.wrong_tag_pairs[(key, subkey)] += 1
                if self.wrong_tags_dicts.get(key, False) is False:
                    self.wrong_tags_dicts[key] = defaultdict(int)
                self.wrong_tags_dicts[key][subkey] += 1
                if verbose:
                    print("Mistake in sentence", i, "(truth, prediction):", key, subkey)
        result = float(correct) / len(truth)
        if verbose:
            print("Accuracy:", result)
        return result, correct

    def confusionMatrix(self, file, n=10):
        """function to produce Confusion Matrix for top n wrong tags in model evaluation
        'tabulate' package is only used for printing in nice table format"""
        top_wrong_tags = sorted(self.wrong_tags, key=self.wrong_tags.get, reverse=True)[:n]
        header = top_wrong_tags
        rows = []
        for truth in top_wrong_tags:
            columns = [truth]
            for prediction in top_wrong_tags:
                if truth == prediction:
                    columns.append(self.correct_tags.get(truth))
                else:
                    columns.append(self.wrong_tag_pairs.get((truth, prediction)))
            rows.append(columns)
        print("Confusion Matrix for " + self.feature_factory.type + " model on " + file + " dataset")
        header.insert(0, "Truth \ Predicted")
        print(tabulate(rows, headers=header))

    def confusion_table(self, file, n=10):
        """function to produce Confusion Table for top n wrong tags in model evaluation
        'tabulate' package is only used for printing in nice table format"""
        top_wrong_tags = sorted(self.wrong_tag_pairs, key=self.wrong_tag_pairs.get, reverse=True)[:n]
        header = ("Correct Tag", "Model's Tag", "Frequency")
        rows = []
        for truth, prediction in tuple(top_wrong_tags):
            freq = self.wrong_tag_pairs.get((truth, prediction))
            rows.append((truth, prediction, freq))
        print("Confusion Table for " + self.feature_factory.type + " model on " + file + " dataset")
        print(tabulate(rows, headers=header))

import numpy as np
from collections import defaultdict
from utils import OrderedCounter
from itertools import combinations_with_replacement
from global_consts import STOP_BEFORE, STOP_AFTER, MIN_DIST, MAX_DIST, SUFFIXES, PREFIXES


class FeaturesFactory:
    """Abstract class to be inherited by Basic and Advanced feature factories,
    which create features for each model respectively.
    contains generic methods and data structures used to create features for each model type.
    data parameter is a DependencyDataReader object instance
    cutoff parameter determines what is the minimum frequency in the data for each feature in the vector,
    in order to trim the vector size and remove very rare feature occurrences.
    In practice, there is no numerical feature vector implemented,
    but features_index dictionary holds the index for each feature instance in the vector"""
    def __init__(self, data, cutoff=0):
        self.data = data
        # list of all graphs generated for sentences in data
        self.graphs = data.get_sentence_graphs()
        # any feature which its frequency <= cutoff will be cut
        self._cutoff = cutoff
        # feature_name: local_feature_instance: freq
        self.features_dicts = {}
        # [(feature_name, local_feature_instance)]
        self.features_list = []
        # (feature_name, local_feature_instance): index
        self._features_vector_dict = defaultdict(int)
        self._features_vector_length = 0
        # numpy vector of counts of all global features (sums of local features)
        self.empirical_counts = []
        # feature_name: freq
        self.feature_freq = defaultdict(int)
        # edge: [feature indices]
        self.edges_dict = {}
        # set of edges which had no feature indices
        self.null_edges_set = set()

    def getCutoffParameter(self):
        return self._cutoff

    def getFeaturesVectorLength(self):
        return self._features_vector_length

    def getFeaturesIndices(self, node):
        """Abstract method - implemented in child classes"""
        pass

    def getFeatureDicts(self):
        """Abstract method - implemented in child classes"""
        pass

    def getFeatureNames(self):
        """Abstract method - implemented in child classes"""
        pass

    def __generate_features_dictionaries__(self):
        """Abstract method - implemented in child classes"""

    def __checkFeatureIndex__(self, index, indexes):
        """function to check if specific feature instance has an index in the features vector"""
        if index is not False:
            indexes.append(index)

    def __return_feature_index__(self, tup):
        """function to return the index of a specific feature instance in the features vector"""
        index = self._features_vector_dict.get(tup, False)
        return index

    def __generate_features_vector__(self):
        """method to populate the features_vector dictionary, get the feature vector size
        and list of features instances which made the cutoff,
        according to feature names and respective DataReader dictionaries"""
        keys = []
        for feature_name in self.getFeatureNames():
            dictionary = self.features_dicts.get(feature_name)
            features = []
            for feature in dictionary.keys():
                if dictionary.get(feature) > self._cutoff:
                    features.append((feature_name, feature))
                    self.feature_freq[feature_name] += 1
            keys.extend(features)
        for i in range(len(keys)):
            self._features_vector_dict[keys[i]] = i
        self.features_list = tuple(keys)
        self._features_vector_length = len(keys)

    def __calc_empirical_counts__(self):
        """method to calculate the empirical counts part in the gradient calculation"""
        self.empirical_counts = np.zeros(self._features_vector_length, dtype=float)
        for feature_name in self.features_dicts.keys():
            for feature, freq in self.features_dicts.get(feature_name).items():
                index = self._features_vector_dict.get((feature_name, feature))
                self.empirical_counts[index] += freq
        assert len(self.empirical_counts) == np.count_nonzero(self.empirical_counts), "0 in empirical counts vector"

    def getEmpiricalCounts(self):
        """method to return the calculated empirical counts vector for gradient calculation"""
        return self.empirical_counts

    def initialize_vector(self):
        self.__generate_features_dictionaries__()
        self.__generate_features_vector__()
        self.__calc_empirical_counts__()


class FullBasicFeatures(FeaturesFactory):
    """Class to implement feature creation for Basic model.
    Inherits from FeatureFactory abstract class"""
    def __init__(self, data, cutoff=0):
        super().__init__(data, cutoff)
        self.type = "basic"

    def __f1__(self, parent_word, parent_tag):
        """Parent node (Word,Tag) pair Feature"""
        return self.__return_feature_index__(("f1", (parent_word, parent_tag)))

    def __f2__(self, parent_word):
        """Parent node Word Feature"""
        return self.__return_feature_index__(("f2", parent_word))

    def __f3__(self, parent_tag):
        """Parent node Tag Feature"""
        return self.__return_feature_index__(("f3", parent_tag))

    def __f4__(self, child_word, child_tag):
        """Child node(Word,Tag) pair Feature"""
        return self.__return_feature_index__(("f4", (child_word, child_tag)))

    def __f5__(self, child_word):
        """Child node Word Feature"""
        return self.__return_feature_index__(("f5", child_word))

    def __f6__(self, child_tag):
        """Child node Tag Feature"""
        return self.__return_feature_index__(("f6", child_tag))

    def __f8__(self, parent_tag, child_word, child_tag):
        """(Parent Tag, Child Word, Child Tag) Feature"""
        return self.__return_feature_index__(("f8", (parent_tag, child_word, child_tag)))

    def __f10__(self, parent_word, parent_tag, child_tag):
        """(Parent Word, Parent Tag, Child Tag) Feature"""
        return self.__return_feature_index__(("f10", (parent_word, parent_tag, child_tag)))

    def __f13__(self, parent_tag, child_tag):
        """(Parent Tag, Child Tag) Feature"""
        return self.__return_feature_index__(("f13", (parent_tag, child_tag)))

    def __generate_features_dictionaries__(self):
        for feature in self.getFeatureNames():
            self.features_dicts[feature] = OrderedCounter()
        for graph in self.graphs:
            for edge in graph.get_edges():
                self.__basic_features_dictionaries__(edge)
        # self.__add_basic_missing_tags__()

    def __basic_features_dictionaries__(self, edge):
        parent = edge.get_source()
        parent_word = parent.get_word()
        parent_tag = parent.get_pos_tag()
        child = edge.get_target()
        child_word = child.get_word()
        child_tag = child.get_pos_tag()
        self.features_dicts["f1"][parent_word, parent_tag] += 1
        self.features_dicts["f2"][parent_word] += 1
        self.features_dicts["f3"][parent_tag] += 1
        self.features_dicts["f4"][child_word, child_tag] += 1
        self.features_dicts["f5"][child_word] += 1
        self.features_dicts["f6"][child_tag] += 1
        self.features_dicts["f8"][parent_tag, child_word, child_tag] += 1
        self.features_dicts["f10"][parent_word, parent_tag, child_tag] += 1
        self.features_dicts["f13"][parent_tag, child_tag] += 1

    def __add_missing_tags__(self, key, all_tags):
        current_tags = set(self.features_dicts[key].keys())
        missing_tags = set(all_tags) - current_tags
        for tag in missing_tags:
            self.features_dicts[key][tag] += 1

    def __add_basic_missing_tags__(self):
        all_tags = self.data.get_pos_tags()
        all_tag_pairs = combinations_with_replacement(all_tags, 2)
        self.__add_missing_tags__("f3", all_tags)
        self.__add_missing_tags__("f6", all_tags)
        self.__add_missing_tags__("f13", all_tag_pairs)

    def get_features(self, edge):
        """method to return list of feature instances indices in features vector
        for a given edge object.
        checks if this edge was seen before for faster returns"""
        edge_key = (edge.get_key())
        if edge_key in self.null_edges_set:
            return []
        feature = self.edges_dict.get(edge_key, None)
        if feature is None:
            feature = self.getFeaturesIndices(edge)
            if len(feature) == 0:
                self.null_edges_set.add(edge_key)
            else:
                self.edges_dict[edge] = feature
        return feature

    def getFeaturesIndices(self, edge):
        """method to return the feature vector indices of the specific feature instances for a given HistoryTuple and tag
        in_data parameter is an optimization for calls which we know were not obeserved in the data, to skip checking f100"""
        indices = []
        parent = edge.get_source()
        parent_word = parent.get_word()
        parent_tag = parent.get_pos_tag()
        child = edge.get_target()
        child_word = child.get_word()
        child_tag = child.get_pos_tag()
        self.__checkFeatureIndex__(self.__f1__(parent_word, parent_tag), indices)
        self.__checkFeatureIndex__(self.__f2__(parent_word), indices)
        self.__checkFeatureIndex__(self.__f3__(parent_tag), indices)
        self.__checkFeatureIndex__(self.__f4__(child_word, child_tag), indices)
        self.__checkFeatureIndex__(self.__f5__(child_word), indices)
        self.__checkFeatureIndex__(self.__f6__(child_tag), indices)
        self.__checkFeatureIndex__(self.__f8__(parent_tag, child_word, child_tag), indices)
        self.__checkFeatureIndex__(self.__f10__(parent_word, parent_tag, child_tag), indices)
        self.__checkFeatureIndex__(self.__f13__(parent_tag, child_tag), indices)
        return indices

    def getFeatureNames(self):
        """method to return feature names for Basic model features creation"""
        return ["f1", "f2", "f3", "f4", "f5", "f6", "f8", "f10", "f13"]


class BasicFeatures(FeaturesFactory):
    """Class to implement feature creation for Basic model.
    Inherits from FeatureFactory abstract class"""
    def __init__(self, data, cutoff=0):
        super().__init__(data, cutoff)
        self.type = "basic"

    def __f1__(self, parent_word, parent_tag):
        """Parent node (Word,Tag) pair Feature"""
        return self.__return_feature_index__(("f1", (parent_word, parent_tag)))

    def __f2__(self, parent_word):
        """Parent node Word Feature"""
        return self.__return_feature_index__(("f2", parent_word))

    def __f3__(self, parent_tag):
        """Parent node Tag Feature"""
        return self.__return_feature_index__(("f3", parent_tag))

    def __f4__(self, child_word, child_tag):
        """Child node(Word,Tag) pair Feature"""
        return self.__return_feature_index__(("f4", (child_word, child_tag)))

    def __f5__(self, child_word):
        """Child node Word Feature"""
        return self.__return_feature_index__(("f5", child_word))

    def __f6__(self, child_tag):
        """Child node Tag Feature"""
        return self.__return_feature_index__(("f6", child_tag))

    def __f8__(self, parent_tag, child_word, child_tag):
        """(Parent Tag, Child Word, Child Tag) Feature"""
        return self.__return_feature_index__(("f8", (parent_tag, child_word, child_tag)))

    def __f10__(self, parent_word, parent_tag, child_tag):
        """(Parent Word, Parent Tag, Child Tag) Feature"""
        return self.__return_feature_index__(("f10", (parent_word, parent_tag, child_tag)))

    def __f13__(self, parent_tag, child_tag):
        """(Parent Tag, Child Tag) Feature"""
        return self.__return_feature_index__(("f13", (parent_tag, child_tag)))

    def __generate_features_dictionaries__(self):
        for feature in self.getFeatureNames():
            self.features_dicts[feature] = OrderedCounter()
        for graph in self.graphs:
            for edge in graph.get_edges():
                self.__basic_features_dictionaries__(edge)
        self.__add_basic_missing_tags__()

    def __basic_features_dictionaries__(self, edge):
        parent = edge.get_source()
        parent_word = parent.get_word()
        parent_tag = parent.get_pos_tag()
        child = edge.get_target()
        child_word = child.get_word()
        child_tag = child.get_pos_tag()
        self.features_dicts["f3"][parent_tag] += 1
        self.features_dicts["f6"][child_tag] += 1
        self.features_dicts["f8"][parent_tag, child_word, child_tag] += 1
        self.features_dicts["f10"][parent_word, parent_tag, child_tag] += 1
        self.features_dicts["f13"][parent_tag, child_tag] += 1

    def __add_missing_tags__(self, key, all_tags):
        current_tags = set(self.features_dicts[key].keys())
        missing_tags = set(all_tags) - current_tags
        for tag in missing_tags:
            self.features_dicts[key][tag] += 1

    def __add_basic_missing_tags__(self):
        all_tags = self.data.get_pos_tags()
        all_tag_pairs = combinations_with_replacement(all_tags, 2)
        self.__add_missing_tags__("f3", all_tags)
        self.__add_missing_tags__("f6", all_tags)
        self.__add_missing_tags__("f13", all_tag_pairs)

    def get_features(self, edge):
        """method to return list of feature instances indices in features vector
        for a given edge object.
        checks if this edge was seen before for faster returns"""
        edge_key = (edge.get_key())
        if edge_key in self.null_edges_set:
            return []
        feature = self.edges_dict.get(edge_key, None)
        if feature is None:
            feature = self.getFeaturesIndices(edge)
            if len(feature) == 0:
                self.null_edges_set.add(edge_key)
            else:
                self.edges_dict[edge] = feature
        return feature

    def getFeaturesIndices(self, edge):
        """method to return the feature vector indices of the specific feature instances for a given HistoryTuple and tag
        in_data parameter is an optimization for calls which we know were not obeserved in the data, to skip checking f100"""
        indices = []
        parent = edge.get_source()
        parent_word = parent.get_word()
        parent_tag = parent.get_pos_tag()
        child = edge.get_target()
        child_word = child.get_word()
        child_tag = child.get_pos_tag()
        self.__checkFeatureIndex__(self.__f3__(parent_tag), indices)
        self.__checkFeatureIndex__(self.__f6__(child_tag), indices)
        self.__checkFeatureIndex__(self.__f8__(parent_tag, child_word, child_tag), indices)
        self.__checkFeatureIndex__(self.__f10__(parent_word, parent_tag, child_tag), indices)
        self.__checkFeatureIndex__(self.__f13__(parent_tag, child_tag), indices)
        return indices

    def getFeatureNames(self):
        """method to return feature names for Basic model features creation"""
        return ["f3", "f6", "f8", "f10", "f13"]


class AdvancedFeatures(BasicFeatures):
    """Class to implement feature creation for Advanced model.
        Inherits from BasicFeatures class"""
    def __init__(self, data, cutoff=0):
        super().__init__(data, cutoff)
        self.type = "advanced"

    def __fDir__(self, parent_index, child_index):
        """Edge Direction Features"""
        return self.__return_feature_index__(("fDir", self.__fDir_internal__(parent_index, child_index)))

    @staticmethod
    def __fDir_internal__(parent_index, child_index):
        return "left" if parent_index > child_index else "right"

    def __fDist__(self, parent_index, child_index):
        """Edge Distance Features"""
        return self.__return_feature_index__(("fDist", self.__fDist_internal__(parent_index, child_index)))

    @staticmethod
    def __fDist_internal__(parent_index, child_index):
        return abs(parent_index - child_index)

    def __fNum__(self, parent_word, child_word):
        """Number Feature"""
        relation = self.__fNum_internal__(parent_word, child_word)
        if relation:
            return self.__return_feature_index__(("fNum", relation))
        else:
            return False

    def __fNum_internal__(self, parent_word, child_word):
        parent_num = self.data.isNumberWord(parent_word)
        child_num = self.data.isNumberWord(child_word)
        if parent_num and child_num:
            return "ParentChild"
        elif parent_num and not child_num:
            return "Parent"
        elif not parent_num and child_num:
            return "Child"
        else:
            return False

    def __fPunct__(self, parent_word, child_word):
        """Punctuation Feature"""
        relation = self.__fPunct_internal__(parent_word, child_word)
        if relation:
            return self.__return_feature_index__(("fPunct", relation))
        else:
            return False

    def __fPunct_internal__(self, parent_word, child_word):
        parent_sym = self.data.isPunctSym(parent_word)
        child_sym = self.data.isPunctSym(child_word)
        if parent_sym and child_sym:
            return "ParentChild"
        elif parent_sym and not child_sym:
            return "Parent"
        elif not parent_sym and child_sym:
            return "Child"
        else:
            return False

    def __fCap__(self, parent_word, parent_index, child_word, child_index):
        """Words which start with Capital letter and don't start sentence Feature"""
        relation = self.__fCap_internal__(parent_word, parent_index, child_word, child_index)
        if relation:
            return self.__return_feature_index__(("fCap", relation))
        else:
            return False

    @staticmethod
    def __fCap_internal__(parent_word, parent_index, child_word, child_index):
        parent_cap = parent_index > 1 and parent_word.istitle()
        child_cap = child_index > 1 and child_word.istitle()
        if parent_cap and child_cap:
            return "ParentChild"
        elif parent_cap and not child_cap:
            return "Parent"
        elif not parent_cap and child_cap:
            return "Child"
        else:
            return False

    def __fAllCap__(self, parent_word, child_word):
        """Words which is all Capital letters Feature"""
        relation = self.__fPunct_internal__(parent_word, child_word)
        if relation:
            return self.__return_feature_index__(("fAllCap", relation))
        else:
            return False

    @staticmethod
    def __fAllCap_internal__(parent_word, child_word):
        parent_allcap = parent_word.isupper()
        child_allcap = child_word.isupper()
        if parent_allcap and child_allcap:
            return "ParentChild"
        elif parent_allcap and not child_allcap:
            return "Parent"
        elif not parent_allcap and child_allcap:
            return "Child"
        else:
            return False

    def __fPTagW1Before__(self, parent_before, parent):
        return self.__return_feature_index__(("fPTagW1Before", self.__fTagW1Before_internal(parent_before, parent)))

    def __fCTagW1Before__(self, child_before, child):
        return self.__return_feature_index__(("fCTagW1Before", self.__fTagW1Before_internal(child_before, child)))

    def __fPTagW1After__(self, parent_after, parent):
        return self.__return_feature_index__(("fPTagW1After", self.__fTagW1After_internal(parent_after, parent)))

    def __fCTagW1After__(self, child_after, child):
        return self.__return_feature_index__(("fCTagW1After", self.__fTagW1After_internal(child_after, child)))

    def __fPTagW1__(self, parent_before, parent_after):
        return self.__return_feature_index__(("fPTagW1", self.__fTagW1_internal(parent_before, parent_after)))

    def __fCTagW1__(self, child_before, child_after):
        return self.__return_feature_index__(("fCTagW1", self.__fTagW1_internal(child_before, child_after)))

    def __fPBeforeCBefore__(self, parent_before, child_before):
        return self.__return_feature_index__(("fPBeforeCBefore", self.__fTagW1_internal(parent_before, child_before)))

    def __fPBeforeCAfter__(self, parent_before, child_after):
        return self.__return_feature_index__(("fPBeforeCAfter", self.__fTagW1_internal(parent_before, child_after)))

    def __fPAfterCBefore__(self, parent_after, child_before):
        return self.__return_feature_index__(("fPAfterCBefore", self.__fTagW1_internal(parent_after, child_before)))

    def __fPAfterCAfter__(self, parent_after, child_after):
        return self.__return_feature_index__(("fPAfterCAfter", self.__fTagW1_internal(parent_after, child_after)))

    @staticmethod
    def __fTagW1Before_internal(before, current):
        if before is None:
            before_tag = STOP_BEFORE
        else:
            before_tag = before.get_pos_tag()
        if current is not None:
            current_tag = current.get_pos_tag()
        else:
            current_tag = None
        return before_tag, current_tag

    @staticmethod
    def __fTagW1After_internal(after, current):
        if after is None:
            after_tag = STOP_AFTER
        else:
            after_tag = after.get_pos_tag()
        if current is not None:
            current_tag = current.get_pos_tag()
        else:
            current_tag = None
        return after_tag, current_tag

    def __fTagW1_internal(self, before, after):
        before_tag = self.__fTagW1Before_internal(before, None)
        after_tag = self.__fTagW1After_internal(after, None)
        return before_tag, after_tag

    def __fPTagW2__(self, parent_2before, parent_before, parent_after, parent_2after):
        return self.__return_feature_index__(("fPTagW2", self.__fTagW2_internal(parent_2before, parent_before, parent_after, parent_2after)))

    def __fCTagW2__(self, child_2before, child_before, child_after, child_2after):
        return self.__return_feature_index__(("fCTagW2", self.__fTagW2_internal(child_2before, child_before, child_after, child_2after)))

    def __fTagW2_internal(self, before2, before, after, after2):
        before_tag, after_tag = self.__fTagW1_internal(before, after)
        before2_tag, after2_tag = self.__fTagW1_internal(before2, after2)
        return before2_tag, before_tag, after_tag, after2_tag

    def __fTagsBetween__(self, parent, child):
        if MIN_DIST < abs(parent.get_index() - child.get_index()) <= MAX_DIST:
            return self.__return_feature_index__(("fTagsBetween", self.__fTagsBetween_internal(parent, child)))
        else:
            return False

    @staticmethod
    def __fTagsBetween_internal(parent, child):
        parent_index = parent.get_index()
        child_index = child.get_index()
        tags = list()
        if parent_index > child_index:
            start = child
            end = parent_index - child_index
        else:
            start = parent
            end = child_index - parent_index
        for i in range(1, end):
            node = start.get_neighbor(i)
            tags.append(node.get_pos_tag())
        return tuple(tags)

    def __fCumDist10__(self, parent_index, child_index):
        """Edge Distance Features"""
        if self.__fCumDist_internal__(parent_index, child_index, 10):
            return self.__return_feature_index__(("fCumDist10", "fCumDist10"))
        else:
            return False

    def __fCumDist25__(self, parent_index, child_index):
        """Edge Distance Features"""
        if self.__fCumDist_internal__(parent_index, child_index, 25):
            return self.__return_feature_index__(("fCumDist25", "fCumDist25"))
        else:
            return False

    @staticmethod
    def __fCumDist_internal__(parent_index, child_index, limit):
        return abs(parent_index - child_index) > limit

    def __fPSuffix__(self, parent_word):
        suffix = self.__fSuffix_internal(parent_word)
        if suffix:
            return self.__return_feature_index__(("fPSuffix", suffix))
        else:
            return False

    def __fCSuffix__(self, child_word):
        suffix = self.__fSuffix_internal(child_word)
        if suffix:
            return self.__return_feature_index__(("fCSuffix", suffix))
        else:
            return False

    @staticmethod
    def __fSuffix_internal(word):
        for suffix in SUFFIXES:
            if word.lower().endswith(suffix):
                return suffix
        return False

    def __fPPrefix__(self, parent_word):
        prefix = self.__fPrefix_internal(parent_word)
        if prefix:
            return self.__return_feature_index__(("fPPrefix", prefix))
        else:
            return False

    def __fCPrefix__(self, child_word):
        prefix = self.__fPrefix_internal(child_word)
        if prefix:
            return self.__return_feature_index__(("fCPrefix", prefix))
        else:
            return False

    @staticmethod
    def __fPrefix_internal(word):
        for prefix in PREFIXES:
            if word.lower().startswith(prefix):
                return prefix
        return False

    def __generate_features_dictionaries__(self):
        for feature in self.getFeatureNames():
            self.features_dicts[feature] = OrderedCounter()
        for graph in self.graphs:
            for edge in graph.get_edges():
                self.__basic_features_dictionaries__(edge)
                self.__advanced_features_dictionaries__(edge)
        self.__add_basic_missing_tags__()
        self.__add_advanced_missing_tags__()

    def __add_advanced_missing_tags__(self):
        all_tags = self.data.get_pos_tags()
        all_tag_pairs = combinations_with_replacement(all_tags, 2)
        self.__add_missing_tags__("fPTagW1Before", all_tag_pairs)
        self.__add_missing_tags__("fCTagW1Before", all_tag_pairs)
        self.__add_missing_tags__("fPTagW1After", all_tag_pairs)
        self.__add_missing_tags__("fCTagW1After", all_tag_pairs)
        self.__add_missing_tags__("fPBeforeCBefore", all_tag_pairs)
        self.__add_missing_tags__("fPAfterCBefore", all_tag_pairs)
        self.__add_missing_tags__("fPBeforeCAfter", all_tag_pairs)
        self.__add_missing_tags__("fPAfterCAfter", all_tag_pairs)

    def __advanced_features_dictionaries__(self, edge):
        parent = edge.get_source()
        parent_word = parent.get_word()
        parent_index = parent.get_index()
        parent_before = parent.get_neighbor(-1)
        parent_after = parent.get_neighbor(1)
        child = edge.get_target()
        child_word = child.get_word()
        child_index = child.get_index()
        child_before = child.get_neighbor(-1)
        child_after = child.get_neighbor(1)
        self.features_dicts["fDir"][self.__fDir_internal__(parent_index, child_index)] += 1
        self.features_dicts["fDist"][self.__fDist_internal__(parent_index, child_index)] += 1
        relation = self.__fNum_internal__(parent_word, child_word)
        if relation:
            self.features_dicts["fNum"][relation] += 1
        relation = self.__fPunct_internal__(parent_word, child_word)
        if relation:
            self.features_dicts["fPunct"][relation] += 1
        relation = self.__fCap_internal__(parent_word, parent_index, child_word, child_index)
        if relation:
            self.features_dicts["fCap"][relation] += 1
        relation = self.__fAllCap_internal__(parent_word, child_word)
        if relation:
            self.features_dicts["fAllCap"][relation] += 1
        self.features_dicts["fPTagW1Before"][self.__fTagW1Before_internal(parent_before, parent)] += 1
        self.features_dicts["fCTagW1Before"][self.__fTagW1Before_internal(child_before, child)] += 1
        self.features_dicts["fPTagW1After"][self.__fTagW1After_internal(parent_after, parent)] += 1
        self.features_dicts["fCTagW1After"][self.__fTagW1After_internal(child_after, child)] += 1
        self.features_dicts["fPBeforeCBefore"][self.__fTagW1_internal(parent_before, child_before)] += 1
        self.features_dicts["fPBeforeCAfter"][self.__fTagW1_internal(parent_before, child_after)] += 1
        self.features_dicts["fPAfterCBefore"][self.__fTagW1_internal(parent_after, child_before)] += 1
        self.features_dicts["fPAfterCAfter"][self.__fTagW1_internal(parent_after, child_after)] += 1
        if MIN_DIST < abs(parent_index-child_index) <= MAX_DIST:
            self.features_dicts["fTagsBetween"][self.__fTagsBetween_internal(parent, child)] += 1
        if self.__fCumDist_internal__(parent_index, child_index, 10):
            self.features_dicts["fCumDist10"]["fCumDist10"] += 1
        prefix = self.__fPrefix_internal(parent_word)
        if prefix:
            self.features_dicts["fPPrefix"][prefix] += 1
        prefix = self.__fPrefix_internal(child_word)
        if prefix:
            self.features_dicts["fCPrefix"][prefix] += 1
        suffix = self.__fSuffix_internal(parent_word)
        if suffix:
            self.features_dicts["fPSuffix"][suffix] += 1
        suffix = self.__fSuffix_internal(child_word)
        if suffix:
            self.features_dicts["fCSuffix"][suffix] += 1

    def getFeaturesIndices(self, edge):
        """method to return the feature vector indices of the specific feature instances for a given HistoryTuple and tag
        first calls on parent (BasicFeatures) method to retrieve th indices of Basic model features, and add the Advanced"""
        indices = super().getFeaturesIndices(edge)
        parent = edge.get_source()
        parent_word = parent.get_word()
        parent_index = parent.get_index()
        parent_before = parent.get_neighbor(-1)
        parent_after = parent.get_neighbor(1)
        child = edge.get_target()
        child_word = child.get_word()
        child_index = child.get_index()
        child_before = child.get_neighbor(-1)
        child_after = child.get_neighbor(1)
        self.__checkFeatureIndex__(self.__fDir__(parent_index, child_index), indices)
        self.__checkFeatureIndex__(self.__fDist__(parent_index, child_index), indices)
        self.__checkFeatureIndex__(self.__fNum__(parent_word, child_word), indices)
        self.__checkFeatureIndex__(self.__fPunct__(parent_word, child_word), indices)
        self.__checkFeatureIndex__(self.__fCap__(parent_word, parent_index, child_word, child_index), indices)
        self.__checkFeatureIndex__(self.__fAllCap__(parent_word, child_word), indices)
        self.__checkFeatureIndex__(self.__fPTagW1Before__(parent_before, parent), indices)
        self.__checkFeatureIndex__(self.__fCTagW1Before__(child_before, child), indices)
        self.__checkFeatureIndex__(self.__fPTagW1After__(parent_after, parent), indices)
        self.__checkFeatureIndex__(self.__fCTagW1After__(child_after, child), indices)
        self.__checkFeatureIndex__(self.__fPBeforeCBefore__(parent_before, child_before), indices)
        self.__checkFeatureIndex__(self.__fPBeforeCAfter__(parent_before, child_after), indices)
        self.__checkFeatureIndex__(self.__fPAfterCBefore__(parent_after, child_before), indices)
        self.__checkFeatureIndex__(self.__fPAfterCAfter__(parent_after, child_after), indices)
        self.__checkFeatureIndex__(self.__fTagsBetween__(parent, child), indices)
        self.__checkFeatureIndex__(self.__fCumDist10__(parent_index, child_index), indices)
        self.__checkFeatureIndex__(self.__fPPrefix__(parent_word), indices)
        self.__checkFeatureIndex__(self.__fCPrefix__(child_word), indices)
        self.__checkFeatureIndex__(self.__fPSuffix__(parent_word), indices)
        self.__checkFeatureIndex__(self.__fCSuffix__(child_word), indices)
        return indices

    def getFeatureNames(self):
        """method to return all feature names of Basic and Advanced model for feature vector creation"""
        feature_names = super().getFeatureNames()
        feature_names.extend(["fDir", "fDist", "fNum", "fPunct", "fCap", "fAllCap", "fCumDist10", "fTagsBetween", "fPBeforeCBefore", "fPAfterCBefore", "fPBeforeCAfter", "fPAfterCAfter", "fPPrefix", "fPSuffix", "fCPrefix", "fCSuffix", "fPTagW1Before", "fCTagW1Before", "fPTagW1After", "fCTagW1After"])
        return feature_names

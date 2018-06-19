from collections import defaultdict

from DependencyGraph import Node, DependencyGraph, FullGraph
from global_consts import ROOT, ROOTINDEX, ENDOFSENTENCE, DELIMITER

PUNCT_SYM = {',', '.', '?', ':', ';', '``', '\'\'', '(', ')', '{', '}', '...', '-', '!', '\'', '"'}


class DependencyDataReader:
    def __init__(self, file):
        self.file = file
        # word: freq
        self.word_dict = defaultdict(int)
        # pos: freq
        self.pos_dict = defaultdict(int)
        # (word, index): freq
        self.word_index_dict = defaultdict(int)
        # (word, pos): freq
        self.word_pos_dict = defaultdict(int)
        # (word, head): freq
        self.word_head_dict = defaultdict(int)
        # sentence_id: [sentence]
        self.sentences_ids = defaultdict(list)
        # sentence_id: [successors]
        self.sentences_successors_dict = {}
        # sentence_id: graph obj
        self.sentence_graphs = {}
        # sentence_id: word_index: head_index
        self.sentence_dependencies = {}
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as data:
            sentence_id = 1
            sentence = []
            successors = defaultdict(list)
            nodes = {ROOTINDEX: Node(ROOT, ROOTINDEX, ROOT)}
            index_head_dict = defaultdict(int)
            for line in data:
                if line is ENDOFSENTENCE:
                    self.sentence_dependencies[sentence_id] = index_head_dict
                    self.sentences_successors_dict[sentence_id] = successors
                    self.sentences_ids[sentence_id] = sentence
                    self.sentence_graphs[sentence_id] = DependencyGraph(sentence_id, nodes, successors)
                    sentence = []
                    index_head_dict = defaultdict(int)
                    successors = defaultdict(list)
                    nodes = {ROOTINDEX: Node(ROOT, ROOTINDEX, ROOT)}
                    sentence_id += 1
                    continue
                items = line.rstrip().split(DELIMITER)
                index = int(items[0])
                word = str(items[1])
                pos = str(items[3])
                head = int(items[6])
                self.word_dict[word] += 1
                self.pos_dict[pos] += 1
                self.word_index_dict[word, index] += 1
                self.word_pos_dict[word, pos] += 1
                self.word_head_dict[word, head] += 1
                index_head_dict[index] = head
                nodes[index] = Node(word, index, pos, head)
                successors[head] += [index]
                sentence.append(word)

    def get_sentence_graphs(self):
        return self.sentence_graphs.values()

    def get_sentence_successors(self, sentence_id):
        return tuple(self.sentences_successors_dict[sentence_id])

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentence_dependencies)

    def get_sentence_dependencies(self, sentence_id):
        return self.sentence_dependencies.get(sentence_id, None)

    def get_sentence_ids(self):
        return self.sentence_dependencies.keys()

    def get_pos_tags(self):
        return self.pos_dict.keys()

    @staticmethod
    def isNumberWord(word):
        if word.isdigit():
            return True
        elif word.isnumeric():
            return True
        elif word.isdecimal():
            return True
        else:
            for char in {'-', ',', '.', '\/'}:
                word = word.replace(char, '')
                if word.isdigit():
                    return True
            return False

    @staticmethod
    def isPunctSym(word):
        if word in PUNCT_SYM:
            return True
        return False


class UnlabeledDataReader:
    def __init__(self, file):
        self.file = file
        # word: freq
        self.word_dict = defaultdict(int)
        # (word, index): freq
        self.word_index_dict = defaultdict(int)
        # (word, pos): freq
        self.word_pos_dict = defaultdict(int)
        # sentence_id: word_id: (word,pos)
        self.sentences_ids = {}
        # sentence_id: [successors]
        self.sentences_successors_dict = {}
        # sentence_id: graph obj
        self.sentence_graphs = {}
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as data:
            sentence_id = 1
            sentence = {}
            nodes = {ROOTINDEX: Node(ROOT, ROOTINDEX, ROOT)}
            for line in data:
                if line is ENDOFSENTENCE:
                    self.sentences_ids[sentence_id] = sentence
                    self.sentence_graphs[sentence_id] = FullGraph(sentence_id, nodes)
                    sentence = {}
                    nodes = {ROOTINDEX: Node(ROOT, ROOTINDEX, ROOT)}
                    sentence_id += 1
                    continue
                items = line.rstrip().split(DELIMITER)
                index = int(items[0])
                word = str(items[1])
                pos = str(items[3])
                self.word_dict[word] += 1
                self.word_index_dict[word, index] += 1
                self.word_pos_dict[word, pos] += 1
                nodes[index] = Node(word, index, pos)
                sentence[index] = (word, pos)

    def get_sentence_graphs(self):
        return self.sentence_graphs.values()

    def get_sentence_successors(self, sentence_id):
        return tuple(self.sentences_successors_dict[sentence_id])

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences_ids)

    def get_sentence_ids(self):
        return self.sentences_ids.keys()

    def get_sentences(self):
        return self.sentences_ids

    @staticmethod
    def isNumberWord(word):
        if word.isdigit():
            return True
        elif word.isnumeric():
            return True
        elif word.isdecimal():
            return True
        else:
            for char in {'-', ',', '.', '\/'}:
                word = word.replace(char, '')
                if word.isdigit():
                    return True
            return False

    @staticmethod
    def isPunctSym(word):
        if word in PUNCT_SYM:
            return True
        return False

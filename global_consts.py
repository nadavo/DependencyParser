"""Constants"""
DELIMITER = '\t'
ENDOFSENTENCE = '\n'
EMPTY = '_'
ROOTINDEX = 0
ROOT = 'ROOT'
NUM_THREADS = 4
STOP_BEFORE = "*"
STOP_AFTER = "STOP"
MIN_DIST = 1
MAX_DIST = 10

"""File names"""
dev_file = "data/devdata.labeled"
dev_train_file = "data/devtrain.labeled"
dev_test_file = "data/devtest.labeled"
train_file = "data/train.labeled"
test_file = "data/test.labeled"
comp_file = "data/comp.unlabeled"
devcomp_file = "data/devcomp.unlabeled"
all_file = "data/all.labeled"

basic_tagged_comp_file = "competition/comp_m1_200689768.wtag"
advanced_tagged_comp_file = "competition/comp_m2_200689768.wtag"

PREFIXES = sorted(['anti', 'auto', 'de', 'dis', 'extra', 'hyper', 'il', 'im', 'in', 'ir', 'inter', 'mega', 'mid', 'mis', 'non',
            'over', 'out', 'post', 'pre', 'pro', 're', 'semi', 'sub', 'super', 'tele', 'trans', 'ultra', 'un', 'under',
            'up'], key=lambda x: len(x), reverse=True)
SUFFIXES = sorted(['age', 'al', 'ance', 'ence', 'dom', 'ee', 'er', 'or', 'hood', 'ism', 'ist', 'ty', 'ment', 'ness', 'ry',
            'ship', 'ion', 'ble', 'al', 'en', 'ese', 'ful', 'ic', 'ish', 'ive', 'ian', 'less', 'ly', 'ous', 'ate',
            'ify', 'ise', 'ize', 'ward', 'wise'], key=lambda x: len(x), reverse=True)

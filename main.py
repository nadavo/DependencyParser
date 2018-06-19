from DependencyDataReader import DependencyDataReader
from DependencyParser import DependencyParser
from FeaturesFactory import BasicFeatures, AdvancedFeatures
from utils import Timer, sendEmail
from global_consts import train_file, dev_file, test_file, dev_test_file, dev_train_file, all_file, comp_file, devcomp_file
from competition import generateCompTagging
from sys import argv


def main():
    global_timer = Timer("Total Runtime")
    if len(argv) == 3:
        NUM_ITERATIONS = int(argv[1])
        FEATURES_CUTOFF = int(argv[2])
    elif len(argv) == 2:
        NUM_ITERATIONS = int(argv[1])
        FEATURES_CUTOFF = 0
    else:
        NUM_ITERATIONS = 20
        FEATURES_CUTOFF = 0
    evaluate_per_iteration = False
    pretrained_weights = None

    time = Timer('Data reader')
    train_data = DependencyDataReader(all_file)
    time.stop()
    print("Number of sentences:", train_data.get_num_sentences())
    time = Timer('Advanced Features')
    features = AdvancedFeatures(train_data, FEATURES_CUTOFF)
    features.initialize_vector()
    time.stop()
    print("Number of Features:", features.getFeaturesVectorLength())
    model = DependencyParser(features, pretrained_weights)
    if pretrained_weights is None:
        model.fit(NUM_ITERATIONS, evaluate_per_iteration)
    results = ["Number of Iterations: " + str(NUM_ITERATIONS), "Feature Cutoff: " + str(FEATURES_CUTOFF)]
    model.predict(train_data)
    results.append(str(model.evaluate(train_data)))
    test_data = DependencyDataReader(dev_test_file)
    print("Number of sentences:", test_data.get_num_sentences())
    model.predict(test_data)
    results.append(str(model.evaluate(test_data)))
    sendEmail(results)
    # generateCompTagging(devcomp_file, model)
    global_timer.stop()


if __name__ == '__main__':
    main()

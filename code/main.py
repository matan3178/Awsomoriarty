from tables.idxutils import infinity

from code.Evaluator import evaluate_classifier, evaluate_ids, train_classifier, evaluate_classifier_in_range
from code._definitions import VERBOSITY_general
from code.classifiers.ClassifierGenerator import *
from code.data_handles.DataCenter import DataCenter
from code.features.FetureExtractorUtil import start
from code.ids.AccumulativeOnesIDS import AccumulativeOnesIDS
from code.ids.ContiguousOnesIDS import ContiguousOnesIDS


def do_something():
    # run_feature_extraction_tests()

    ds = DataCenter()
    ds.load_data_collection3v2()
    print("(finished loading)")

    best_dist = infinity
    for h in ds.user_hashes:
        print("USER {}".format(h), HEADER)
        training_set = start(ds.users_training[h][:1000])
        testing_benign, testing_theft = start(ds.users_testing[h][0][0][:1000]), start(ds.users_testing[h][0][1][:1000])

        classifiers = list()
        classifiers.append(generate_one_class_svm_linear())
        classifiers.append(generate_one_class_svm_sigmoid())
        classifiers.append(generate_one_class_svm_rbf())
        classifiers.append(generate_one_class_svm_poly())
        classifiers.append(generate_lstm_autoencoder(len(training_set[0]), 10))

        for c in classifiers:
            train_classifier(c, training_set)

        for c in classifiers:
            if c.has_threshold():
                evaluate_classifier_in_range(classifier=c, training_set=training_set,
                                         test_set_benign=testing_benign, test_set_fraud=testing_theft,
                                         threshold_begin=0.001, threshold_end=0.1, num_of_steps=10,
                                         verbosity=VERBOSITY_general)
            else:
                print("Normal Evaluation (no threshold):", UNDERLINE + OKGREEN)
                evaluate_classifier(c, training_set, testing_benign, testing_theft, VERBOSITY_general)

        IDSs = list()
        IDSs.extend([ContiguousOnesIDS(classifier=c, threshold=10) for c in classifiers])
        IDSs.extend([AccumulativeOnesIDS(classifier=c, threshold=10) for c in classifiers])

        for ids in IDSs:
            current_dist = evaluate_ids(ids=ids, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=VERBOSITY_general)
            if abs(current_dist) < abs(best_dist):
                best_dist = current_dist

        print("best distance (for user {}): {}".format(h, best_dist))
    return


def main():
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)
    return

if __name__ == "__main__":
    main()

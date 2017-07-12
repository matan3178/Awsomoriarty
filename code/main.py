from tables.idxutils import infinity

from code.DataCenter import DataCenter
from code.Evaluator import train_and_evaluate
from code.classifiers.ClassifierGenerator import *
from code._definitions import VERBOSITY_general
from code.features.FetureExtractorUtil import start


def do_something():
    # run_feature_extraction_tests()

    ds = DataCenter()
    ds.load_data_collection3v2()
    print("(finished loading)")

    best_dist = infinity
    for h in ds.user_hashes:
        print("USER {}".format(h), HEADER)
        training_set = start(ds.users_training[h])
        testing_benign, testing_theft = start(ds.users_testing[h][0][0]), start(ds.users_testing[h][0][1])

        classifiers = list()
        # classifiers.append(generate_one_class_svm_linear())
        # classifiers.append(generate_one_class_svm_sigmoid())
        # classifiers.append(generate_one_class_svm_rbf())
        # classifiers.append(generate_one_class_svm_poly())
        classifiers.append(generate_recurrent_autoencoder(len(training_set[0]), 10))

        for c in classifiers:
            current_dist = train_and_evaluate(classifier=c, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=VERBOSITY_general)[0]
            if abs(current_dist) < abs(best_dist):
                best_dist = current_dist

    print("best distance: {}".format(best_dist))
    return


def main():
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)

    LogSingleton.get_singleton().print_log()
    return

if __name__ == "__main__":
    main()

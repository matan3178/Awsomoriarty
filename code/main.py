from definitions import VERBOSITY
from log.Print import *
from code.DataCenter import DataCenter
from code.Evaluator import train_and_evaluate
from code.FetureExtractorUtil import start, run_feature_extraction_tests
from code.classifiers.ClassifierGenerator import *


def do_something():
    # run_feature_extraction_tests()

    ds = DataCenter()
    ds.load_data_collection3v2()
    h = ds.user_hashes[0]
    training_set = start(ds.users_training[h][:2000])
    print(training_set[0])
    testing_benign, testing_theft = start(ds.users_testing[h][0][0]), start(ds.users_testing[h][0][1])

    svm_sigmoid = generate_one_class_svm_sigmoid()
    svm_poly = generate_one_class_svm_poly()
    svm_rbf = generate_one_class_svm_rbf()
    svm_linear = generate_one_class_svm_linear()

    train_and_evaluate(classifier=svm_linear, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=VERBOSITY)
    train_and_evaluate(classifier=svm_rbf, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=VERBOSITY)
    train_and_evaluate(classifier=svm_sigmoid, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=VERBOSITY)
    train_and_evaluate(classifier=svm_poly, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=VERBOSITY)

    return


def main():
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)
    return

if __name__ == "__main__":
    main()

import definitions
from code.DataCenter import DataCenter
from code.Evaluator import train_and_evaluate
from code.FetureExtractorUtil import start, samples_to_np_arrays
from log.Print import *
from code.FileLoader import FileLoader
from code.ClassifierGenerator import *


def do_something():
    ds = DataCenter()
    ds.load_data_collection3v2()
    h = ds.user_hashes[0]
    training_set = start(ds.users_training[h][:10000])
    testing_benign, testing_theft = start(ds.users_testing[h][0][0][:10000]), start(ds.users_testing[h][0][1][:10000])

    svm_sigmoid = generate_one_class_svm_sigmoid()
    svm_poly = generate_one_class_svm_poly()
    svm_rbf = generate_one_class_svm_rbf()

    print("One Class SVM (rbf)", HEADER)
    train_and_evaluate(classifier=svm_rbf, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=1)

    print("One Class SVM (sigmoid)", HEADER)
    train_and_evaluate(classifier=svm_sigmoid, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=1)

    print("One Class SVM (poly)", HEADER)
    train_and_evaluate(classifier=svm_poly, training_set=training_set, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=1)

    return


def main():
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)
    return

if __name__ == "__main__":
    main()

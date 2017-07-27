import datetime
import matplotlib
from matplotlib.pyplot import plot
from skimage.color.rgb_colors import green, red, blue
from tables.idxutils import infinity
from win32timezone import now

from code.classifiers.OfflineLOF import OfflineLOF
from code.log.Print import *
from code.Evaluator import evaluate_classifier, evaluate_ids, train_classifier, evaluate_classifier_in_range, \
    benign_and_fraud_sets_to_x_y
from code._definitions import VERBOSITY_general
from code.classifiers.ClassifierGenerator import *
from code.data_handles.DataCenter import DataCenter
from code.features.FeatureSelection import split_information_gain, select_k_best_features
from code.features.FetureExtractorUtil import start, remove_all_columns_except, encode
from code.ids.AccumulativeOnesIDS import AccumulativeOnesIDS
from code.ids.ContiguousOnesIDS import ContiguousOnesIDS
import code.plotting.scatter_plot as scatplot


def do_something():
    # run_feature_extraction_tests()

    dc = DataCenter()
    dc.load_data_collection3v2()
    print("(finished loading)")

    for h in dc.user_hashes:
        print("USER {}".format(h), HEADER)
        print("extracting features...", HEADER)
        training_set = start(dc.users_training[h])
        testing_benign, testing_theft = start(dc.users_testing[h][0][0]), start(dc.users_testing[h][0][1])

        print("selecting features...", HEADER)
        selected_feature_indexes = select_k_best_features(k=15, label1_data=testing_benign, label2_data=testing_theft)
        training_set = remove_all_columns_except(training_set, selected_feature_indexes)
        testing_benign = remove_all_columns_except(testing_benign, selected_feature_indexes)
        testing_theft = remove_all_columns_except(testing_theft, selected_feature_indexes)

        print("generating classifiers...", HEADER)
        classifiers = list()
        # classifiers.append(generate_one_class_svm_linear())
        # classifiers.append(generate_one_class_svm_sigmoid())
        classifiers.append(generate_one_class_svm_rbf())
        classifiers.append(generate_one_class_svm_poly())
        encoder_decoder, encoder = generate_autoencoder(len(training_set[0]), hidden_to_input_ratio=0.3)
        # classifiers.append(encoder_decoder)
        # classifiers.append(generate_lstm_autoencoder(len(training_set[0]), 5))
        classifiers.append(OfflineLOF(k=10))

        train_classifier(encoder_decoder, training_set)
        testing_benign = encoder.predict(testing_benign)
        testing_theft = encoder.predict(testing_theft)
        training_set = encoder.predict(training_set)

        # scatplot.plot3d(testing_benign, testing_theft, color1=blue, color2=red)

        print("training classifiers...", HEADER)
        for c in classifiers:
            train_classifier(c, training_set)

        distances = list()
        for i in range(1, len(dc.users_testing[h])):
            # load test data
            testing_benign, testing_theft = start(dc.users_testing[h][i][0]), start(dc.users_testing[h][i][1])

            # filter out features based on previous feature selection
            testing_benign = remove_all_columns_except(testing_benign, selected_feature_indexes)
            testing_theft = remove_all_columns_except(testing_theft, selected_feature_indexes)

            testing_benign = encoder.predict(testing_benign)
            testing_theft = encoder.predict(testing_theft)

            best_dist = evaluate(classifiers, testing_benign, testing_theft)
            distances.append(best_dist)
            print("best distance (for user {}, test {}): {}".format(h, i, best_dist), BOLD + OKBLUE)
            print("plotting...", COMMENT)
            # scatplot.plot3d(testing_benign, testing_theft, color1=blue, color2=red)
            print("_____________")

        print("\ndistances: {}\n".format(distances), BOLD + OKBLUE)
    return


def evaluate(classifiers, testing_benign, testing_theft):
    print("evaluating...", HEADER)
    for c in classifiers:
        if c.has_threshold():
            opt_threshold = evaluate_classifier_in_range(classifier=c, test_set_benign=testing_benign,
                                                         test_set_fraud=testing_theft, threshold_begin=0,
                                                         threshold_end=0.1, num_of_steps=10000,
                                                         verbosity=0)
            c.set_threshold(opt_threshold)
            print("{} has been set threshold {}".format(c.get_name(), opt_threshold), COMMENT)
        else:
            print("Normal Evaluation (no threshold):", UNDERLINE + OKGREEN)
            evaluate_classifier(classifier=c, test_set_benign=testing_benign, test_set_fraud=testing_theft, verbosity=VERBOSITY_general)

    IDSs = list()
    for c in classifiers:
        for j in range(15):
            IDSs.append(ContiguousOnesIDS(classifier=c, threshold=j + 1))
            IDSs.append(AccumulativeOnesIDS(classifier=c, threshold=j + 1))
    print("EVALUATING ANOMALY DETECTORS BBAAAAATTTTT ZZZONNNNNNAAAAAAAA !!!!!!!!!!", UNDERLINE + FAIL)
    print("")
    best_dist = infinity
    for ids in IDSs:
        current_dist = evaluate_ids(ids=ids, test_set_benign=testing_benign, test_set_fraud=testing_theft,
                                    verbosity=0)
        if abs(current_dist) < abs(best_dist):
            best_dist = current_dist
    return best_dist


def main():
    # with open("log.txt", "r") as text_file:
    #     print("Printing LOG:____________________________________________", HEADER + BOLD + UNDERLINE)
    #     blank_line()
    #     print(text_file.read())
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)
    with open("log_{}.txt".format(datetime.datetime.now()), "w") as text_file:
        text_file.write(LogSingleton.get_singleton().get_log_string())
    return

if __name__ == "__main__":
    main()

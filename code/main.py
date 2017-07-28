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
from code.features.FeatureSelection import split_information_gain, select_k_best_features_fisher_score
from code.features.FetureExtractorUtil import *
from code.ids.AccumulativeOnesIDS import AccumulativeOnesIDS
from code.ids.ContiguousOnesIDS import ContiguousOnesIDS
import code.plotting.scatter_plot as scatplot


def vectorize(list_of_numbers):
    return [[n] for n in list_of_numbers]


def do_something():
    # run_feature_extraction_tests()

    dc = DataCenter()
    dc.load_data_collection3v2()
    print("(finished loading)")

    for h in dc.user_hashes:
        print("USER {}".format(h), HEADER)
        print("extracting features...", HEADER)
        agg_win_size, agg_slide_size = 1, 1
        training_set = extract_features(dc.users_training[h], agg_win_size, agg_slide_size)[:100]
        testing_benign, testing_theft = dc.users_testing[h][0][0], dc.users_testing[h][0][1]
        testing_benign, testing_theft = extract_features_tests_separated(testing_benign, testing_theft,
                                                                         agg_win_size, agg_slide_size)

        print("selecting features...", HEADER)
        selected_feature_indexes = select_k_best_features_fisher_score(k=15, label1_data=testing_benign, label2_data=testing_theft)
        training_set = remove_all_columns_except(training_set, selected_feature_indexes)
        testing_benign = remove_all_columns_except(testing_benign, selected_feature_indexes)
        testing_theft = remove_all_columns_except(testing_theft, selected_feature_indexes)

        print("generating classifiers...", HEADER)
        classifiers = list()
        # classifiers.append(generate_one_class_svm_linear())
        # classifiers.append(generate_one_class_svm_sigmoid())
        # classifiers.append(generate_one_class_svm_rbf())
        # classifiers.append(generate_one_class_svm_poly())
        encoder_decoder, encoder = generate_autoencoder(len(training_set[0]), hidden_to_input_ratio=0.3)
        # classifiers.append(encoder_decoder)
        # classifiers.append(generate_lstm_autoencoder(len(training_set[0]), 5))
        classifiers.append(OfflineLOF(k=3))

        train_classifier(encoder_decoder, training_set)
        training_set = vectorize(encoder_decoder.predict_raw(training_set))

        # testing_benign = vectorize(encoder_decoder.predict_raw(testing_benign))
        # testing_theft = vectorize(encoder_decoder.predict_raw(testing_theft))
        # scatplot.plot1d(testing_benign, testing_theft, color1=blue, color2=red, offset_dim=0)

        print("training classifiers...", HEADER)
        for c in classifiers:
            train_classifier(c, training_set)

        distances = list()
        for i in range(1, len(dc.users_testing[h])):
            # load test data
            testing_benign, testing_theft = dc.users_testing[h][i][0], dc.users_testing[h][i][1]
            testing_benign, testing_theft = extract_features_tests_separated(testing_benign, testing_theft, agg_win_size, agg_slide_size)

            # filter out features based on previous feature selection
            testing_benign = remove_all_columns_except(testing_benign, selected_feature_indexes)
            testing_theft = remove_all_columns_except(testing_theft, selected_feature_indexes)

            testing_benign = vectorize(encoder_decoder.predict_raw(testing_benign))
            testing_theft = vectorize(encoder_decoder.predict_raw(testing_theft))

            best_dist = evaluate(classifiers, testing_benign, testing_theft)
            distances.append(best_dist)
            print("best distance (for user {}, test {}): {}".format(h, i, best_dist), BOLD + OKBLUE)
            print("plotting...", COMMENT)
            # scatplot.plot1d(testing_benign, testing_theft, color1=blue, color2=red, offset_dim=0)
            print("_____________")

        print("\ndistances: {}\n".format(distances), BOLD + OKBLUE)
    return


def extract_features_tests_separated(testing_benign, testing_theft, agg_win_size, agg_slide_size):
    testing = list()
    testing_benign, testing_theft = cleanup(testing_benign), cleanup(testing_theft)
    testing.extend(testing_benign)
    testing.extend(testing_theft)
    testing = extract_features(testing, agg_win_size, agg_slide_size)
    testing_benign = testing[:int(len(testing_benign) / agg_slide_size)]
    testing_theft = testing[int(len(testing_benign) / agg_slide_size):]

    return testing_benign, testing_theft


def extract_features(list_of_samples, agg_win_size=10, agg_slide_size=10):
    list_of_samples = cleanup(list_of_samples)
    list_of_samples = aggregate_samples_using_sliding_windows(list_of_samples, agg_win_size, agg_slide_size)
    list_of_samples = normalize_feature_vector_to_unit_size(list_of_samples)
    # list_of_samples = derivate_samples(list_of_samples)
    list_of_samples = finish(list_of_samples)
    return list_of_samples


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
        for j in range(0, 90, 3):
            IDSs.append(ContiguousOnesIDS(classifier=c, threshold=j + 1))
            IDSs.append(AccumulativeOnesIDS(classifier=c, threshold=j + 1))
    print("EVALUATING ANOMALY DETECTORS", UNDERLINE + OKBLUE)
    print("")
    best_dist = infinity
    for ids in IDSs:
        current_dist = evaluate_ids(ids=ids, test_set_benign=testing_benign, test_set_fraud=testing_theft,
                                    verbosity=0)
        print("{}: distance={}".format(ids, current_dist))
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

    # save log
    date = datetime.datetime.now()
    filename = "log_{}-{}-{}.{}-{}-{}.{}.txt".format(date.day, date.month, date.year,
                                                     date.hour, date.minute, date.second,
                                                     date.microsecond)
    with open(filename, "w") as text_file:
        text_file.write(LogSingleton.get_singleton().get_log_string())

    return

if __name__ == "__main__":
    main()

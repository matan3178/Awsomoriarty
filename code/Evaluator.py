from math import isnan
from numbers import Number

import numpy as np
from numpy import nan
from tables.idxutils import infinity
from code.log.Print import *


def benign_and_fraud_sets_to_x_y(test_set_benign, test_set_fraud):
    test_set = list()
    test_set.extend(test_set_benign)
    test_set.extend(test_set_fraud)
    target_labels = list()
    target_labels.extend(np.zeros(len(test_set_benign)))
    target_labels.extend(np.ones(len(test_set_fraud)))
    return test_set, target_labels


# ____________________________________________________ Classifiers


def evaluate_classifier(classifier, test_set_benign, test_set_fraud, verbosity=0):
    test_set, target_labels = benign_and_fraud_sets_to_x_y(test_set_benign, test_set_fraud)
    if verbosity > 0:
        print("evaluating classifier {}...".format(classifier.get_name()), UNDERLINE)
        print("predicting...")
    return evaluate_predictions(target_labels, list(classifier_predict(classifier, test_set)), verbosity)


def reevaluate_classifier(classifier, raw_predictions, target_labels, verbosity=0):
    if verbosity > 0:
        print("reevaluating classifier {}".format(classifier.get_name()), COMMENT)
    return evaluate_predictions(target_labels, list(classifier.repredict(raw_predictions)), verbosity)


def train_classifier(classifier, training_set):
    print("training classifier {}...".format(classifier.get_name()))
    classifier.fit(training_set)  # unsupervised learning interface


def classifier_predict(classifier, x_test):
    return classifier.predict(x_test)


def evaluate_predictions(target_labels, predictions, verbosity=0):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    length = len(target_labels)
    for t, p in zip(target_labels, predictions):
        if t == 0 and p == 1:
            fp += 1
        if t == 1 and p == 0:
            fn += 1
        if t == 0 and p == 0:
            tn += 1
        if t == 1 and p == 1:
            tp += 1

    if verbosity > 1:
        print("found {} total number of samples (?= {} ?= {})".format(tp + fp + tn + fn, len(target_labels),
                                                                      len(predictions)), COMMENT)

    err_rate = (fp + fn) / length
    recall = tp / (tp + fn)             if tp + fn > 0 else "denominator ({}) is 0".format("tp + fn")
    precision = tp / (tp + fp)          if tp + fp > 0 else "no positive predictions were found"
    specificity = tn / (fp + tn)        if tn + fp > 0 else "denominator ({}) is 0".format("fp + tn")
    false_alarm_rate = fp / (fp + tn)   if tn + fp > 0 else "denominator ({}) is 0".format("fp + tn")

    if verbosity > 0:
        color = COMMENT
        if verbosity > 1:
            color = NORMAL
        print("tp: {}; fp: {}; tn: {}; fn: {}".format(tp, fp, tn, fn), color)
        print("error rate: {}".format(err_rate), COMMENT)
        print("recall: {}".format(recall), color)
        print("precision: {}".format(precision), COMMENT)
        print("specificity: {}".format(specificity), color)
        print("false alarm rate: {}".format(false_alarm_rate), COMMENT)
        print("")

    return tp, fp, tn, fn, err_rate, precision, recall, specificity, false_alarm_rate


# ____________________________________________________ IDS


def ids_predict(ids, x_test):
    for sample, index in zip(x_test, range(len(x_test))):
        if ids.alert_if_theft(sample) == 1:
            return index
    return infinity


def evaluate_ids_using_real_index_inner(real_fraud_index, predicted_fraud_index, verbosity=0):
    distance = predicted_fraud_index - real_fraud_index
    return distance


def evaluate_ids(ids, test_set_benign, test_set_fraud, verbosity=0):
    print("evaluating ids {}...".format(ids.get_name()), UNDERLINE)
    test_set = list(test_set_benign)
    test_set.extend(test_set_fraud)

    real_fraud_index, predicted_fraud_index = len(test_set_benign), ids_predict(ids, test_set)
    distance = evaluate_ids_using_real_index_inner(real_fraud_index, predicted_fraud_index, verbosity)
    if verbosity > 0:
        print("real_index: {}; predicted_index: {}".format(real_fraud_index, predicted_fraud_index), COMMENT)
        print("distance: {}".format(distance))
        print("")
    return distance


# might need to be changed to support ids/classifier
def do_for_threshold_range(threshold_settable, generate_results_func,
                           num_of_steps, threshold_begin, threshold_end,
                           should_stop=lambda: False):
    results = list()
    t = threshold_begin
    t_step = (threshold_end - threshold_begin) / num_of_steps
    for i in range(num_of_steps):
        threshold_settable.set_threshold(t)
        current_result = generate_results_func()
        results.append([t, current_result])
        if should_stop(current_result):
            print("'do_for_threshold_range' was instructed to stop! returning {} results instead of {}...".format(i, num_of_steps), WARNING)
            break
        t += t_step

    return results


def evaluate_classifier_in_range(classifier, training_set, test_set_benign, test_set_fraud,
                                 num_of_steps=100, threshold_begin=0.01, threshold_end=1,
                                 verbosity=0):
    print("evaluating {} with threshold in range [{},{}] and number of steps = {}".format(classifier.get_name(),
                                                                                          threshold_begin,
                                                                                          threshold_end, num_of_steps),
          UNDERLINE + OKBLUE)
    if not classifier.has_threshold():
        print("classifier doesn't have a threshold property", FAIL)
        raise Exception("(evaluate_classifier_in_range): classifier doesn't have a threshold property")

    lambda_func = "UNINITIALIZED"
    should_stop = lambda res: False
    if classifier.supports_repredictions():
        test_set, target_labels = benign_and_fraud_sets_to_x_y(test_set_benign, test_set_fraud)
        raw_predictions = classifier.predict_raw(test_set)
        print("classifier {} supports re-evaluations!".format(classifier.get_name()), COMMENT)
        print("predictions: {}".format(raw_predictions), COMMENT)
        lambda_func = lambda: reevaluate_classifier(classifier, raw_predictions, target_labels, verbosity)
    else:
        lambda_func = lambda: evaluate_classifier(classifier, test_set_benign, test_set_fraud, verbosity)

    results = do_for_threshold_range(threshold_settable=classifier,
                                     generate_results_func=lambda_func,
                                     num_of_steps=num_of_steps,
                                     threshold_begin=threshold_begin,
                                     threshold_end=threshold_end,
                                     should_stop=should_stop)
    best_threshold = "UNINITIALIZED"
    best_performance_grade = -infinity
    best_result = "UNINITIALIZED"
    for result in results:
        tp, fp, tn, fn = result[1][0], result[1][1], result[1][2], result[1][3]
        current_grade = 5 * tp + tn - 5 * fp - fn
        if isinstance(current_grade, Number):
            if current_grade > best_performance_grade:
                best_performance_grade = current_grade
                best_threshold = result[0]
                best_result = result[1]
    print("best results achieved: {} (threshold={})".format(best_result, best_threshold), BOLD)
    return best_threshold

import numpy as np
from log.Print import *


def train_and_evaluate(classifier, training_set, test_set_benign, test_set_fraud, verbosity=0):
    print("evaluation begun:", UNDERLINE)
    print("Name: {}".format(classifier.name))

    test_set = list()
    test_set.extend(test_set_benign)
    test_set.extend(test_set_fraud)

    target_labels = list()
    target_labels.extend(np.zeros(len(test_set_benign)))
    target_labels.extend(np.ones(len(test_set_fraud)))

    print("training...")
    classifier.fit(training_set)  # unsupervised learning interface

    print("predicting...")
    predictions = list(classifier.predict(test_set))
    prediction_index = 0 #CHANGE THIS
    return evaluate_sequence(len(test_set_benign),prediction_index)
   # return evaluate(target_labels, predictions, verbosity)


def evaluate_sequence(real_fraud_index, predicted_fraud_index):
    if predicted_fraud_index == "NO ALERT":
        return "NO ALERT"
    return predicted_fraud_index-real_fraud_index;

def evaluate(target_labels, predictions, verbosity=0):
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
        print("found {} total number of samples (?= {} ?= {})".format(tp + fp + tn + fn, len(target_labels), len(predictions)), COMMENT)

    err_rate = (fp + fn) / length
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else "no positive predictions were found"
    specificity = tn / (tn + fp)
    false_alarm_rate = fp / (fp + tn)

    if verbosity > 0:
        color = COMMENT
        if verbosity > 1:
            color = NORMAL
        print("error rate: {}".format(err_rate), color)
        print("recall: {}".format(recall), color)
        print("precision: {}".format(precision), color)
        print("specificity: {}".format(specificity), color)
        print("false alarm rate: {}".format(false_alarm_rate), color)

    return tp, fp, tn, fn, err_rate, precision, recall, specificity, false_alarm_rate

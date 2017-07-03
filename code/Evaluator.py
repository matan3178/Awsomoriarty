import numpy as np
from log.Print import *


def train_and_evaluate(self, classifier, training_set, test_set_benign, test_set_fraud):
    test_set = test_set_benign.extend(test_set_fraud)
    target_labels = np.zeros(len(test_set_benign)).extend(np.ones(len(test_set_fraud)))
    classifier.train(training_set) # unsupervised learning interface

    predictions = classifier.predict(test_set)

    return self.evaluate(target_labels, predictions)


def evaluate(self, target_labels, prediction, verbosity=0):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    length = len(target_labels)
    for t, p in zip(target_labels, prediction):
        if t == 0 and p == 1:
            fp += 1
        if t == 1 and p == 0:
            fn += 1
        if t == 0 and p == 0:
            tn += 1
        if t == 1 and p == 1:
            tp += 1
    err_rate = (fp + fn) / length
    recall = tp / (tp + fn)
    percision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    false_alarm_rate = fp  / (fp + tn)

    if verbosity > 0:
        color = COMMENT
        if verbosity > 1:
            color = NORMAL
        print("error rate: {}".format(err_rate), color)
        print("recall: {}".format(recall), color)
        print("percision: {}".format(percision), color)
        print("specificity: {}".format(specificity), color)
        print("false alarm rate: {}".format(false_alarm_rate), color)

    return tp, fp, tn, fn, err_rate, percision, recall, specificity, false_alarm_rate

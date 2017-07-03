import numpy as np
from log.Print import *

class Evaluator:
    def train(self, classifier, training_set, test_set_benign, test_set_fraud):
        test_set = test_set_benign.extend(test_set_fraud)
        test_labels = np.zeros(len(test_set_benign)).extend(np.ones(len(test_set_fraud)))
        classifier.train(training_set)
        prediction=classifier.predict(test_set)
        self.evaluate(test_labels,prediction)

    def evaluate(self, test_lables, prediction):
        FP_counter=0
        FN_counter=0
        TP_counter=0
        TN_counter=0
        length = len(test_lables)
        for t,p in test_lables,prediction:
            if t==0 and p == 1:
                FP_counter += 1
            if t == 1 and p == 0:
                FN_counter += 1
            if t == 0 and p == 0:
                TN_counter += 1
            if t == 1 and p == 1:
                TP_counter += 1
        err_rate = (FP_counter + FN_counter) / length
        recall = TP_counter / (TP_counter + FN_counter)
        percision = TP_counter / (TP_counter + FP_counter)
        specificity = TN_counter / (TN_counter + FP_counter)
        false_alarm_rate = FP_counter  / (FP_counter + TN_counter)

        print("error rate: {}".format(err_rate))
        print("recall: {}".format(recall))
        print("percision: {}".format(percision))
        print("specificity: {}".format(specificity))
        print("false alarm rate: {}".format(false_alarm_rate))

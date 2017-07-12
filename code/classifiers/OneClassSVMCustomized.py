from code.log.Print import *


class OneClassSVMCustomized:

    svm = "UNINITIALIZED"
    name = "UNINITIALIZED"

    def __init__(self, inner_svm, name="OneClassSVM (?)"):
        self.svm = inner_svm
        self.name = name
        return

    def fit(self, x_train):
        self.svm.fit(x_train)
        return

    def predict(self, x_test):
        return [0. if p == -1 else 1. for p in self.svm.predict(x_test)]

    num_of_alerts = 0
    threshold = 1

    def alert_if_theft(self, sample):
        y = 1 if self.svm.predict(list([sample])) > 0 else 0
        if y == 1:
            self.num_of_alerts += 1
        else:
            self.num_of_alerts = 0
        return 1 if self.num_of_alerts > self.threshold else 0

    def has_predict_threshold(self):
        return False

    def has_alert_threshold(self):
        return True

    def set_alert_threshold(self, threshold):
        self.threshold = threshold
        return

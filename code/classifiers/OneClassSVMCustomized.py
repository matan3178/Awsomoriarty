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


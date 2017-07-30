from math import sqrt

from numpy import average

from code._definitions import LOF_TRAIN_VERBOSITY, WARNING, MLOF_TRAIN_VERBOSITY
from code.classifiers.MultiLOF import MultiLOF
from code.classifiers.OfflineLOF import OfflineLOF


class MultiMultiLOF:
    def __init__(self, num_of_mlofs=10, num_of_samples_per_lof=10, k=3):
        self.k = k
        self.num_of_samples_per_lof = num_of_samples_per_lof
        self.num_of_mlofs = num_of_mlofs
        self.mlofs = list()
        for i in range(num_of_mlofs):
            self.mlofs.append(MultiLOF(num_of_samples_per_lof=num_of_samples_per_lof, k=k))
        return

    def fit(self, xs, verbosity=MLOF_TRAIN_VERBOSITY):
        if verbosity > 0:
            print("fittin mmlof with {} mlofs".format(len(self.mlofs)))
        samples_per_partition = int(len(xs)/self.num_of_mlofs)
        for i in range(self.num_of_mlofs-1):
            if verbosity > 1:
                print("mlof {}...".format(i))
            self.mlofs[i].fit(xs[:samples_per_partition])
            xs = xs[samples_per_partition:]
        self.mlofs[-1].fit(xs)
        return

    def predict_raw_single(self, x):
        return average([mlof.predict_raw_single(x) for mlof in self.mlofs])

    def predict_raw(self, xs):
        return [self.predict_raw_single(x) for x in xs]

    def predict_single(self, x):
        predictions = [mlof.predict_single(x) for mlof in self.mlofs]
        # minority voting
        return 0 if 0 in predictions else 1

    def predict(self, xs):
        xs = list(xs)
        return [self.predict_single(x) for x in xs]

    def has_threshold(self):
        return False

    def get_name(self):
        return "MultiMultiLOF({} mlofs)".format(len(self.mlofs))

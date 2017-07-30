from math import sqrt

from numpy import average

from code._definitions import LOF_TRAIN_VERBOSITY, WARNING, MLOF_TRAIN_VERBOSITY
from code.classifiers.OfflineLOF import OfflineLOF


class NamedPoint:
    name = 0

    def __init__(self, value):
        self.name = NamedPoint.name = NamedPoint.name + 1
        self.value = value
        return


class MultiLOF:
    def __init__(self, num_of_samples_per_lof=10, k=3):
        self.k = k
        self.num_of_samples_per_lof = num_of_samples_per_lof
        self.lofs = list()
        return

    def fit(self, xs, verbosity=MLOF_TRAIN_VERBOSITY):
        lof_data_pairs = list()
        if verbosity > 0:
            print("generating lof-data pairs...")

        nps = [NamedPoint(x) for x in xs]

        while len(nps) > 0:
            if verbosity > 1:
                print("samples left: {}...".format(len(nps)))

            point = nps[0]
            nps.remove(point)
            current_lof_points = list()
            bfs_queue = list([point])
            for i in range(self.num_of_samples_per_lof - 1):    # -1 for the point initial point
                for point in bfs_queue:
                    current_lof_points.append(point)
                    bfs_queue.remove(point)

                    nearest_neighbors = self.k_nearest_neighbors(o=point, xs=nps, k=self.k)
                    bfs_queue.extend(nearest_neighbors)
                    for np in nearest_neighbors:
                        nps.remove(np)
                    i += len(nearest_neighbors)
            current_lof_points.extend(bfs_queue)
            if len(current_lof_points) > 1:
                lof_data_pairs.append([OfflineLOF(k=min([self.k, len(current_lof_points)])), [np.value for np in current_lof_points]])

        if verbosity > 0:
            print("training individual lofs...")
        for lof_data in lof_data_pairs:
            lof_data[0].fit(lof_data[1])
            self.lofs.append(lof_data[0])

        return

    @staticmethod
    def distance(np1, np2):
        if len(np1.value) != len(np2.value):
            print("warning: length of p1 != length of p2 ({} != {})".format(len(np1.value), len(np2.value)), WARNING)
        return sqrt(sum([(np1.value[i] - np2.value[i]) ** 2 for i in range(len(np1.value))]))

    def k_nearest_neighbors(self, o, xs, k):
        if len(xs) <= k:
            return xs
        # algorithm runtime: O(k * n)
        # where n is the number of points in the training set
        #
        # (a naive algorithm will have a runtime of O(n * log(n))

        # all possible neighbors and their distances from 'o'
        left_d_ns = [[self.distance(o, x), x] for x in xs]

        # a pool consisting of the k nearest neighbors of o at a given point
        k_nearest = sorted(left_d_ns[:k], key=lambda d_n: d_n[0])
        left_d_ns = left_d_ns[k:]
        max_dist = k_nearest[k - 1][0]

        for d_n in left_d_ns:
            if d_n[0] < max_dist:
                k_nearest.remove(k_nearest[len(k_nearest) - 1])
                k_nearest.append(d_n)
                k_nearest = sorted(k_nearest, key=lambda d_n: d_n[0])
                max_dist = k_nearest[k - 1][0]

        # print([kn[0] for kn in k_nearest], OKBLUE)
        return [d_n[1] for d_n in k_nearest]

    def predict_raw_single(self, x):
        return average([lof.lof(x) for lof in self.lofs])

    def predict_raw(self, xs):
        return [self.predict_raw_single(x) for x in xs]

    def predict_single(self, x):
        predictions = [lof.predict_single(x) for lof in self.lofs]
        # minority voting
        return 0 if 0 in predictions else 1

    def predict(self, xs):
        xs = list(xs)
        return [self.predict_single(x) for x in xs]

    def has_threshold(self):
        return False

    def get_name(self):
        return "MultiLOF({} lofs)".format(len(self.lofs))

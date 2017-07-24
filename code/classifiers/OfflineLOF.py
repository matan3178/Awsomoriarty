from math import sqrt

from numpy import average
import numpy as np
from code.log.Print import *
from code._definitions import *

class LofPoint:
    name = "NO-NAME"
    value = "NO-VALUE"

    def __init__(self, name, value):
        self.name = name
        self.value = value
        return


class OfflineLOF:
    k = 0
    lpoints = list()
    distances = dict()
    nearest_neighbors = dict()
    k_dists = dict()
    lrds = dict()

    current_id = 0
    lower_threshold = ""
    upper_threshold = ""

    def __init__(self, k=10):
        self.k = k
        return

    def fit(self, xs, verbosity=LOF_TRAIN_VERBOSITY):
        if verbosity > 0:
            print("training {}".format(self.get_name()))
            print("generating lpoints...")
        for x in xs:
            self.lpoints.append(LofPoint(name=self.current_id, value=x))
            self.current_id += 1

        if verbosity > 0:
            print("generating properties...")
        for lpsrc in self.lpoints:
            if verbosity > 1:
                print("lpoint {} (out of {})...".format(lpsrc.name, len(self.lpoints)), COMMENT)

            nearest_neighbors = self.k_nearest_neighbors(o=lpsrc.value, k=self.k+1)
            nearest_neighbors.remove(lpsrc)

            self.nearest_neighbors[lpsrc.name] = nearest_neighbors
            self.k_dists[lpsrc.name] = self.distance(lpsrc.value, nearest_neighbors[self.k - 1].value)

            lrd = 0
            for neighbor in nearest_neighbors:
                reachabiliy_dist = max(self.distance(lpsrc.value, neighbor.value), self.k_dists[lpsrc.name])
                lrd += reachabiliy_dist
            lrd /= self.k
            self.lrds[lpsrc.name] = lrd

        print("calculating boundaries...")
        ps = sorted([self.predict_raw_known_single(lp) for lp in self.lpoints])
        print(ps)
        middle = int(len(ps)/2)
        self.lower_threshold = average(ps[:middle])
        self.upper_threshold = average(ps[middle:])
        print("lof set boundaries ({},{})".format(self.lower_threshold, self.upper_threshold))
        return

    @staticmethod
    def distance(p1, p2):
        if len(p1) != len(p2):
            print("warning: length of p1 != length of p2 ({} != {})".format(len(p1), len(p2)), WARNING)
        return sqrt(sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]))

    def reachability_dist(self, x, lp):
        d = self.distance(x, lp.value)
        k_d = self.k_dists[lp.name]
        return max([d, k_d])

    def k_nearest_neighbors(self, o, k):
        # algorithm runtime: O(k * n)
        # where n is the number of points in the training set
        #
        # (a naive algorithm will have a runtime of O(n * log(n))

        # all possible neighbors and their distances from 'o'
        left_d_ns = [[self.distance(lp.value, o), lp] for lp in self.lpoints]

        # a pool consisting of the k nearest neighbors of o at a given point
        k_nearest = sorted(left_d_ns[:k], key=lambda d_n: d_n[0])
        left_points = left_d_ns[k:]
        max_dist = k_nearest[k - 1][0]

        for d_n in left_d_ns:
            if d_n[0] < max_dist:
                k_nearest.remove(k_nearest[len(k_nearest) - 1])
                k_nearest.append(d_n)
                k_nearest = sorted(k_nearest, key=lambda d_n: d_n[0])
                max_dist = k_nearest[k - 1][0]

        # print([kn[0] for kn in k_nearest], OKBLUE)
        return [d_n[1] for d_n in k_nearest]

    def lrd(self, x):
        return self.k / sum([self.reachability_dist(x, neighbor) for neighbor in self.k_nearest_neighbors(x, self.k)])

    def lof_known(self, lp):
        return sum([self.lrds[n.name] for n in self.nearest_neighbors[lp.name]]) / (self.k * self.lrds[lp.name])

    def lof(self, x):
        return sum([self.lrds[n.name] for n in self.k_nearest_neighbors(x, self.k)]) / (self.k * self.lrd(x))

    def predict_raw_single(self, x):
        return self.lof(x)

    def predict_raw_known_single(self, lp):
        return self.lof_known(lp)

    def predict_raw(self, xs):
        return [self.predict_raw_single(x) for x in xs]

    def predict_single(self, x):
        p = self.predict_raw_single(x)
        # print(p, FAIL)
        if self.upper_threshold > p > self.lower_threshold:
            # print("p: {} E ({}, {})".format(p, self.lower_threshold, self.upper_threshold))
            return 0
        return 1

    def predict(self, xs):
        xs = list(xs)
        return [self.predict_single(x) for x in xs]

    def has_threshold(self):
        return False

    def get_name(self):
        return "OfflineLOF(k={})".format(self.k)

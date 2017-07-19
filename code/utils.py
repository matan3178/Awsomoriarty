from math import sqrt, log

from tables.idxutils import infinity

from code.log.Print import *


def flatten_list(lst):
    flat_list = list();

    for el in lst:
        flat_list.extend(el)

    return flat_list


def sum_vectors_binary(vec1, vec2):
    return [(v1 + v2) for v1, v2 in zip(vec1, vec2)]


def average_vectors(* vectors):
    denominator = len(vectors)
    v_average = list()
    for j in range(len(vectors[0])):
        values = [vector[j] for vector in vectors]
        v_average.append(sum(values))
        v_average = [value / denominator for value in v_average]
    # print("+{} = {}".format(vectors, v_average))

    return v_average


def mse(x_vector, y_vector):
    sum_of_squared_errors = 0
    for x, y in zip(x_vector, y_vector):
        sum_of_squared_errors += y**2 - x**2

    return sum_of_squared_errors / len(x_vector)


def mean(xs):
    return sum(xs)/len(xs)


def variance(xs, u=None):
    if u is None:
        u = mean(xs)
    return sum([(x - u)**2 for x in xs])/len(xs)


def standard_deviation(xs, u=None):
    return sqrt(variance(xs, u))


def entropy(l1_size, l2_size):
    if l2_size == 0:
        return infinity
    elif l1_size == 0:
        return 0

    n = l1_size + l2_size
    p1 = l1_size / n
    p2 = l1_size / n

    return (p1 * log(p1)) + (p2 * log(p2))
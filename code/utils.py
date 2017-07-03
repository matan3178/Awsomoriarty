from log.Print import *


def flatten_list(lst):
    flat_list = list();

    for el in lst:
        flat_list.extend(el)

    return flat_list


def sum_vectors_binary(vec1, vec2):
    return [(v1 + v2) for v1, v2 in zip(vec1, vec2)]


def sum_vectors(*vectors):
    return [sum(values) for values in [vectors[i] for i in range(len(vectors))]]

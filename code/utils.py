from log.Print import *


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

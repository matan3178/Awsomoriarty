import math

import numpy as np

from code._definitions import NUMBER_OF_REDUNDENT_LINES, NUMBER_OF_REDUNDENT_COLUMNS, VERBOSITY_general
from code.log.Print import *
from code.utils import average_vectors, flatten_list


def start(list_of_samples):
    return samples_to_np_arrays(
        normalize_feature_vector_to_unit_size(
            aggregate_samples_using_sliding_windows(
                string_to_float(
                    remove_null_rows(
                        remove_redundent_columns(list_of_samples)
                    )
                ),
                5,
                2
            )
        )
    )


# data manipulation


def samples_to_np_arrays(list_of_samples):
    return [np.array(sample) for sample in list_of_samples]


def string_to_float(list_of_samples):
    return [[np.float64(feature) for feature in sample] for sample in list_of_samples]


def remove_null_rows(list_of_samples, verbosity=VERBOSITY_general):
    null_counter = 0
    result = list()
    for sample in list_of_samples:
        if 'Null' in sample:
            null_counter += 1
        else:
            result.append(sample)
            if null_counter > 1 and verbosity > 1:
                print("Warning: {} nulls in a row".format(null_counter), WARNING)
            null_counter = 0
    return result


def remove_redundent_rows(list_of_samples):
    return [sample for sample in list_of_samples[NUMBER_OF_REDUNDENT_LINES:]]


def remove_redundent_columns(list_of_samples):
    return [sample[NUMBER_OF_REDUNDENT_COLUMNS:] for sample in list_of_samples]


def sliding_windows(list_of_samples, window_size, slide_size):
    return [list_of_samples[i:i + window_size] for i in range(0, len(list_of_samples) - window_size + slide_size, slide_size)]


def windows(list_of_samples, window_size):
    return sliding_windows(list_of_samples, window_size, window_size)


def flatten_windows(list_of_windows):
    return [flatten_list(sample_list) for sample_list in list_of_windows]


# feature extraction


def normalize_feature_vector_to_unit_size(list_of_samples):
    result = list()
    for lst in list_of_samples:
        v_size = math.sqrt(sum(x**2 for x in lst))
        result.append([x / v_size for x in lst])

    return result


def derivate_samples(list_of_samples):
    result = list()
    for i in range(len(list_of_samples) - 1):
        temp = list()
        for j in range(len(list_of_samples[i])):
            temp.append(list_of_samples[i + 1][j] - list_of_samples[i][j])
        result.append(temp)
    return result


def aggregate_samples_using_sliding_windows(list_of_samples, window_size, slide_size):
    return [average_vectors(* window) for window in sliding_windows(list_of_samples, window_size, slide_size)]


def aggregate_samples_using_windows(list_of_samples, window_size):
    return aggregate_samples_using_sliding_windows(list_of_samples, window_size, slide_size=window_size)


# testing


def run_feature_extraction_tests():
    # print(derivate_samples([[1, 2, 3], [-5, 5, 6], [7, 8, 9]]))
    # print(normalize_feature_vector_to_unit_size([[1, 2, 3], [-5, 5, 6], [7, 8, 9]]))
    # print(remove_null_rows([[1,2,3],[4,'null',6],[7,6,5],[1,'null','null'],[4,5,'null'],[1,2,3],[4,'null',6],[7,6,5],[1,'null','null']]));
    # print(remove_redundent_lines_and_rows([[1,2,3],[-5,5,6],[7,8,9]]))
    # print(string_to_float([['1','2.5','3'],['-5','5','6'],['7','8','9']]))
    print("aggregate: {} (expected {})".format(aggregate_samples_using_sliding_windows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 2, 2), [[5, 7, 9], [11, 13, 15], [17, 19, 21]]))

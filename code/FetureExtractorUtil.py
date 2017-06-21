from log.Print import *
from definitions import NUMBER_OF_REDUNDENT_LINES, NUMBER_OF_REDUNDENT_COLUMNS
import math
import numpy as np


def start(data):
    return samples_to_np_arrays(normalize_feature_vector_to_unit_size(string_to_float(remove_null_rows(remove_redundent_lines_and_rows(data)))))


def samples_to_np_arrays(data):
    return np.array([np.array(sample) for sample in data])


def string_to_float(data):
    return [[np.float32(feature) for feature in sample] for sample in data]


def remove_redundent_lines_and_rows(data):
    return [sample[NUMBER_OF_REDUNDENT_COLUMNS:] for sample in data[NUMBER_OF_REDUNDENT_LINES:]]


def remove_null_rows(list_of_lists):
    null_counter = 0
    result = list()
    for lst in list_of_lists:
        if 'NULL' in [str(x_string).upper() for x_string in lst]:
            null_counter += 1
        else:
            result.append(lst)
            if null_counter > 1:
                print("Warning: {} nulls in a row".format(null_counter) , WARNING)
            null_counter = 0
    return result


def normalize_feature_vector_to_unit_size(list_of_samples):
    result = list()
    for lst in list_of_samples:
        v_size = math.sqrt(sum(x**2 for x in lst))
        result.append([x / v_size for x in lst])

    return result


def derivation(lst):
    result = list()
    for i in range(len(lst)-1):
        temp = list()
        for j in range(len(lst[i])):
            temp.append(lst[i + 1][j] - lst[i][j])
        result.append(temp)
    return result


# print(derivation([[1,2,3],[-5,5,6],[7,8,9]]))
# print(normalize_feature_vector_to_unit_size([[1, 2, 3], [-5, 5, 6], [7, 8, 9]]))
# print(remove_null_rows([[1,2,3],[4,'null',6],[7,6,5],[1,'null','null'],[4,5,'null'],[1,2,3],[4,'null',6],[7,6,5],[1,'null','null']]));
# print(remove_redundent_lines_and_rows([[1,2,3],[-5,5,6],[7,8,9]]))
# print(string_to_float([['1','2.5','3'],['-5','5','6'],['7','8','9']]))
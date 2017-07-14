from math import log

from tables.idxutils import infinity

from code.log.Print import *


def calc_entropy(l1_size, l2_size):
    if l2_size == 0:
        return infinity
    elif l1_size == 0:
        return 0

    n = l1_size + l2_size
    p1 = l1_size / n
    p2 = l1_size / n

    return (p1 * log(p1)) + (p2 * log(p2))


def information_gain_using_split_value(l1_data, l2_data, feature_index, split_value):
    l1_split = [[sample for sample in l1_data if sample[feature_index] > split_value],
                [sample for sample in l1_data if sample[feature_index] <= split_value]]
    l2_split = [[sample for sample in l2_data if sample[feature_index] > split_value],
                [sample for sample in l2_data if sample[feature_index] <= split_value]]

    len1 = len(l1_data)
    len2 = len(l2_data)
    len_split10 = len(l1_split[0])
    len_split11 = len(l1_split[1])
    len_split20 = len(l2_split[0])
    len_split21 = len(l2_split[1])

    return calc_entropy(len1, len2) - calc_entropy(len_split10, len_split11) - calc_entropy(len_split20, len_split21)


def split_information_gain(l1_data, l2_data, feature_index, number_of_steps=100):
    split_min = max([min([sample[feature_index] for sample in l1_data]),
                     min([sample[feature_index] for sample in l1_data])])
    split_max = min([max([sample[feature_index] for sample in l1_data]),
                     max([sample[feature_index] for sample in l1_data])])

    step_size = (split_max - split_min) / number_of_steps
    info_gains = [information_gain_using_split_value(l1_data, l2_data, feature_index, split_value)
                  for split_value in [split_min + step_size * i for i in range(number_of_steps)]]
    return max(info_gains)

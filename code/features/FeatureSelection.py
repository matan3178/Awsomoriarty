from code.features.FetureExtractorUtil import get_column
from code.log.Print import *
from code.utils import mean, standard_deviation, entropy


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

    return entropy(len1, len2) - entropy(len_split10, len_split11) - entropy(len_split20, len_split21)


def split_information_gain(l1_data, l2_data, feature_index, number_of_steps=100):
    split_min = max([min([sample[feature_index] for sample in l1_data]),
                     min([sample[feature_index] for sample in l1_data])])
    split_max = min([max([sample[feature_index] for sample in l1_data]),
                     max([sample[feature_index] for sample in l1_data])])

    step_size = (split_max - split_min) / number_of_steps
    info_gains = [information_gain_using_split_value(l1_data, l2_data, feature_index, split_value)
                  for split_value in [split_min + step_size * i for i in range(number_of_steps)]]
    return max(info_gains)


def fisher_score(label1_data_column, label2_data_column):
    u1 = mean(label1_data_column)
    u2 = mean(label2_data_column)
    s1 = standard_deviation(label1_data_column, u1)
    s2 = standard_deviation(label2_data_column, u2)
    return abs(u1 - u2) / s1 + s2


def select_k_best_features(k, label1_data, label2_data):
    num_of_features = len(label1_data[0])
    scores = [[i, fisher_score(get_column(label1_data, i), get_column(label2_data, i))] for i in range(num_of_features)]
    scores = sorted(scores, key=lambda tuple: tuple[1], reverse=True)
    print("scores: {}".format(scores), OKBLUE)
    selected_features = [tuple[0] for tuple in scores[:k]]
    print("selected: {}".format(selected_features), OKBLUE + BOLD)
    return selected_features

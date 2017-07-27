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
    if len(label1_data_column) == 0 or len(label2_data_column) == 0:
        return 0
    u1 = mean(label1_data_column)
    u2 = mean(label2_data_column)
    s1 = standard_deviation(label1_data_column, u1)
    s2 = standard_deviation(label2_data_column, u2)
    return abs(u1 - u2) / s1 + s2


# takes a list of features and the data, and calculates the fisher-score for each feature based on the data
# returns a list of tuples: <feature, rank>
def fisher_score_feature_rankings(features, label1_data, label2_data):
    f_ss = [[f, fisher_score(get_column(label1_data, f), get_column(label2_data, f))] for f in features]
    return f_ss


# takes a list of tuples: <feature, rank>
# returns a sorted list
def sort_features_by_ranking(f_rs):
    return [f_r[0] for f_r in sorted(f_rs, key=lambda f_r: f_r[1], reverse=True)]


def choose_best_split_value_based_on_mean_values(feature, l1_data, l2_data):
    l1mean = mean(get_column(list_of_samples=l1_data, column_index=feature))
    l2mean = mean(get_column(list_of_samples=l2_data, column_index=feature))
    return (l1mean + l2mean) / 2


def select_k_best_features(k, label1_data, label2_data):
    num_of_features = len(label1_data[0])
    chosen_features = list()
    remaining_features = list(range(num_of_features))

    test_data_pairs = list()
    test_data_pairs.append([label1_data, label2_data])
    for i in range(k):
        # choose best feature:
        #   calculate score for each feature for each test-data-pair
        #   average score over different pairs
        #   sort final scores based on average scores (?) and choose the best feature

        feature_ranking_mapping = dict()    # feature rankings (stays true through the iteration)
        for f in remaining_features:
            feature_ranking_mapping[f] = 0  # initialize rankings with 0

        # update ranking using all the data pairs
        for data_pair in test_data_pairs:
            f_rs = fisher_score_feature_rankings(remaining_features, data_pair[0], data_pair[1])
            for f_r in f_rs:
                feature_ranking_mapping[f_r[0]] += f_r[1]
        for f in remaining_features:
            feature_ranking_mapping[f] /= len(remaining_features)

        features_sorted = sort_features_by_ranking([[key, feature_ranking_mapping[key]] for key in feature_ranking_mapping.keys()])
        best_feature = features_sorted[0]
        chosen_features.append(best_feature)
        remaining_features.remove(best_feature)

        # reshape data:
        #   split data-pairs for chosen feature's value
        #   collect a list of split data-pairs
        #   replace old data-pairs list with new one
        new_test_data_pairs = list()
        for data_pair in test_data_pairs:
            if len(data_pair[0]) == 0 or len(data_pair[1]) == 0:  # don't create more data-pairs originating from this pair
                continue

            chosen_split_value = choose_best_split_value_based_on_mean_values(best_feature, data_pair[0], data_pair[1])
            l1_upper, l1_lower, l2_upper, l2_lower = list(), list(), list(), list()
            for l1sample, l2sample in zip(data_pair[0], data_pair[1]):
                if l1sample[best_feature] > chosen_split_value:
                    l1_upper.append(l1sample)
                else:
                    l1_lower.append(l1sample)
                if l2sample[best_feature] > chosen_split_value:
                    l2_upper.append(l1sample)
                else:
                    l2_lower.append(l1sample)

            l1_upper = l1_upper

            new_test_data_pairs.append([l1_upper, l2_upper])
            new_test_data_pairs.append([l1_lower, l2_lower])
        test_data_pairs = new_test_data_pairs

    return chosen_features

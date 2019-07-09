import numpy as np
import pandas as pd
import math
from collections import Counter
import sys
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class Tree(object):
    def __init__(self):
        self.attribute = None
        self.threshold = None
        self.classification = []
        self.left = None
        self.right = None


def balance_data_set(credit_card_data):
    # get number of examples of fraud class
    num_fraud_class = len(credit_card_data[credit_card_data.Class == 1])

    # get indices of fraud data samples
    fraud_indices = np.array(credit_card_data[credit_card_data.Class == 1].index)

    # get number of examples of authentic transaction class
    not_fraud_indices = credit_card_data[credit_card_data.Class == 0].index
    random_not_fraud_indices = np.random.choice(not_fraud_indices, num_fraud_class, replace=False)
    random_not_fraud_indices = np.array(random_not_fraud_indices)
    under_sample_indices = np.concatenate([fraud_indices, random_not_fraud_indices])

    balanced_credit_card_data = credit_card_data.iloc[under_sample_indices, :]

    # print("Percentage of authentic transactions: ",
    #       len(balanced_credit_card_data[balanced_credit_card_data.Class == 0]) / len(balanced_credit_card_data))
    # print("Percentage of fraud transactions: ",
    #       len(balanced_credit_card_data[balanced_credit_card_data.Class == 1]) / len(balanced_credit_card_data))
    # print("Total number of transactions in re-sampled data: ", len(balanced_credit_card_data))
    return balanced_credit_card_data


def normalize_feature(feature, range_val):
    feature_min = feature.min()
    feature_max = feature.max()
    # feature_mean = feature.mean()
    # print(feature_min, feature_max, feature_mean)

    # rescale feature between a-b
    feature = ((feature - feature_min)*(range_val[1] - range_val[0])/(feature_max-feature_min))+range_val[0]
    return feature


def split_train_test_data(credit_card_data, train_split_percentage):
    # Shuffle the data set
    credit_card_data = credit_card_data.sample(frac=1).reset_index(drop=True)

    # calculate train and test data lengths based on percentage
    train_data_len = int(credit_card_data.shape[0]*(train_split_percentage/100))
    test_data_len = credit_card_data.shape[0] - train_data_len

    # Split the data
    train_credit_card_data = credit_card_data[:train_data_len]
    test_credit_card_data = credit_card_data[train_data_len:]
    return train_credit_card_data, test_credit_card_data


def calc_num_of_classes(credit_card_data, class_column):
    # get classes
    classes_lst = credit_card_data[class_column].unique()
    num_of_classes = len(classes_lst)
    return num_of_classes, classes_lst


def check_same_class(credit_card_data):
    # get count of examples having class 1 (fraud)
    num_records_class_1 = len(credit_card_data[credit_card_data.Class == 1])
    # if all are class 1 or None have class 1, that means all examples have same class either 0 or 1
    if (num_records_class_1 == len(credit_card_data['Class'])) or num_records_class_1 == 0:
        return 1
    else:
        return 0


def calc_prob_distribution(examples):
    prob_distribution = []
    total_ex = len(examples['Class'])
    prob_distribution.append(len(examples[examples.Class == 0])/total_ex)
    prob_distribution.append(len(examples[examples.Class == 1])/ total_ex)

    return prob_distribution


def calc_entropy(node):
    total_ex = len(node)
    class0 = 0
    class1 = 0
    for class_num in node:
        if class_num == 0:
            class0 += 1
        elif class_num == 1:
            class1 += 1
    entropy = 0
    if class0 > 0:
        entropy += (-1 * ((class0 / total_ex) * (math.log2(class0 / total_ex))))
    if class1 > 0:
        entropy += (-1 * ((class1 / total_ex) * (math.log2(class1 / total_ex))))

    return entropy


def calc_information_gain(credit_card_data, feature_data, threshold):
    left_subtree = []
    right_subtree = []
    root = list(credit_card_data['Class'])
    # divide data into left and right child based on current threshold
    for index, val in zip(feature_data.index, feature_data.values):
        if val < threshold:
            left_subtree.append(credit_card_data.at[index, 'Class'])
        elif val >= threshold:
            right_subtree.append(credit_card_data.at[index, 'Class'])
    information_gain = calc_entropy(root)
    # print(informationGain)

    entropy = calc_entropy(left_subtree)
    information_gain -= ((len(left_subtree) / len(feature_data)) * entropy)
    # print(informationGain)

    entropy = calc_entropy(right_subtree)
    information_gain -= ((len(right_subtree) / len(feature_data)) * entropy)
    # print(informationGain)
    # print(leftSubtree)
    # print(rightSubtree)
    return information_gain


def get_best_attribute(credit_card_data, features_lst):
    max_gain = -1
    best_attribute = -1
    best_threshold = -1
    for feature in features_lst:
        feature_data = credit_card_data[feature]
        max_val = feature_data.max()
        min_val = feature_data.min()
        # Check for 50 different thresholds
        for k in range(1, 51, 1):
            threshold = min_val + k * (max_val - min_val) / 51
            gain = calc_information_gain(credit_card_data, feature_data, threshold)
            if gain > max_gain:
                max_gain = gain
                best_attribute = feature
                best_threshold = threshold
    return best_attribute, best_threshold


def dtl(tree_node, credit_card_data, prob_distribution, features_lst):
    if credit_card_data.shape[0] == 0:
        # No examples left
        tree_node.classification = list(prob_distribution)
        return tree_node
    elif check_same_class(credit_card_data):
        # all examples have same class
        prob_distribution = calc_prob_distribution(credit_card_data)
        tree_node.classification = list(prob_distribution)
        return tree_node
    else:
        # Choose best attribute for current node
        best_attribute, best_threshold = get_best_attribute(credit_card_data, features_lst)

        tree_node.left = Tree()
        tree_node.right = Tree()
        tree_node.attribute = best_attribute
        tree_node.threshold = best_threshold

        # Calculate probability distribution at current node
        prob_distribution = calc_prob_distribution(credit_card_data)
        tree_node.classification = list(prob_distribution)
        left_examples = pd.DataFrame()
        right_examples = pd.DataFrame()

        # divide examples available at current node
        for index, val in zip(credit_card_data[best_attribute].index, credit_card_data[best_attribute].values):
            if val < best_threshold:
                left_examples = left_examples.append(credit_card_data.loc[index], ignore_index=False)
            elif val >= best_threshold:
                right_examples = right_examples.append(credit_card_data.loc[index], ignore_index=False)
        tree_node.left = dtl(tree_node.left, left_examples, prob_distribution, features_lst)
        tree_node.right = dtl(tree_node.right, right_examples, prob_distribution, features_lst)
        return tree_node


def recursive_classify(tree_node, row_data):
    data_val = row_data[tree_node.attribute]
    if data_val < tree_node.threshold:
        if tree_node.left is None:
            return tree_node.classification
        if tree_node.left.attribute is None:
            if tree_node.left.classification is None:
                return tree_node.classification
            else:
                return tree_node.left.classification
        prob_distribution = recursive_classify(tree_node.left, row_data)
    else:
        if tree_node.right is None:
            return tree_node.classification
        if tree_node.right.attribute is None:
            if tree_node.right.classification is None:
                return tree_node.classification
            else:
                return tree_node.right.classification
        prob_distribution = recursive_classify(tree_node.right, row_data)

    return prob_distribution


def classify_using_dtl(tree_node, test_credit_card_data):
    classified_data = []
    for row in test_credit_card_data.iterrows():
        row_data = row[1]
        prob_distribution = recursive_classify(tree_node, row_data)
        if prob_distribution[0] >= 0.5:
            classified_data.append(0)
        else:
            classified_data.append(1)
    return classified_data


def main():
    try:
        data_filename = sys.argv[1]
        num_of_trees = sys.argv[2]
        train_split_percentage = sys.argv[3]
    except:
        data_filename = 'creditcard.csv'
        num_of_trees = 5
        train_split_percentage = 80

    credit_card_data = pd.read_csv(data_filename, sep=",")

    # check class distribution
    fraud_count = len(credit_card_data[credit_card_data.Class == 1])
    authentic_count = len(credit_card_data[credit_card_data.Class == 0])
    plt.bar([0, 1], [authentic_count, fraud_count], align='center', width=0.2)
    plt.xticks([0, 1], ('Authentic', 'Fraud'))
    plt.ylabel('Number of data samples')
    plt.title('Class Distribution')
    plt.show()

    # print(credit_card_data.shape)
    credit_card_data['Amount'] = normalize_feature(credit_card_data['Amount'], [0, 1])
    credit_card_data['Time'] = normalize_feature(credit_card_data['Time'], [0, 1])

    for i in range(0, 11):
        print("------------------Iteration ", i, "---------------------------")
        # split the data between test and
        train_credit_card_data, test_credit_card_data = split_train_test_data(credit_card_data, train_split_percentage)
        print(train_credit_card_data.shape, test_credit_card_data.shape)
        # Balance the data set, under-sampling based on number of examples of fraud
        train_credit_card_data = balance_data_set(train_credit_card_data)

        # credit_card_data_target = credit_card_data.ix[:, train_credit_card_data.columns == 'Class']
        # credit_card_data = credit_card_data.ix[:, credit_card_data.columns != 'Class']

        # get total feature list, and remove Class from it because don't want it while creating decision tree
        features_lst = list(train_credit_card_data.columns)
        features_lst.remove('Class')

        prob_distribution = []
        forest = list()

        # Divide train data set into multiple data sets for forest (different trees)
        size_of_data_sets = int(train_credit_card_data.shape[0]/num_of_trees)
        first_data_set = size_of_data_sets + train_credit_card_data.shape[0] - (size_of_data_sets*num_of_trees)
        end_index = first_data_set
        start_index = 0

        # Shuffle train data
        train_credit_card_data = train_credit_card_data.sample(frac=1).reset_index(drop=True)

        # build forest
        for tree_num in range(0, num_of_trees, 1):
            # print("Percentage of normal transactions: ",
            #       len(train_credit_card_data[start_index:end_index][train_credit_card_data[start_index:end_index].Class == 0]) / len(train_credit_card_data[start_index:end_index]))
            # print("Percentage of fraud transactions: ",
            #       len(train_credit_card_data[start_index:end_index][train_credit_card_data[start_index:end_index].Class == 1]) / len(train_credit_card_data[start_index:end_index]))
            # print("Total number of transactions in resampled data: ", len(train_credit_card_data[start_index:end_index]))
            root_dtl = Tree()
            root_dtl = dtl(root_dtl, train_credit_card_data[start_index:end_index], prob_distribution, features_lst)
            start_index = end_index
            end_index += size_of_data_sets
            forest.append(root_dtl)
            del root_dtl
        # print(forest)

        # extract target classes
        test_data_target = np.array(test_credit_card_data.ix[:, test_credit_card_data.columns == 'Class'])
        # Predict for test data point with created forest
        classification_list = []
        for tree_num in forest:
            classification_list.append(classify_using_dtl(tree_num, test_credit_card_data))
        correct_decision = 0

        final_classification = list()
        # check prediction given by every tree for a data point, and take vote
        test_data_indices = test_credit_card_data.index
        for index in range(0, len(classification_list[0]), 1):
            democracy = []
            for classification_num in classification_list:
                democracy.append(classification_num[index])
            count = Counter(democracy)
            class_num = count.most_common()[0][0]
            final_classification.append(class_num)
            if class_num == test_credit_card_data.at[test_data_indices[index], 'Class']:
                correct_decision += 1
        final_classification = np.array(final_classification)
        # Compute confusion matrix
        conf_matrix = confusion_matrix(test_data_target, final_classification)

        print("confusion matrix: \n", conf_matrix)
        print("Recall in test data: ", conf_matrix[1, 1]/(conf_matrix[1, 0]+conf_matrix[1, 1]))
        accuracy = (correct_decision / test_credit_card_data.shape[0])
        print("Accuracy: ", accuracy)
    return


if __name__ == '__main__':
    main()

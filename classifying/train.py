#! /usr/bin/env python
from tqdm import tqdm
tqdm.pandas()
import string
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from typing import List
from utils.save_and_load import load_obj


def check_two(word: str) -> bool:
    """
    Check the final matches for matches that are 2-gram matches
    :param word:
    :return:
    """
    splits = word.split()
    if len(splits) == 2:
        return True
    else: return False

def get_features(two_grams: str) -> List[int]:
    """
    Function to get features from the two grams
    :param two_grams: two gram string to obtain features from
    :return: List of integers representing features
    """
    feature_list = []

    # Split the word into two
    splits = two_grams.split()

    # Upper case in whole word, first word, second word
    feature_list.append(sum(c.isupper() for c in two_grams))
    feature_list.append(sum(c.isupper() for c in splits[0]))
    feature_list.append(sum(c.isupper() for c in splits[1]))

    # Upper cases normalized
    feature_list.append(sum(c.isupper() for c in two_grams) / len(two_grams))
    feature_list.append(sum(c.isupper() for c in splits[0]) / len(splits[0]))
    feature_list.append(sum(c.isupper() for c in splits[1]) / len(splits[1]))

    # Lower case in whole word, first word, second word
    feature_list.append(sum(c.islower() for c in two_grams))
    feature_list.append(sum(c.islower() for c in splits[0]))
    feature_list.append(sum(c.islower() for c in splits[1]))

    # Lower case normalized in whole word, first word, second word
    feature_list.append(sum(c.islower() for c in two_grams) / len(two_grams))
    feature_list.append(sum(c.islower() for c in splits[0]) / len(splits[0]))
    feature_list.append(sum(c.islower() for c in splits[1]) / len(splits[1]))

    # Dots in whole word, first word, second word
    feature_list.append(sum(c == '.'for c in two_grams))
    feature_list.append(sum(c == '.' for c in splits[0]))
    feature_list.append(sum(c == '.' for c in splits[1]))

    #  in whole word, first word, second word
    feature_list.append(sum(c == '.' for c in two_grams))
    feature_list.append(sum(c == '.' for c in splits[0]))
    feature_list.append(sum(c == '.' for c in splits[1]))

    # General punctuation in first and second word
    feature_list.append(sum(c in string.punctuation for c in two_grams))
    feature_list.append(sum(c in string.punctuation for c in splits[0]))
    feature_list.append(sum(c in string.punctuation for c in splits[1]))

    # - whole word, first word, second word
    feature_list.append(sum(c == '-' for c in two_grams))
    feature_list.append(sum(c == '-' for c in two_grams) / len(two_grams))  #  Mulder-Derksen Lange namen lage ratio

    # Len word
    feature_list.append(len(two_grams))
    feature_list.append(len(splits[0]))
    feature_list.append(len(splits[1]))

    # Normalized length
    feature_list.append(len(splits[0]) / len(two_grams))
    feature_list.append(len(splits[1]) / len(two_grams))

    return feature_list


def check_potential_of_ngram(n_gram: str) -> bool:
    """
    Inputs a string and checks if there are:
    at least two capital letters present
    at least one lowercase
    at least 3 characters
    at max 25 characters

    :param n_gram: n_gram to check for potential
    :return: true if all conditions are met
    """

    rules = [lambda s: sum(x.isupper() for x in s) >= 2,    # must have at least two uppercase
             lambda s: len(s) >= 5,                         # must be at least 5 characters
             lambda s: len(s) <= 32,                        # must be at max 32 characters
             lambda s: sum(x.isdigit() for x in s) <= 2,    # no more than 2 digits
             ]

    return all(rule(n_gram) for rule in rules)


# number_of_negative_matches_to_train = 12 000
def generate_train_test_split(data, number_of_negative_matches_to_train=2500):
    """
    :param data:
    :param number_of_negative_matches_to_train: number_of_negative_matches_to_train: integer to control amount of negative examples to start with
    :return:
    """

    # Only keep two gram matches from the column that contains all thresholded matches
    data['best_match_2_grams'] = data['best_match_all_cleaned'].progress_apply(
        lambda matches: [match for match in matches if check_two(match[2])])

    print("Length before deleting empty match rows", len(data))
    data = data[data['best_match_2_grams'].map(lambda d: len(d) > 0)]
    print("Length after deleting empty match rows", len(data))

    match_2_grams = data['best_match_2_grams'].progress_apply(lambda matches:[match[2] for match in matches]).tolist()
    flat_list_match = [item for sublist in match_2_grams for item in sublist]

    # take all two grams put them into a list and check them for being a potential name
    non_match_2_grams = data['n_gram_2'].progress_apply(lambda matches: [match[2] for match in matches]).tolist()
    non_match_2_grams = [two_gram for two_gram in non_match_2_grams if check_potential_of_ngram(two_gram)]

    # Negative matches should not be in the positive two-grams list
    non_match_2_grams = [two_gram for two_gram in non_match_2_grams if two_gram not in flat_list_match]

    flat_list_non_match = [item for sublist in non_match_2_grams for item in sublist]
    flat_list_non_match = flat_list_non_match[:len(flat_list_match)+500]

    x_train_zero = [get_features(n_gram) for n_gram in flat_list_non_match]
    y_train_zero = len(flat_list_non_match) * [0]

    x_train_one = [get_features(n_gram) for n_gram in flat_list_match]
    y_train_one = len(flat_list_match) * [1]

    X = x_train_zero + x_train_one
    y = y_train_zero + y_train_one

    return train_test_split(X, y, test_size=0.33, random_state=42)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_conf_matrix(y_pred: List[int], y_test: List[int], class_names = ["Name", 'No-name']):
    """
    :param y_pred: list of predictions from x_test
    :param y_test: list of true values y_test
    :param class_names: axis names for confusion matrix
    :return:
    """

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title='Confusion matrix')
    plt.savefig("../data/confusion_matrix.png")
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig("../data/confusion_matrix_normalized.png")
    plt.show()


def plot_f1_curve(y_test: List[int], y_pred_prob: List[int]):
    """
    Saves a F1 curve in the matching folder
    :param y_test: list of true y values
    :param y_pred_prob: predicted probabilities
    :return:
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.savefig("../data/2_class_precision_recall.png")
    plt.show()

def train(data_file: str):
    """
    :param data_file: name of datafile to load pd.DataFrame from the ./data folder
    :return:
    """
    data = load_obj(file_name=data_file)

    X_train, X_test, y_train, y_test = generate_train_test_split(data)

    classifier = svm.SVC(kernel='rbf', C=1)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # Now predict with probabilities
    classifier.probability = True
    y_pred_prob = classifier.predict(X_test)

    plot_conf_matrix(y_pred=y_pred, y_test=y_test)
    plot_f1_curve(y_pred_prob=y_pred_prob, y_test=y_test)

if __name__ == '__main__':
    train('12-12-match')

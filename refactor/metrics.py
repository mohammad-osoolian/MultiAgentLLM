import pandas as pd
import numpy as np
from typing import List


def accuracy(labels, predicts):
    cnt = 0
    for i in range(len(labels)):
        if labels[i] == predicts[i]:
            cnt += 1
    return cnt / len(labels)

def list2onehot(lst, n_labels):
    # lst = list(map(int, lst.split(',')))
    vector = np.zeros(shape=(n_labels,))
    for l in lst:
        vector[l - 1] = 1
    return vector

def multi_label_acc(labels, predictions, n_labels):
    result = []
    for label, prediction in zip(labels, predictions):
        gt_onehot = list2onehot(list(map(int, label.split(','))), n_labels)
        pred_onehot = list2onehot(prediction, n_labels)
        dist = np.abs(gt_onehot - pred_onehot).sum()
        result.append(1.0 - dist/n_labels)
    return sum(result) / len(result)


def agreement_rate(predicts):
    if type(predicts[0]) == list:
        first = predicts[0]
        return all(p == first for p in predicts)
    else:
        return len(set(predicts)) == 1


def error_rate(errors):
    return errors.count(1)/len(errors)


def get_intersection(agents_prediction):
    first = agents_prediction[0]
    intersect = set(first)
    for p in agents_prediction:
        intersect = intersect.intersection(set(p))
    return intersect


def common_results(predictions: List[List], labels: List = None) -> float:
    cnt = 0
    if type(predictions[0][0]) != list:
        predictions = np.array(predictions).reshape(len(predictions), len(predictions[0], 1))
    for i in range(len(predictions[0])):
        agents_prediction = [p[i] for p in predictions]
        intersect = get_intersection(agents_prediction)
        if sum([len(x) for x in agents_prediction]):
            cnt += (len(intersect) * len(predictions)) / sum([len(x) for x in agents_prediction])
    return cnt / len(predictions[0])


def common_mistakes(predictions: List[List], labels: List) -> float:
    cnt = 0
    if type(predictions[0][0]) != list:
        predictions = np.array(predictions).reshape(len(predictions), len(predictions[0], 1))
    labels = [list(map(int, l.split(','))) for l in labels]
    # labels = np.array(labels).reshape(len(labels), 1)
    for i in range(len(labels)):
        agents_prediction = [p[i] for p in predictions]
        intersect = get_intersection(agents_prediction)
        correct_predictions = intersect.intersection(set(labels[i]))
        cnt += (len(labels[i]) - len(correct_predictions)) / (len(labels[i]))
    return cnt / len(labels)


def at_least_one_correct(predictions: List[List], labels: List) -> float:
    cnt = 0
    for i in range(len(labels)):
        if any(p[i] == labels[i] for p in predictions):
            cnt += 1
    return cnt / len(labels)

def uncommon_with_at_least_one_correct(predictions: List[List], labels: List) -> float:
    cnt = 0
    for i in range(len(labels)):
        first = predictions[0][i]
        all_same = all(p[i] == first for p in predictions)
        any_correct = any(p[i] == labels[i] for p in predictions)
        if not all_same and any_correct:
            cnt += 1
    return cnt / len(labels)


def analysis_debate_potential(predictions, labels):
    analysis = {}
    analysis['common_answers'] = common_results(predictions)
    analysis['at_least_one_correct'] = at_least_one_correct(predictions, labels)
    analysis['common_mistakes'] = common_mistakes(predictions, labels)
    analysis['uncommon_with_at_least_one_correct'] = uncommon_with_at_least_one_correct(predictions, labels)
    return analysis
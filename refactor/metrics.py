import pandas as pd
import numpy as np


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

def common_results(p1, p2, p3):
    cnt = 0
    for i in range(len(p1)):
        if p1[i] == p2[i] and p2[i] == p3[i]:
            cnt += 1
    return cnt / len(p1)

def common_mistakes(p1, p2, p3, labels):
    cnt = 0
    for i in range(len(p1)):
        if p1[i] == p2[i] and p2[i] == p3[i] and p3[i] != labels[i]:
            cnt += 1
    return cnt / len(p1)

def at_least_one_correct(p1, p2, p3, labels):
    cnt = 0
    for i in range(len(p1)):
        if (p1[i] == labels[i] or p2[i] == labels[i] or p3[i] == labels[i]):
            cnt += 1
    return cnt / len(p1)

def uncommon_with_at_least_one_correct(p1, p2, p3, labels):
    cnt = 0
    for i in range(len(p1)):
        if not(p1[i] == p2[i] and p2[i] == p3[i]) and (p1[i] == labels[i] or p2[i] == labels[i] or p3[i] == labels[i]):
            cnt += 1
    return cnt / len(p1)

def analysis_debate_potential(p1, p2, p3, labels):
    print('Agent1 Accuracy:', accuracy(p1, labels))
    print('Agent2 Accuracy:', accuracy(p2, labels))
    print('Agent3 Accuracy:', accuracy(p3, labels))

    print('Common Answers:', common_results(p1, p2, p3))
    print('Common Mistakes:', common_mistakes(p1, p2, p3, labels))
    print('At Least One Correct:', at_least_one_correct(p1, p2, p3, labels))
    print('Uncommon with at Least One Correct:', uncommon_with_at_least_one_correct(p1, p2, p3, labels))
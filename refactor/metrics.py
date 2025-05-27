import pandas as pd

def accuracy(labels, predicts):
    cnt = 0
    for i in range(len(labels)):
        if labels[i] == predicts[i]:
            cnt += 1
    return cnt / len(labels)

def error_rate(errors):
    return errors.count(1)

from typing import List

def common_results(predictions: List[List], labels: List = None) -> float:
    cnt = 0
    for i in range(len(predictions[0])):
        first = predictions[0][i]
        if all(p[i] == first for p in predictions):
            cnt += 1
    return cnt / len(predictions[0])

def common_mistakes(predictions: List[List], labels: List) -> float:
    cnt = 0
    for i in range(len(labels)):
        first = predictions[0][i]
        if all(p[i] == first for p in predictions) and first != labels[i]:
            cnt += 1
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
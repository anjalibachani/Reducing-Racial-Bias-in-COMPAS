from utils import *
import copy
import numpy as np
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: Accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""


def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}

    keyList = sorted(categorical_results.keys())
    temp_data = dict.fromkeys(categorical_results.keys())
    thres = dict.fromkeys(categorical_results.keys())
    prob = dict.fromkeys(categorical_results.keys())

    acc = 0
    for i,group in enumerate(keyList):
        prev_acc = 0
        for t1 in np.arange(0,1,0.01):
            temp_data[group] = apply_threshold(categorical_results[group], t1)
            num_positive_predictions = get_num_predicted_positives(temp_data[group])
            prob[group] = num_positive_predictions / len(temp_data[group])
            for j in [x for x in range(4) if x != i]:
                for t2 in np.arange(0,1,0.01):
                    temp_data[keyList[j]] = apply_threshold(categorical_results[keyList[j]], t2)
                    num_positive_predictions = get_num_predicted_positives(temp_data[keyList[j]])
                    prob[keyList[j]] = num_positive_predictions / len(temp_data[keyList[j]])
                    if abs(prob[keyList[j]]-prob[group])<=epsilon:
                        thres[group] = t1
                        thres[keyList[j]] = t2
                        break
            current_acc = get_total_accuracy(temp_data)
            if current_acc<prev_acc and abs(current_acc - prev_acc)>0.004:
                break
            prev_acc = current_acc
            acc = max(acc,current_acc)
            if acc == current_acc:
                thresholds = copy.deepcopy(thres)
                demographic_parity_data = copy.deepcopy(temp_data)

    return demographic_parity_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""


def enforce_equal_opportunity(categorical_results, epsilon):
    thresholds = {}
    equal_opportunity_data = {}

    keyList = sorted(categorical_results.keys())
    temp_data = dict.fromkeys(categorical_results.keys())
    thres = dict.fromkeys(categorical_results.keys())
    prob = dict.fromkeys(categorical_results.keys())

    acc = 0
    for i, group in enumerate(keyList):
        prev_acc = 0
        for t1 in np.arange(0, 1, 0.01):
            temp_data[group] = apply_threshold(categorical_results[group], t1)
            prob[group] = get_true_positive_rate(temp_data[group])
            for j in [x for x in range(4) if x != i]:
                for t2 in np.arange(0, 1, 0.01):
                    temp_data[keyList[j]] = apply_threshold(categorical_results[keyList[j]], t2)
                    prob[keyList[j]] = get_true_positive_rate(temp_data[keyList[j]])
                    if abs(prob[keyList[j]] - prob[group]) <= epsilon:
                        thres[group] = t1
                        thres[keyList[j]] = t2
                        break
            current_acc = get_total_accuracy(temp_data)
            if current_acc<prev_acc and abs(current_acc - prev_acc)>0.002:
                break
            prev_acc = current_acc
            acc = max(acc, current_acc)
            if acc == current_acc:
                thresholds = copy.deepcopy(thres)
                equal_opportunity_data = copy.deepcopy(temp_data)

    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def make_thresholds():
    attempts = []
    for i in range(0,100):
        attempts.append(i/100)
    return attempts

def find_optima(pred_labels):
    N = len(pred_labels)
    current_maxima = 0.0
    current_threshold = 0
    values = make_thresholds()
    for value in values:
        thresholded_predictions = apply_threshold(pred_labels, value)
        accuracy = get_num_correct(thresholded_predictions)/N
        if accuracy >= current_maxima:
            current_maxima = accuracy
            current_threshold = value

    return [current_threshold, current_maxima]


def remake_data(results, thresholds):
    modified = {}
    for k in results:
        modified[k] = apply_threshold(results[k], thresholds[k])
    return modified


def enforce_maximum_profit(categorical_results):
    mp_data = {}
    accuracies = {}
    thresholds = {}
    for k in categorical_results.keys():
        optima = find_optima(categorical_results[k])
        thresholds[k] = optima[0]
        accuracies[k] = optima[1]

    mp_data = remake_data(categorical_results, thresholds)

    return mp_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    keyList = sorted(categorical_results.keys())
    temp_data = dict.fromkeys(categorical_results.keys())
    thres = dict.fromkeys(categorical_results.keys())
    ppv = dict.fromkeys(categorical_results.keys())

    acc = 0
    for i,group in enumerate(keyList):
        prev_acc = 0
        for t1 in np.arange(0,1,0.01):
            temp_data[group] = apply_threshold(categorical_results[group], t1)
            ppv[group] = get_positive_predictive_value(temp_data[group])
            for j in [x for x in range(4) if x != i]:
                for t2 in np.arange(0,1,0.01):
                    temp_data[keyList[j]] = apply_threshold(categorical_results[keyList[j]], t2)
                    ppv[keyList[j]] = get_positive_predictive_value(temp_data[keyList[j]])
                    if abs(ppv[keyList[j]]-ppv[group])<=epsilon:
                        thres[group] = t1
                        thres[keyList[j]] = t2
                        break
            current_acc = get_total_accuracy(temp_data)
            if current_acc<prev_acc and abs(current_acc - prev_acc)>0.002:
                break
            prev_acc = current_acc
            acc = max(acc,current_acc)
            if acc == current_acc:
                thresholds = copy.deepcopy(thres)
                predictive_parity_data = copy.deepcopy(temp_data)

    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    thresholds = {}
    pred_labels = []
    for k,v in categorical_results.items():
        pred_labels += v

    threshold, maxima = find_optima(pred_labels)

    for k,v in categorical_results.items():
        thresholds[k] = threshold

    single_threshold_data = remake_data(categorical_results, thresholds)

    return single_threshold_data, thresholds


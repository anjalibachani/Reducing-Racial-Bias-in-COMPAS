from sklearn.naive_bayes import MultinomialNB
from Preprocessing import preprocess
from Postprocessing import *
from utils import *
from datetime import datetime

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

begin = datetime.now()

print("Attempting to enforce equal opportunity...")
training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, 0.01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

if training_race_cases is not None:
    print("--------------------EQUAL OPPORTUNITY RESULTS--------------------")
    print("")
    print("----------------------TRAINING DATA RESULTS----------------------")
    print("")
    for group in training_race_cases.keys():
        accuracy = get_num_correct(training_race_cases[group]) / len(training_race_cases[group])
        print("Accuracy for " + group + ": " + str(accuracy))

    print("")
    for group in training_race_cases.keys():
        FPR = get_false_positive_rate(training_race_cases[group])
        print("FPR for " + group + ": " + str(FPR))

    print("")
    for group in training_race_cases.keys():
        FNR = get_false_negative_rate(training_race_cases[group])
        print("FNR for " + group + ": " + str(FNR))

    print("")
    for group in training_race_cases.keys():
        TPR = get_true_positive_rate(training_race_cases[group])
        print("TPR for " + group + ": " + str(TPR))

    print("")
    for group in training_race_cases.keys():
        TNR = get_true_negative_rate(training_race_cases[group])
        print("TNR for " + group + ": " + str(TNR))

    print("")
    for group in training_race_cases.keys():
        print("Threshold for " + group + ": " + str(thresholds[group]))

    print("")
    total_cost = apply_financials(training_race_cases)
    print("Cost on training data: ")
    print('${:,.0f}'.format(total_cost))
    total_accuracy = get_total_accuracy(training_race_cases)
    print("Accuracy on training data: " + str(total_accuracy))
    print("")
    print("------------------------TEST DATA RESULTS------------------------")
    print("")
    for group in test_race_cases.keys():
        accuracy = get_num_correct(test_race_cases[group]) / len(test_race_cases[group])
        print("Accuracy for " + group + ": " + str(accuracy))

    print("")
    for group in test_race_cases.keys():
        FPR = get_false_positive_rate(test_race_cases[group])
        print("FPR for " + group + ": " + str(FPR))

    print("")
    for group in test_race_cases.keys():
        FNR = get_false_negative_rate(test_race_cases[group])
        print("FNR for " + group + ": " + str(FNR))

    print("")
    for group in test_race_cases.keys():
        TPR = get_true_positive_rate(test_race_cases[group])
        print("TPR for " + group + ": " + str(TPR))

    print("")
    for group in test_race_cases.keys():
        TNR = get_true_negative_rate(test_race_cases[group])
        print("TNR for " + group + ": " + str(TNR))

    print("")
    total_cost = apply_financials(test_race_cases)
    print("Cost on test data: ")
    print('${:,.0f}'.format(total_cost))
    total_accuracy = get_total_accuracy(test_race_cases)
    print("Accuracy on test data: " + str(total_accuracy))
    print("-----------------------------------------------------------------")
    print("")

    end = datetime.now()

    seconds = end-begin
    print("Postprocessing took approximately: " + str(seconds) + " seconds")



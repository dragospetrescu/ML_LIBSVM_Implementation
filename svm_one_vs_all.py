from concurrent.futures import ThreadPoolExecutor
import numpy
import random
import sys
from svmutil import *
import datetime


def one_vs_all_predict(y_predict, x_predict, svms, no_classes):

    matrix = numpy.zeros(shape=(len(y_predict), no_classes))

    for cls_minus_1 in range(0, no_classes):
        svm = svms[cls_minus_1]
        p_label, p_acc, p_val = svm_predict(y_predict, x_predict, svm, '-b 1')

        if p_val[0][0] > p_val[0][1]:
            max_index = 0
        else:
            max_index = 1
        if p_label[0] == 21.0:
            if max_index == 0:
                class_index = 1
            else:
                class_index = 0
        else:
            class_index = max_index

        for i in range(0, len(p_label)):
            val = p_val[i][class_index]
            matrix[i][cls_minus_1] = val

    my_label = []
    for i in range(0, len(y_predict)):
        arr = matrix[i]
        possible_classes = numpy.where(arr == numpy.amax(arr))
        label = random.choice(possible_classes)[0] + 1
        if int(label) == 0:
            print('ERROR')
        my_label.append(label)

    success_inputs = 0.0
    for i in range(0, len(y_predict)):
        if int(my_label[i]) == int(y_predict[i]):
            success_inputs += 1.0
    accuracy = success_inputs / len(y_predict)
    error = 1.0 - success_inputs / len(y_predict)
    return my_label, accuracy, error


def run_svm(parameter, y_train, x_train, y_predict, x_predict, no_classes):

    y_map = {}
    for i in range(1, no_classes + 1):
        y_map[i] = []


    for i in range(0, len(y_train)):
        y = y_train[i]
        for clss in range(1, no_classes + 1):
            if y == clss:
                y_map[clss].append(clss)
            else:
                y_map[clss].append(no_classes + 1)

    svms = []

    for clss in range(1, no_classes + 1):
        svm = svm_train(y_map[clss], x_train, parameter)
        svms.append(svm)

    my_label, accuracy, error = one_vs_all_predict(y_predict, x_predict, svms, no_classes)

    print('Parameters: ' + parameter)
    print('Marime set test: ' + str(len(y_predict)))

    print('Success rate predicting: ' + str(accuracy))
    print('Error rate predicting: ' + str(error))

    print('Matrice de confuzie set date testare')
    create_confussion_matrix(my_label, y_predict, no_classes)

    my_label, accuracy, error = one_vs_all_predict(y_train, x_train, svms, no_classes)
    print('Success rate training: ' + str(accuracy))
    print('Error rate training: ' + str(error))

    print('Matrice de confuzie set date antrenare')
    create_confussion_matrix(my_label, y_train, no_classes)


def create_confussion_matrix(predict, actual, no_classes):
    matrix = numpy.zeros(shape=(no_classes,no_classes))
    for i in range(0, len(predict)):
        matrix[int(actual[i]) - 1][int(predict[i]) - 1] += 1

    for i in range(0, no_classes):
        for j in range(0, no_classes):
            print(str(int(matrix[i][j])) + ' ', end="", flush=True)
        print()

# Read data in LIBSVM format

parameter = ''
for i in range(1, len(sys.argv)):
    parameter = str(parameter) + str(sys.argv[i]) + ' '
print('Date: ' + str(datetime.datetime.now()))

y_train, x_train = svm_read_problem('input/news20/news20_training')
y_predict, x_predict = svm_read_problem('input/news20/news20_predict')

run_svm(parameter, y_train[:1000], x_train[:1000], y_predict[:100], x_predict[:100], 20)

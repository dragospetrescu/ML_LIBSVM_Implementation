from concurrent.futures import ThreadPoolExecutor
import numpy
import random
from svmutil import *
import sys
import datetime


def one_vs_one_predict(y_predict, x_predict, svms, no_classes):
     matrix = numpy.zeros(shape=(len(y_predict), no_classes))
     for svm in svms:
         p_label, p_acc, p_val = svm_predict(y_predict, x_predict, svm)
         for i in range(0, len(p_label)):
             matrix[i][int(p_label[i]) - 1] += 1

     my_label = []
     for i in range(0, len(y_predict)):
         arr = matrix[i]
         possible_classes = numpy.where(arr == numpy.amax(arr))
         my_label.append(random.choice(possible_classes)[0] + 1)

     success_inputs = 0.0
     for i in range(0, len(y_predict)):
         if int(my_label[i]) == int(y_predict[i]):
             success_inputs += 1.0
     accuracy = success_inputs / len(y_predict)
     error = 1.0 - success_inputs / len(y_predict)
     return my_label, accuracy, error


def run_svm(parameter, y_train, x_train, y_predict, x_predict, no_classes):

    x_map = {}
    for i in range(1, no_classes + 1):
        x_map[i] = []

    for i in range(0, len(y_train)):
        y = y_train[i]
        x = x_train[i]
        x_map[y].append(x)

    svms = []

    for i in range(1, no_classes + 1):
        for j in range(i + 1, no_classes + 1):
            y_train_2_classes = [i] * len(x_map[i]) + [j] * len(x_map[j])
            x_train_2_classes = x_map[i] + x_map[j]

            svm = svm_train(y_train_2_classes, x_train_2_classes, parameter)
            svms.append(svm)

    my_label, accuracy, error = one_vs_one_predict(y_predict, x_predict, svms, no_classes)

    print('Parameters: ' + parameter)
    print('Marime set test: ' + str(len(y_predict)))

    print('Success rate predicting: ' + str(accuracy))
    print('Error rate predicting: ' + str(error))

    print('Matrice de confuzie set date testare')
    create_confussion_matrix(my_label, y_predict, no_classes)

    my_label, accuracy, error = one_vs_one_predict(y_train, x_train, svms, no_classes)
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

run_svm(parameter, y_train, x_train, y_predict, x_predict, 20)

from concurrent.futures import ThreadPoolExecutor
import numpy
import random
import sys
import datetime
from svmutil import *


def run_svm(parameter, y_train, x_train, y_predict, x_predict, no_classes):


    m = svm_train(y_train, x_train, parameter)
    p_label, p_acc, p_val = svm_predict(y_predict, x_predict, m)

    print('Parameters: ' + parameter)
    print('Marime set test: ' + str(len(y_predict)))
    print('Success rate predicting: ' + str(p_acc[0]))
    print('Error rate predicting: ' + str(p_acc[1]))

    print('Matrice de confuzie set date testare')
    create_confussion_matrix(p_label, y_predict, no_classes)

    p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    print('Success rate predicting: ' + str(p_acc[0]))
    print('Error rate predicting: ' + str(p_acc[1]))
    print('Matrice de confuzie set date antrenare')
    create_confussion_matrix(p_label, y_train, no_classes)


def create_confussion_matrix(predict, actual, no_classes):
    matrix = numpy.zeros(shape=(no_classes,no_classes))
    for i in range(0, len(predict)):
        matrix[int(actual[i]) - 1][int(predict[i]) - 1] += 1

    for i in range(0, no_classes):
        for j in range(0, no_classes):
            print(str(int(matrix[i][j])) + ' ', end="", flush=True)
        print()

# Read data in LIBSVM format
y_train, x_train = svm_read_problem('input/skin/skin_training')
y_predict, x_predict = svm_read_problem('input/skin/skin_predict')
executor = ThreadPoolExecutor(max_workers=3)

parameter = ''
for i in range(1, len(sys.argv)):
    parameter = str(parameter) + str(sys.argv[i]) + ' '
print('Date: ' + str(datetime.datetime.now()))

run_svm(parameter, y_train, x_train, y_predict, x_predict, 2)

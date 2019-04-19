from concurrent.futures import ThreadPoolExecutor
import numpy

from svmutil import *


def run_svm(parameter, y_train, x_train, y_predict, x_predict, no_classes):
    print('Parameters: ' + parameter)
    print('Marime set test: ' + str(len(y_predict)))

    m = svm_train(y_train, x_train, parameter)
    p_label, p_acc, p_val = svm_predict(y_predict, x_predict, m)
    print(p_acc)

    print('Matrice de confuzie set date testare')
    create_confussion_matrix(p_label, y_predict, no_classes)

    p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
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
y_train, x_train = svm_read_problem('input/skin/skin_training2')
y_predict, x_predict = svm_read_problem('input/skin/skin_predict2')
executor = ThreadPoolExecutor(max_workers=3)

parameters = [
    # '-t 0',
    # '-t 1 -d 3',
    # '-t 1 -d 2',
    # '-t 1 -d 1',
    # '-t 1 -d 5',
    # '-t 1 -g 0.0001',
    # '-t 1 -g 0.1',
    # '-t 1 -g 0.5',
    # '-t 1 -g 1',
    # '-t 1 -r 0',
    # '-t 1 -r 0.0001',
    # '-t 1 -r -0.0001',
    # '-t 1 -r 0.1',
    # '-t 1 -r -0.1',
    # '-t 2 -g 0.0001',
    # '-t 2 -g 0.01',
    # '-t 2 -g 0.5',
    # '-t 2 -g 1',
    # '-t 3 -g 0.0001',
    # '-t 3 -g 0.01',
    # '-t 3 -g 0.5',
    # '-t 3 -g 1',
    # '-t 3 -r 0',
    # '-t 3 -r 0.0001',
    # '-t 3 -r -0.0001',
    # '-t 3 -r 0.1',
    '-t 3 -r -0.1'
]
for parameter in parameters:
    executor.submit(run_svm(parameter, y_train[:5000], x_train[:5000], y_predict[:1000], x_predict[:1000], 2))
executor.shutdown(wait=True)

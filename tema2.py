from concurrent.futures import ThreadPoolExecutor

from svmutil import *


def run_svm(parameter, y_train, x_train, y_predict, x_predict):
    m = svm_train(y_train[:500], x_train[:500], parameter)
    p_label, p_acc, p_val = svm_predict(y_predict[:200], x_predict[:200], m)

    print(p_label)

    folder_name = parameter.replace(" ", "")
    f = open("results/skin/" + str(folder_name), "w+")

    f.write(str(parameter) + "\r\n")
    f.write(str(p_acc) + "\r\n")
    f.close()


# Read data in LIBSVM format
y_train, x_train = svm_read_problem('input/skin_training')
y_predict, x_predict = svm_read_problem('input/skin_predict')
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
    executor.submit(run_svm(parameter, y_train, x_train, y_predict, x_predict))
executor.shutdown(wait=True)





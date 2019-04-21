#!/bin/bash
import re
import os

def average(lst):
    return sum(lst) / len(lst)

def getKernelName(nr):
    if nr == 0:
        return 'Liniar'
    if nr == 1:
        return 'Polinomial'
    if nr == 2:
        return 'Radial'
    if nr == 3:
        return 'Sigmoid'

dirname = 'results/news20/svm_one_vs_all/'
files = sorted(os.listdir(dirname))
for filename in files:
    f = open(dirname + filename, "r")
    itr = []
    nu = []
    nSV = []
    nBSV = []
    success_rate_predicting = None
    error_rate_predicting = None
    error_rate_training = None
    success_rate_training = None
    for line in f:
        if line.startswith('optimization finished'):
            nums = list(map(int, re.findall(r'\d+', line)))
            itr.append(nums[0])
        if line.startswith('nu'):
            nums = list(map(int, re.findall(r'\d+', line)))
            nu.append(nums[0])
        if line.startswith('nSV'):
            nums = list(map(int, re.findall(r'\d+', line)))
            nSV.append(nums[0])
            nBSV.append(nums[1])
        if line.startswith('Parameters'):
            parameters = line
            nums = list(map(int, re.findall(r'\d+', line)))
            kernel = getKernelName(nums[0])
        if line.startswith('Marime set test'):
            marime_set_test = line
        if line.startswith('Success rate predicting:'):
            if success_rate_predicting is None:
                success_rate_predicting = line
            else:
                 success_rate_training = line
        if line.startswith('Error rate predicting:'):
            if error_rate_predicting is None:
                error_rate_predicting = line
            else:
                 error_rate_training = line
        if line.startswith('Matrice de confuzie set date testare'):
            test_matrix_name = 'confussion_matrix/' + os.path.basename(f.name) + '_predict_matrixs'
            test_matrix_file = open(test_matrix_name,"w+")
            for i in range(0, 20):
                line = f.readline()
                test_matrix_file.write(line)
            test_matrix_file.close()
        if line.startswith('Success rate training:'):
            success_rate_training = line
        if line.startswith('Error rate training:'):
            error_rate_training = line
        if line.startswith('Matrice de confuzie set date antrenare'):
            test_matrix_name = 'confussion_matrix/' + os.path.basename(f.name) + '_train_matrixs'
            test_matrix_file = open(test_matrix_name,"w+")
            for i in range(0, 20):
                line = f.readline()
                test_matrix_file.write(line)
            test_matrix_file.close()

    result = open("svm_one_vs_all_report.txt", "a")
    result.write('Kernel: ' + kernel +'\n')
    result.write(parameters)
    result.write('Average numar vectori suport: ' + str(average(nSV)) +'\n')
    result.write('Numar total numar vectori suport:' + str(sum(nSV)) +'\n')
    result.write('Average numar iteratii: ' + str(average(itr) * 12) +'\n')
    result.write('Numar total iteratii:' + str(sum(itr)) +'\n')
    result.write(success_rate_predicting)
    result.write(error_rate_predicting)
    if not success_rate_training is None:
        result.write(success_rate_training)
        result.write(error_rate_training)
    result.write(marime_set_test)
    result.write('Matrice de confuzie set predict\n\n\n')
    result.write('Matrice de confuzie set train\n\n')
    result.write('\n\n')

    result.close

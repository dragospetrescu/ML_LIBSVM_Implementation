import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
import sys

def threaded_func(args):
    parameter = args[0]
    fil = args[1]

    test_name = parameter.replace(" ", "")
    dir_name = fil.split('.')[0]
    log_path = 'results/news20/' + str(dir_name) + '/' + str(test_name)
    command = "python3.5 " + fil + " " + parameter
    command_parts = command.split(' ')
    with open(log_path, "wb", 0) as out:
    	subprocess.run(command_parts, stdout=out, stderr=out, check=True)


executor = ThreadPoolExecutor(max_workers=4)

parameters = [
    [
    '-t 0 -b 1',
    '-t 1 -b 1',
    '-t 1 -d 1 -b 1',
    '-t 1 -d 5 -b 1',
    '-t 1 -g 0.5 -b 1',
    '-t 1 -g 1 -b 1',
    '-t 1 -r 0.0001 -b 1',
    '-t 1 -r 0.1 -b 1',
    '-t 2 -g 0.0001 -b 1',
    '-t 2 -g 0.5 -b 1',
    '-t 2 -g 1 -b 1',
    '-t 3 -b 1',
    '-t 3 -g 0.5 -b 1',
    '-t 3 -g 1 -b 1',
    '-t 3 -r 0.0001 -b 1',
    '-t 3 -r 0.1 -b 1'
    ],
    [
    '-t 0',
    '-t 1',
    '-t 1 -d 1',
    '-t 1 -d 5',
    '-t 1 -g 0.5',
    '-t 1 -g 1',
    '-t 1 -r 0.0001',
    '-t 1 -r 0.1',
    '-t 2 -g 0.0001',
    '-t 2 -g 0.5',
    '-t 2 -g 1',
    '-t 3',
    '-t 3 -g 0.5',
    '-t 3 -g 1',
    '-t 3 -r 0.0001',
    '-t 3 -r 0.1'
    ],
]
files = ['svm_one_vs_all.py', 'svm_one_vs_one.py']

for j in range(0, len(files)):
    my_file = files[j]
    my_parameters = parameters[j]
    for i in range(0, len(my_parameters)):
        parameter = my_parameters[i]
        executor.submit(threaded_func, (parameter, my_file))
executor.shutdown(wait=True)

import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

def threaded_func(parameter):
    test_name = parameter.replace(" ", "")
    log_path = 'results/news20/all/' + test_name
    command = "python3.5 svm_one_vs_all.py " + parameter
    command_parts = command.split(' ')
    with open(log_path, "wb", 0) as out:
    	subprocess.run(command_parts, stdout=out, check=True)


executor = ThreadPoolExecutor(max_workers=3)

parameters = [
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
    '-t 3 -r 0.1 -b 1',
]
for i in range(0, len(parameters)):
    parameter = parameters[i]
    executor.submit(threaded_func, parameter)
executor.shutdown(wait=True)

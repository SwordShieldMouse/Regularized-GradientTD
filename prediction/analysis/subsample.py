import sys
import math
import os
sys.path.append(os.getcwd())

import numpy as np
from functools import partial
from multiprocessing.pool import Pool
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.path import rest
from src.utils.model import loadExperiment
from src.utils.path import up

EVERY = 10

PERCENTAGES =   [0.1, 0.2, 0.7]
WINDOWS =       [4, 10, 20]
def windowAverage(arr):
    assert sum(PERCENTAGES) == 1
    perc_sum = 0
    i_sum = 0
    parts = []
    for i in range(len(WINDOWS)):
        window = WINDOWS[i]
        perc_sum += PERCENTAGES[i]

        l = math.floor(len(arr) * perc_sum)

        slices = math.ceil(l / window)
        s = np.split(arr[i_sum : i_sum + l], slices)

        i_sum += l

        part = np.mean(s, axis=1)
        parts.append(part)

    return np.concatenate(parts)

def everyN(arr, n):
    out = []
    for i in range(arr.shape[0]):
        if i % n == 0:
            out.append(arr[i])

    return out


def loadResults(path):
    summary_path = path + '/errors_summary.npy'
    return np.load(summary_path, allow_pickle=True), summary_path

def processResultPath(new_base, resultAndPath):
    result, path = resultAndPath
    mean, stderr, count = result

    # sampled_mean = windowAverage(mean)
    # sampled_stderr = windowAverage(stderr)

    sampled_mean = everyN(mean, EVERY)
    sampled_stderr = everyN(stderr, EVERY)

    sampled = [sampled_mean, sampled_stderr, count]

    new_path = new_base + '/' + rest(path)
    os.makedirs(up(new_path), exist_ok=True)

    np.save(new_path, sampled)

if __name__ == '__main__':
    pool = Pool()

    new_base = sys.argv[1]
    exp_paths = sys.argv[2:]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)

        result_paths = listResultsPaths(exp)
        results = map(loadResults, result_paths)

        pool.map(partial(processResultPath, new_base), results)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest
from src.analysis.colors import colors
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults
from PyExpUtils.utils.arrays import first

def getBest(results):
    best = first(results)
    bestVal = np.mean(best.load()[0])

    for r in results:
        a = r.load()[0]
        am = np.mean(a)
        if am > bestVal:
            best = r
            bestVal = am

    return best

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)

        results = loadResults(exp, 'return_summary.npy')

        best = getBest(results)
        print('best parameters:', exp_path)
        print(best.params)

        alg = exp.agent

        b = plotBest(best, ax, label=alg, color=colors[alg])
        bounds.append(b)


if __name__ == "__main__":
    f, axes = plt.subplots(1)

    bounds = []

    exp_paths = sys.argv[1:]

    generatePlot(axes, exp_paths, bounds)

    save_path = 'figures'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/learning-curve.png', dpi=100)

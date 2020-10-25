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

TITLE="MountainCar"
SAVE_PATH=os.path.join(os.getcwd(),"figures/MountainCar")

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_PATHS = list(map(lambda alg: os.path.join(FILE_DIR, f'{alg}.json'),['PFGQ','PFRobust', 'QC','QRC', 'QLearning']))


def getBest(results):
    best = first(results)
    bestVal = np.mean(best.load()[0])

    for r in results:
        a = r.load()[0]
        am = np.mean(a[-1000:])
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

    generatePlot(axes, EXP_PATHS, bounds)

    os.makedirs(SAVE_PATH, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    axes.set_title(TITLE)
    plt.savefig(f'{SAVE_PATH}/learning-curve.png', dpi=100)

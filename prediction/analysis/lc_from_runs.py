import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest
from src.analysis.colormap import colors
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults
from PyExpUtils.utils.arrays import first

def getBest(results):
    best = first(results)
    bestVal = np.mean( # AUC
        np.mean(best.load(), axis=0) # avg over runs
    )

    for r in results:
        if not np.isfinite(bestVal):
           best = r
           bestVal = np.mean( #AUC
               np.mean(r.load(), axis=0) # Avg over runs
            )
           continue
        a = np.mean(r.load(), axis=0) # Avg over runs
        am = np.mean(a) # AUC
        if am < bestVal:
            best = r
            bestVal = am

    print(f"{bestVal} <= {best.params}")
    return best

def generatePlot(ax, exp_paths):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)
        alg = exp.agent

        results = loadResults(exp, 'rmspbe.npy')


        best = getBest(results)
        print('best parameters:', exp_path)
        print(best.params)

        best_data = np.mean(best.load(), axis=0) # Avg over runs
        ax.plot(best_data,  label=alg, color=colors[alg])


if __name__ == "__main__":
    f, axes = plt.subplots(1)

    exp_paths = sys.argv[1:]

    generatePlot(axes, exp_paths)

    save_path = 'figures'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/learning-curve-allRuns.pdf')

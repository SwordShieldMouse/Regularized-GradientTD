import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest, save
from src.analysis.results import loadResults, whereParameterEquals, getBestEnd, find
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    ax = plt.gca()
    # ax.semilogx()
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)

        # load the errors and hnorm files
        errors = loadResults(exp, 'errors_summary.npy')
        results = loadResults(exp, 'hupd_summary.npy')

        # choose the best parameters from the _errors_
        best = getBestEnd(errors)

        best_hupd = find(results, best)

        label = fileName(exp_path).replace('.json', '')

        plotBest(best_hupd, ax, label=label)

    plt.show()
    # save(exp, f'learning-curve')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)

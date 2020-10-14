import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest, save
from src.analysis.results import loadResults, whereParameterEquals, getBestEnd, find
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_path):
    ax = plt.gca()
    # ax.semilogx()
    exp = loadExperiment(exp_path)

    # load the errors and hnorm files
    errors = loadResults(exp, 'errors_summary.npy')
    results = loadResults(exp, 'stepsize_summary.npy')

    # choose the best parameters from the _errors_
    best = getBestEnd(errors)

    best_ss = find(results, best)

    alg = exp.agent.replace('adagrad', '')

    plotBest(best_ss, ax, label=['w', 'h'])

    ax.set_ylim([0, 4])

    print(alg)
    # plt.show()
    save(exp, f'stepsizes-{alg}')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    for exp_path in exp_paths:
        f = fileName(exp_path)
        if 'ideal' in f:
            continue

        generatePlot(exp_path)

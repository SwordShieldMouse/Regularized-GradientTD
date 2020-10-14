import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals, splitOverParameter
from src.utils.model import loadExperiment

colors = ['blue', 'red', 'green', 'black']

def generatePlot(exp_paths):
    ax = plt.gca()

    bounds = []
    i = -1
    for exp_path in exp_paths:
        i += 1
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'errors_summary.npy')

        param_dict = splitOverParameter(results, 'ratio')

        for key in param_dict:
            bound = plotSensitivity(param_dict[key], 'alpha', ax, color=colors[i], bestBy='end')
            bounds.append(bound)

    lower = min(map(lambda x: x[0], bounds))
    upper = max(map(lambda x: x[1], bounds))

    ax.set_ylim([lower, upper])

    ax.set_xscale("log", basex=2)
    # plt.show()
    save(exp, f'alpha_over_beta_rmsve', type='svg')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)


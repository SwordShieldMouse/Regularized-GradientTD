import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest, save
from src.analysis.results import loadResults, whereParameterEquals, getBestEnd, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    ax = plt.gca()
    # ax.semilogx()
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)

        if exp.agent == 'TDadagrad':
            continue

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)

        dashed = use_ideal_h
        color = colors[exp.agent]

        # load the errors and hnorm files
        errors = loadResults(exp, 'errors_summary.npy')
        results = loadResults(exp, 'ndh_summary.npy')

        # choose the best parameters from the _errors_
        best = getBestEnd(errors)

        best_ndh = find(results, best)

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        plotBest(best_ndh, ax, label=label, color=color, dashed=dashed)

    # plt.show()
    save(exp, f'norm_delta-hat')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)

import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from functools import partial
from multiprocessing.pool import Pool

from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

def generatePlot(exp_paths):
    ax = plt.gca()

    bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        bound = plotSensitivity(results, 'alpha', ax, label=label, color=color, dashed=dashed)
        bounds.append(bound)

    lower = min(map(lambda x: x[0], bounds))
    upper = max(map(lambda x: x[1], bounds))

    ax.set_ylim([lower, upper])

    ax.set_xscale("log", basex=2)
    # plt.show()
    save(exp, f'alpha-sensitivity')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)


import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals
from src.utils.model import loadExperiment
from src.analysis.colormap import colors

def generatePlot(exp_paths):
    ax = plt.gca()

    bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        # residuals = loadResults(exp, 'residuals_summary.npy')
        results = loadResults(exp)

        use_ideal_h = exp.getPermutation(0)['metaParameters'].get('use_ideal_h', False)
        agent = exp.agent.replace('adagrad', '')
        color = colors[exp.agent]

        label = agent
        if use_ideal_h:
            label += '-h*'

        bound = plotSensitivity(results, 'tiles', ax, label=label, color=color, dashed = use_ideal_h)
        bounds.append(bound)

    lower = min(map(lambda x: x[0], bounds))
    upper = max(map(lambda x: x[1], bounds))

    ax.set_ylim([lower, upper])

    # save(exp, f'representability')
    plt.show()
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)


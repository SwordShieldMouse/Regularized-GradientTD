import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterEquals, getBestEnd, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    f, axes = plt.subplots(2,2)
    # ax.semilogx()

    # RMSPBE plots
    ax = axes[0,0]
    rmspbe_bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        bounds = plot(results, ax, label=label, color=color, dashed=dashed)#, bestby=auc or end
        rmspbe_bounds.append(bounds)
        ax.set_ylabel("RMSPBE")
        ax.set_title("RMSPBE")

    ax = axes[0,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        results = loadResults(exp, 'rmspbe_summary.npy')

        best = getBestEnd(rmsve) # getBest(rmsve)
        best_rmspbe = find(results, best)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        bounds = plotBest(best_rmspbe, ax, label=label, color=color, dashed=dashed)
        rmspbe_bounds.append(bounds)
        ax.set_title("RMSVE")

    # RMSVE plots
    ax = axes[1,0]
    rmsve_bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        rmspbe = loadResults(exp, 'rmspbe_summary.npy')

        # if exp.agent == 'TDadagrad':
        #     continue

        # best PBE using AUC
        best = getBestEnd(rmspbe)
        best_rmsve = find(rmsve, best)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        bounds = plotBest(best_rmsve, ax, label=label, color=color, dashed=dashed)
        rmsve_bounds.append(bounds)
        ax.set_ylabel("RMSVE")

    ax = axes[1,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        bounds = plot(results, ax, label=label, color=color, dashed=dashed)
        rmsve_bounds.append(bounds)

    # rmspbe
    rmspbe_lower = min(map(lambda x: x[0], rmspbe_bounds)) * 0.9
    rmspbe_upper = max(map(lambda x: x[1], rmspbe_bounds)) * 1.05
    axes[0, 0].set_ylim([rmspbe_lower, rmspbe_upper])
    axes[0, 1].set_ylim([rmspbe_lower, rmspbe_upper])

    # rmsve
    rmsve_lower = min(map(lambda x: x[0], rmsve_bounds)) * 0.9
    rmsve_upper = max(map(lambda x: x[1], rmsve_bounds)) * 1.05
    axes[1, 0].set_ylim([rmsve_lower, rmsve_upper])
    axes[1, 1].set_ylim([rmsve_lower, rmsve_upper])

    plt.show()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)

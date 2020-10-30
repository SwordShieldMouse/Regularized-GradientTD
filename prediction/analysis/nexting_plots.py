import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from PyExpUtils.results.results import loadResults

from src.analysis.learning_curve import plotBest
from src.analysis.colormap import colors
from src.experiment import ExperimentModel
from PyExpUtils.utils.arrays import first
from src.utils.Critterbot import loadReturns, getSensorNum

SUMMARY = 'nmse_summary.npy'
YLIM=[0,500]

def getResult(r):
    return r.load()

def getResults(exp):
    if exp.agent in ['PFCombined','PFResidual']:
       algb = exp._d['metaParameters']['algB'][0]
       result = loadResults(exp, SUMMARY)
       return {f"{exp.agent} ({algb})": result}
    elif exp.agent == 'TD':
        lmda = exp._d["metaParameters"]["lambda"]
        return {f"{exp.agent}({lmda})": loadResults(exp, SUMMARY)}
    return {f"{exp.agent}": loadResults(exp, SUMMARY)}

def getBest(results):
    best = first(results)
    bestVal = np.mean(getResult(best))

    for r in results:
        if not np.isfinite(bestVal):
           best = r
           bestVal = np.mean(getResult(r))
           continue
        a = getResult(r)
        am = np.mean(a)
        if am < bestVal:
            best = r
            bestVal = am

    print(f"{bestVal} <= {best.params}")
    return best

def extract(results):
    means=[]
    for r in results:
        means.append(getResult(r))
    return np.array(means)

def generatePlot(ax, exp_path, aggregate, getColor=lambda agent: colors[agent], plotAllSensors=False):
    exp = ExperimentModel.load(exp_path)

    # Each result = settings for one agent
    for alg, result in getResults(exp).items():
        means = extract(result)
        # av = np.median(means, axis=0)
        # ax.plot(av, color=colors[exp.agent], label=alg)

        if plotAllSensors:
            for line in means:
                ax.plot(line, color=colors[exp.agent], alpha=0.4)
        ax.plot(aggregate(means), color=getColor(exp.agent), linewidth=2.0)
        ax.set_title(alg)

def generateAggregatePlot(exp_paths, aggregate, ylim):
    f, axes = plt.subplots(1)
    for exp_path in exp_paths:
        generatePlot(axes, exp_path, aggregate=aggregate, plotAllSensors=False)
    axes.set_ylim(ylim)
    return f, axes

def generateAllSensorPlots(exp_paths, aggregate, ylim):
    f, axes = plt.subplots(1, len(exp_paths))
    for i in range(len(exp_paths)):
        ax = axes[i] if len(exp_paths) > 1 else axes
        generatePlot(ax, exp_paths[i], aggregate=aggregate, getColor=lambda agent: 'black', plotAllSensors = True)
        ax.set_ylim(ylim)
    return f, axes

# ==========================================================
# entry points
# ==========================================================
Median = lambda d: np.median(d, axis=0)
Mean = lambda d: np.mean(d, axis=0)

def generateMedianPlot(exp_paths, ylim=None):
    return generateAggregatePlot(exp_paths, lambda data: Median(data), ylim)

def generateMeanPlot(exp_paths, ylim=None):
    return generateAggregatePlot(exp_paths, lambda data: Mean(data), ylim)

def generateMeanAndAllSensorsPlots(exp_paths, ylim=None):
    return generateAllSensorPlots(exp_paths, lambda data: Mean(data), ylim)

def generateMedianAndAllSensorsPlots(exp_paths, ylim=None):
    return generateAllSensorPlots(exp_paths, lambda data: Median(data), ylim)


if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    width = 8
    height = (24/5)
    save_path = 'figures'

    os.makedirs(save_path, exist_ok=True)

    f, axes = generateMedianPlot(exp_paths)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/medians-nexting.png', dpi=100)

    f, axes = generateMeanPlot(exp_paths)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/means-nexting.png', dpi=100)

    f, axes = generateMedianAndAllSensorsPlots(exp_paths)
    f.set_size_inches((width*len(exp_paths), height), forward=False)
    plt.savefig(f'{save_path}/allSensors-median-nexting.png', dpi=100)

    f, axes = generateMeanAndAllSensorsPlots(exp_paths)
    f.set_size_inches((width*len(exp_paths), height), forward=False)
    plt.savefig(f'{save_path}/allSensors-median-nexting.png', dpi=100)

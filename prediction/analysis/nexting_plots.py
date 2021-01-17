import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from PyExpUtils.results.results import loadResults, whereParametersEqual
from PyExpUtils.utils.permute  import getParameterPermutation, getNumberOfPermutations

from src.analysis.learning_curve import plotBest
from src.analysis.colormap import colors
from src.experiment import ExperimentModel
from PyExpUtils.utils.arrays import first
from src.utils.Critterbot import loadReturns, getSensorNum

def _getDiscounts(exp):
    return [exp._d["metaParameters"]["gamma"][-1],]

def getResult(r):
    return np.array(r.load(), dtype='float')

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
        if r.params['sensorIdx'] == 43 and SUMMARY=='smape_summary.npy':
            # This sensor can't be measured w/ smape (always zero)
            continue
        res = getResult(r)
        if SUMMARY == 'mse_summary.npy':
            # RMSE
            res = np.sqrt(res)
        means.append(res)
    return np.array(means)

def generatePlot_old(ax, exp_path, aggregate, getColor=lambda agent: colors[agent], plotAllSensors=False):
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

def generatePlot(ax, exp_path, aggregate, getColor=lambda agent: colors[agent], plotAllSensors=False):
    print(f"Experiment: {exp_path}")

    exp = ExperimentModel.load(exp_path)
    data = getBestSensorData(exp_path, aggregate)

    if plotAllSensors:
        for line in data:
            ax.plot(line, color=colors[exp.agent], alpha=0.4)
    ax.plot(aggregate(data,axis=0), color=getColor(exp.agent), linewidth=2.5)
    ax.set_title(exp.agent)
    print()


def generateAggregatePlot(exp_paths, aggregate, ylim):
    f, axes = plt.subplots(1)
    for exp_path in exp_paths:
        generatePlot(axes, exp_path, aggregate=aggregate, plotAllSensors=False)
    axes.set_ylim(ylim)
    axes.set_title(None)
    return f, axes

def generateAllSensorPlots(exp_paths, aggregate, ylim):
    f, axes = plt.subplots(1, len(exp_paths))
    for i in range(len(exp_paths)):
        ax = axes[i] if len(exp_paths) > 1 else axes
        generatePlot(ax, exp_paths[i], aggregate=aggregate, getColor=lambda agent: 'black', plotAllSensors = True)
        ax.set_ylim(ylim)
    return f, axes

def getBestOverall(exp_path, aggregate):
    # Get the best parameter settings according to the AUC
    # of all sensors, aggregated by some func aggregate:R^T->R
    exp = ExperimentModel.load(exp_path)
    params = exp._d['metaParameters'].copy()
    del params['sensorIdx']
    del params['gamma']

    N = getNumberOfPermutations(params)
    best = np.inf
    bestIdx = 0
    for paramIdx in range(N):
        d = getParameterPermutation(params, paramIdx)
        results = whereParametersEqual(loadResults(exp, SUMMARY),d)
        means = []
        # Get AUC for each sensor
        for r in results:
            means.append(np.mean(r.load()[-20000:]))
        # Aggregate the AUCs and check if have best aggregate value
        mean = aggregate(means)
        if mean <= best:
            best = mean
            bestIdx = paramIdx
    # Get the best param settings again
    bestParams = getParameterPermutation(params, bestIdx)
    return bestParams

def getBestSensorData(exp_path, aggregate):
    exp = ExperimentModel.load(exp_path)
    indices = exp._d["metaParameters"]['sensorIdx']
    discounts = _getDiscounts(exp)
    bestSettings = getBestOverall(exp_path, aggregate)

    print(f"Sensor Indices: {indices}")
    print(f"Discounts: {discounts}")
    print(f"Best Settings: {bestSettings}")

    data = []
    for idx in indices:
        for gamma in discounts:
            bestSettings['sensorIdx'] = idx
            bestSettings['gamma'] = gamma
            data.append(first(whereParametersEqual(loadResults(exp, SUMMARY), bestSettings)).load())
    return np.array(data)

# ==========================================================
# entry points
# ==========================================================
Median = np.median
Mean = np.mean

def generateMedianPlot(exp_paths, ylim=None):
    return generateAggregatePlot(exp_paths, Median, ylim)
def generateMeanPlot(exp_paths, ylim=None):
    return generateAggregatePlot(exp_paths, Mean, ylim)

def generateMedianAndAllSensorsPlots(exp_paths, ylim=None):
    return generateAllSensorPlots(exp_paths, Median, ylim)
def generateMeanAndAllSensorsPlots(exp_paths, ylim=None):
    return generateAllSensorPlots(exp_paths, Mean, ylim)


if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    width = 8
    height = (24/5)
    lmda = list(filter(lambda t: t.startswith('lambda'),exp_paths[0].split("/")))[0]
    save_path = f'figures/Nexting/{lmda}'

    os.makedirs(save_path, exist_ok=True)

    median_ylims = {
        "mse": [0,1e4],
        "nmse": [0,3],
        "smape":[0,1]
    }

    mean_ylims = {
        "mse": [0,2.0e5],
        "nmse": [0,100],
        "smape":[0,1]
    }


    #datatypes = ["mse","nmse","smape"]
    datatypes = ['smape', ]
    for datatype in datatypes:
        SUMMARY = f"{datatype}_summary.npy"
        print(f"plotting {datatype}...")

        f, axes = generateMedianPlot(exp_paths, ylim=median_ylims[datatype])
        f.set_size_inches((width, height), forward=False)
        plt.savefig(f'{save_path}/{datatype}-medians-nexting.pdf')

        # f, axes = generateMeanPlot(exp_paths, ylim=mean_ylims[datatype])
        # f.set_size_inches((width, height), forward=False)
        # plt.savefig(f'{save_path}/{datatype}-means-nexting.png', dpi=100)

        f, axes = generateMedianAndAllSensorsPlots(exp_paths, ylim=median_ylims[datatype])
        f.set_size_inches((width*len(exp_paths), height), forward=False)
        plt.savefig(f'{save_path}/{datatype}-allSensors-median-nexting.pdf')

        # f, axes = generateMeanAndAllSensorsPlots(exp_paths, ylim=mean_ylims[datatype])
        # f.set_size_inches((width*len(exp_paths), height), forward=False)
        # plt.savefig(f'{save_path}/{datatype}-allSensors-mean-nexting.png', dpi=100)

    print("done!")

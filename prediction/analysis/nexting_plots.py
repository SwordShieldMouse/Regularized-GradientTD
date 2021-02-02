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
    return [0.9875]

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

# Wow fuck this turned into a mess
# --------------------------------
def generateAllSensorPlots(exp_paths, aggregate, ylim):
    f, axes = plt.subplots(1, len(exp_paths))
    for i in range(len(exp_paths)):
        ax = axes[i] if len(exp_paths) > 1 else axes
        generatePlot(ax, exp_paths[i], aggregate=aggregate, getColor=lambda agent: 'black', plotAllSensors = True)
        ax.set_ylim(ylim)
    return f, axes

def generateBestParamPlot(exp_paths, aggregate, ylim):
    f, axes = plt.subplots()
    for i in range(len(exp_paths)):
        generateBestSensorPlot(axes, exp_paths[i], aggregate=aggregate)
        axes.set_ylim(ylim)
    return f, axes

def generateAllBestSensorPlots(exp_paths, aggregate, ylim):
    f, axes = plt.subplots(1, len(exp_paths))
    for i in range(len(exp_paths)):
        ax = axes[i] if len(exp_paths) > 1 else axes
        generateBestSensorPlot(ax, exp_paths[i], aggregate=aggregate, getColor=lambda agent: 'black', plotAllSensors = True)
        ax.set_ylim(ylim)
    return f, axes

def getBestForSensor(exp_path, aggregate, sensorIdx):
    exp = ExperimentModel.load(exp_path)
    results = whereParametersEqual(loadResults(exp, SUMMARY), {'sensorIdx': sensorIdx})
    best = None
    bestval = np.inf
    for r in results:
        val = np.mean(r.load()[:300])
        if val <= bestval:
            best = r
            bestval = val
    if best is None:
        return np.ones_like(r.load())
    return best.load()

def generateBestSensorPlot(ax, exp_path, aggregate, getColor=lambda agent: colors[agent], plotAllSensors=False):
    print(f"Experiment: {exp_path}")

    exp = ExperimentModel.load(exp_path)
    sensors = exp._d['metaParameters']["sensorIdx"]

    all_data = []
    for sensorIdx in sensors:
        best_data = getBestForSensor(exp_path, aggregate, sensorIdx)
        all_data.append(best_data)

        if plotAllSensors:
            ax.plot(best_data, color=colors[exp.agent], alpha=0.4)
    all_data = aggregate(np.array(all_data), axis=0)
    ax.plot(all_data, color=getColor(exp.agent), linewidth=2.5)
    ax.set_title(exp.agent)
    print()

def generateAllSensorPlotsBest(exp_paths, aggregate, ylim):
    f, axes = plt.subplots(1, len(exp_paths))
    for i in range(len(exp_paths)):
        ax = axes[i] if len(exp_paths) > 1 else axes
        generatePlot(ax, exp_paths[i], aggregate=aggregate, getColor=lambda agent: 'black', plotAllSensors = True)
        ax.set_ylim(ylim)
    return f, axes

def generateSensitivityPlot(exp_paths):
    exp_paths = list(filter(lambda p: 'alpha' in ExperimentModel.load(p)._d["metaParameters"].keys(), exp_paths))
    f, axes = plt.subplots(1,len(exp_paths))
    for (i,exp_path) in enumerate(exp_paths):
        generateSensitivity(axes if len(exp_paths)==1 else axes[i], exp_path)
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
            if r.params['sensorIdx']!=43:
                means.append(np.mean(r.load()[-200:]))
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

def getSensitivity(exp):
    alphas = exp._d['metaParameters']["alpha"].copy()
    alphas.sort()

    settings = {}
    if 'eta' in exp._d['metaParameters']:
        settings['eta'] = 1

    full_data = []
    for alpha in alphas:
        settings['alpha'] = alpha
        subresults = whereParametersEqual(loadResults(exp, SUMMARY), settings)
        data = []
        for r in subresults:
            if r.params['sensorIdx'] != 43:
                # smape undefined on sensor 43 (all zeros)
                data.append(getResult(r)[-200:])
        #full_data.append(np.median(np.array(data), axis=0)[-20000:].mean())
        data=np.array(data)
        full_data.append(np.mean(data, axis=1))
    return alphas, np.array(full_data)

def generateSensitivity(ax, exp_path):
    exp = ExperimentModel.load(exp_path)
    if 'alpha' in exp._d["metaParameters"].keys():
        alphas, data = getSensitivity(exp)
        data=np.swapaxes(data,0,1)
        data = np.median(data, axis=0)
        ax.plot(alphas, data, color=colors[exp.agent])
        # for datum in data:
        #     ax.plot(alphas, datum, color=colors[exp.agent])
        ax.set_title(exp.agent)
        ax.set_xlabel(r"$\alpha$")
        ax.set_xscale('log')
        ax.set_ylabel("AUC")
        ax.set_ylim([0,1])



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

def generateBestMedianPlot(exp_paths, ylim=None):
    return generateBestParamPlot(exp_paths, Median, ylim)
def generateMedianAndAllBestSensorsPlots(exp_paths, ylim=None):
    return generateAllBestSensorPlots(exp_paths, Median, ylim)


if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    width = 8
    height = (24/5)
    lmda = list(filter(lambda t: t.startswith('lambda'),exp_paths[0].split("/")))[0]
    save_path = 'figures'

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
        plt.savefig(f'{save_path}/Nexting-{datatype}-medians.png')
        plt.savefig(f'{save_path}/Nexting-{datatype}-medians.pdf')

        f, axes = generateMedianAndAllSensorsPlots(exp_paths, ylim=median_ylims[datatype])
        f.set_size_inches((width*len(exp_paths), height), forward=False)
        plt.savefig(f'{save_path}/Nexting-{datatype}-allSensors-median.png')

        f, axes = generateBestMedianPlot(exp_paths, ylim=median_ylims[datatype])
        f.set_size_inches((width, height), forward=False)
        plt.savefig(f'{save_path}/Nexting-{datatype}-best-median.png')
        plt.savefig(f'{save_path}/Nexting-{datatype}-best-median.pdf')

        f, axes = generateMedianAndAllBestSensorsPlots(exp_paths, ylim=median_ylims[datatype])
        f.set_size_inches((width*len(exp_paths), height), forward=False)
        plt.savefig(f'{save_path}/Nexting-{datatype}-allBestSensors-median.png')
        plt.savefig(f'{save_path}/Nexting-{datatype}-allBestSensors-median.pdf')


        print("plotting sensitivity...")
        f, axes = generateSensitivityPlot(exp_paths)
        f.set_size_inches((width*len(axes), height), forward=False)
        plt.savefig(f'{save_path}/Nexting-{datatype}-medians-sensitivity.png')
        plt.savefig(f'{save_path}/Nexting-{datatype}-medians-sensitivity.pdf')


    print("done!")

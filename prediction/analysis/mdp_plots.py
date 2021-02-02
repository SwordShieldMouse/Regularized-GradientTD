import os
import sys
sys.path.append(os.getcwd())

from functools import reduce
import numpy as np

import matplotlib.pyplot as plt
from analysis.learning_curve import generatePlot, getBest

from src.analysis.learning_curve import plotBest
from src.analysis.colormap import colors
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults
from PyExpUtils.utils.arrays import first

plt.rcParams.update({'font.size': 20})

def plotMDP(paths, savepath, title, xlim=None, ylim=None):
    prefix = "batch-" if '/batch/' in paths[0] else ""
    f, axes = plt.subplots(1)
    bounds = []
    os.makedirs(savepath, exist_ok=True)

    def _getBest(results):
        best = first(results)
        bestVal = np.mean(best.load()[0])

        for r in results:
            if not np.isfinite(bestVal):
                best = r
                bestVal = np.mean(r.load()[0])
                continue
            a = r.load()[0]
            am = np.mean(a)
            if am < bestVal:
                best = r
                bestVal = am

        print(f"{bestVal} <= {best.params}")
        return best

    def _generatePlot(ax, exp_paths, bounds, fltr = None):
        for exp_path in exp_paths:
            exp = ExperimentModel.load(exp_path)
            alg = exp.agent

            results = loadResults(exp, 'rmspbe_summary.npy')

            if fltr is not None:
                results = filter(fltr, results)

            best = _getBest(results)
            print('best parameters:', exp_path)
            print(best.params)

            b = plotBest(best, ax, label=alg, color=colors[alg], dashed=False,linewidth=4.0)
            bounds.append(b)

        B = [np.inf, -np.inf]
        for (mn,mx) in bounds:
            if mx>=B[1]:
                B[1]=mx
            if mn<=B[0]:
                B[0]=mn
        ax.set_ylim(B)
        # for line in ax.get_lines():
        #     line.set_linewidth(4.0)



    #_generatePlot(axes, paths, bounds)
    # width = 8
    # height = (24/5)
    # f.set_size_inches((width, height), forward=False)
    # axes.set_title(title)
    # set_limits(axes,xlim,ylim)
    # plt.savefig(f'{savepath}/{prefix}learning-curve-allParams.png')

    # f, axes = plt.subplots(1)

    bounds = []

    fltr = lambda r: r.params.get('eta', 1) == 1
    _generatePlot(axes, paths, bounds, fltr)

    os.makedirs(savepath, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    axes.set_title(title)
    set_limits(axes,xlim,ylim)
    #axes.set_yscale('log')
    plt.savefig(f'{savepath}/{title}-{prefix}learning-curve.png')
    plt.savefig(f'{savepath}/{title}-{prefix}learning-curve.pdf')

def set_limits(axes, xlim,ylim):
    if ylim is not None:
        axes.set_ylim(ylim)
    if xlim is not None:
        axes.set_xlim(xlim)

def getConfigs(path):
    return list(filter(lambda p: p.endswith(".json"), os.listdir(path)))

def getSavePath(expname):
    #return os.path.join(os.getcwd(),f'figures/{expname}')
    return os.path.join(os.getcwd(),'figures')

def experiment_is(name, exp_paths):
    return reduce(lambda prev, path: prev and name in path, exp_paths)

def plotEach(exp_paths, savepath, problem):
    width = 8
    height = (24/5)
    os.makedirs(savepath, exist_ok=True)

    def _generatePlot(ax, exp_paths, bounds, feats, fltr = None):
        mn,mx=np.inf,-np.inf
        for exp_path in exp_paths:
            exp = ExperimentModel.load(exp_path)
            alg = exp.agent

            results = loadResults(exp, 'rmspbe_summary.npy')
            sub_results = filter(lambda r: r.params["representation"]==feats, results)

            if fltr is not None:
                sub_results = filter(fltr, sub_results)

            best = getBest(sub_results)
            print(f'best parameters ({feats}):', exp_path)
            print(best.params)


            b = plotBest(best, ax, label=alg, color=colors[alg], dashed=False,linewidth=4.0)
            bounds.append(b)

        B = [np.inf, -np.inf]
        for (mn,mx) in bounds:
            if mx>=B[1]:
                B[1]=mx
            if mn<=B[0]:
                B[0]=mn
        ax.set_ylim(B)
        ax.set_xlim([0,exp._d["steps"]])
        # for line in ax.get_lines():
        #     line.set_linewidth(4.0)

    for feats in ['tabular','inverted','dependent']:
        # bounds = []

        # f, axes = plt.subplots(1)
        # _generatePlot(axes, exp_paths, bounds, feats)
        # f.set_size_inches((width, height), forward=False)
        # axes.set_title(f"{problem} ({feats})")
        # plt.savefig(f'{savepath}/{feats}-learning-curve-allParams')

        bounds = []

        f, axes = plt.subplots(1)
        fltr = lambda r: r.params.get('eta', 1) == 1
        _generatePlot(axes, exp_paths, bounds, feats, fltr)
        f.set_size_inches((width, height), forward=False)

        axes.set_title(f"{problem} ({feats})")
        #axes.set_yscale('log')
        plt.savefig(f'{savepath}/RandomWalk-{feats}-learning-curve.png')
        plt.savefig(f'{savepath}/RandomWalk-{feats}-learning-curve.pdf')

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    if experiment_is("Baird", exp_paths):
        problem = "Baird"
        plotMDP(exp_paths, getSavePath(problem), problem, xlim=[0, 3000], ylim=[0,4])
    elif experiment_is("Boyan", exp_paths):
        problem = "Boyan"
        plotMDP(exp_paths, getSavePath(problem), problem, xlim=[0,6000])
    elif experiment_is("RandomWalk", exp_paths):
        problem = "RandomWalk"
        plotEach(exp_paths, getSavePath(problem), problem)
    else:
        print(f"Experiment not found!")
        print(f"paths: {exp_paths}")

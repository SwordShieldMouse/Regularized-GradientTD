import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from analysis.learning_curve import generatePlot

def plotMDP(paths, savepath, title):
    prefix = "batch-" if '/batch/' in paths[0] else ""
    f, axes = plt.subplots(1)
    bounds = []

    generatePlot(axes, paths, bounds)

    os.makedirs(savepath, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    axes.set_title(title)
    plt.savefig(f'{savepath}/{prefix}learning-curve-allParams.png', dpi=100)

    f, axes = plt.subplots(1)

    bounds = []

    fltr = lambda r: r.params.get('eta', 1) == 1
    generatePlot(axes, EXP_PATHS, bounds, fltr)

    os.makedirs(SAVE_PATH, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    axes.set_title(title)
    plt.savefig(f'{savepath}/{prefix}learning-curve.png', dpi=100)

def getConfigs(path):
    return list(filter(lambda p: p.endswith(".json"), os.listdir(path)))

def getSavePath(expname):
    return os.path.join(os.getcwd(),f'figures/{expname}')

def experiment_is(name, exp_paths):
    return reduce(lambda prev, path: prev and name in path, exp_paths)

def plotEach(exp_paths, savepath, problem):
    width = 8
    height = (24/5)
    os.makedirs(savepath, exist_ok=True)

    def _generatePlot(ax, exp_paths, bounds, feats, fltr = None):
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

            b = plotBest(best, ax, label=alg, color=colors[alg], dashed=False)
            bounds.append(b)

    for feats in ['tabular','inverted','dependent']:
        bounds = []

        f, axes = plt.subplots(1)
        _generatePlot(axes, exp_paths, bounds, feats)
        f.set_size_inches((width, height), forward=False)
        axes.set_title(f"{problem} ({feats})")
        plt.savefig(f'{savepath}/{feats}-learning-curve-allParams.png', dpi=100)

        bounds = []

        f, axes = plt.subplots(1)
        fltr = lambda r: r.params.get('eta', 1) == 1
        _generatePlot(axes, exp_paths, bounds, feats, fltr)
        f.set_size_inches((width, height), forward=False)

        axes.set_title(f"{problem} ({feats})")
        plt.savefig(f'{savepath}/{feats}-learning-curve.png', dpi=100)

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    if experiment_is(exp_paths,"Baird"):
        problem="Baird"
        plotMDP(exp_paths, getSavePath(problem), problem)

    if experiment_is("Baird", exp_paths):
        problem = "Baird"
        plotMDP(exp_paths, getSavePath(problem), problem)
    elif experiment_is("Boyan", exp_paths):
        problem = "Boyan"
        plotMDP(exp_paths, getSavePath(problem), problem)
    elif experiment_is("RandomWalk", exp_paths):
        problem = "RandomWalk"
        plotEach(exp_paths, getSavePath(problem), problem)
    else:
        print(f"Experiment not found!")
        print(f"paths: {exp_paths}")

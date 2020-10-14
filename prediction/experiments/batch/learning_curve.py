import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

name = 'bakeoff'
problem = 'Baird'
# algorithms = ['td', 'tdc', 'vtrace', 'htd', 'regh_tdc']
# algorithms = ['td', 'gtd2', 'tdc', 'htd', 'regh_tdc']
# algorithms = ['td', 'tdc', 'regh_tdc']
algorithms = ['tdc', 'gtd2', 'regh_tdc', 'tdrcc', 'gaussiankernel', 'linearkernel']
stepsize = 'constant'

bestBy = 'auc'
show_unconst = False
lstd_baseline = False
window = 3
smoothing = 0.0
XMAX = 100

SMALL = 8
MEDIUM = 16
BIGGER = 20

plt.rc('font', size=SMALL)          # controls default text sizes
plt.rc('axes', titlesize=SMALL)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM)    # legend fontsize
plt.rc('figure', titlesize=BIGGER)  # fontsize of the figure title

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def generatePlotTTA(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, errorfile)
    const, unconst = tee(results)

    color = colors[exp.agent]
    label = exp.agent

    const = whereParameterEquals(const, 'batch_size', 4)
    const = whereParameterGreaterEq(const, 'ratio', 1)
    if 'ReghTDC' in label:
        const = whereParameterEquals(const, 'reg_h', 1)

    best_const = getBest(const, bestBy=bestBy)
    best_unconst = getBest(unconst, bestBy=bestBy)

    print(label, best_const.params)

    if show_unconst and best_const != best_unconst:
        b = plotBest(best_unconst, ax, window=window, smoothing=smoothing, label=label + '_unc', color=color, alpha=0.2, dashed=True)
        bounds.append(b)

    b = plotBest(best_const, ax, window=window, smoothing=smoothing, label=label, color=color, alpha=0.2, dashed=False)
    bounds.append(b)

def generatePlotSSA(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, errorfile)

    color = colors[exp.agent]
    label = exp.agent

    b = plot(results, ax, window=window, smoothing=smoothing, label=label, color=color, alpha=0.2, dashed=False, bestBy=bestBy)
    bounds.append(b)


if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()

    bounds = []

    if lstd_baseline:
        path = f'experiments/batch/{problem}/lstd.json'
        lstd_exp = loadExperiment(path)
        LSTD_res = loadResults(lstd_exp, errorfile)

        LSTD_best = getBest(LSTD_res)

        b = plotBest(LSTD_best, ax, window=window, color=colors['LSTD'], label='LSTD', alphaMain=0.5, dashed=True)
        bounds.append(b)

    for alg in algorithms:
        if stepsize == 'constant':
            exp_path = f'experiments/batch/{problem}/{alg}/{alg}.json'
        else:
            exp_path = f'experiments/batch/{problem}/{alg}/{alg}{stepsize}.json'

        if '_h' in alg or alg == 'td' or alg == 'vtrace':
            generatePlotSSA(ax, exp_path, bounds)
        else:
            generatePlotTTA(ax, exp_path, bounds)

    lower = min(map(lambda x: x[0], bounds)) * 0.9
    upper = max(map(lambda x: x[1], bounds)) * 1.05

    if lower < 0.01:
        lower = -0.01

    # ax.set_ylim([lower, upper])
    ax.set_ylim([0, 0.1])
    # ax.set_xlim([0, XMAX])

    # ax.set_xscale("log", basex=10)
    plt.legend()

    # plt.show()
    # exit()

    save_path = 'experiments/batch/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_learning-curve_{error}_{stepsize}_{problem}_{bestBy}.png', bbox_inches='tight', dpi=100)

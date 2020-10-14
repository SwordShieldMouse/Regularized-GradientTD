import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, getBestEnd, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment


error = 'rmspbe'

# name = 'bakeoff'
# problem = 'SmallChainDependent4060'
# # algorithms = ['tdc', 'gtd2']
# algorithms = ['tdc', 'gtd2', 'htd']
# baselines = ['regh_tdc', 'td']
# stepsize = 'constant'
# param = 'ratio'

# problem = 'SmallChainDependent4060'
# # algorithms = ['tdc', 'htd']
# # algorithms = ['tdc', 'gtd2', 'regh_tdc']
# algorithms = ['tdc', 'td', 'gtd2', 'htd', 'vtrace', 'regh_tdc']
# # algorithms = ['tdc', 'td', 'gtd2', 'regh_tdc']
# baselines = []
# stepsize = 'adagrad'
# param = 'alpha'

name = 'beta-sensitivity'
problem = 'Boyan'
algorithms = ['regh_tdc']
baselines = ['td', 'tdc']
stepsize = 'adagrad'
param = 'reg_h'

# name = 'bakeoff'
# problem = 'Baird'
# algorithms = ['replay_gtd2']
# baselines = []
# stepsize = 'constant'
# param = 'replay'

# name = 'broken-htd'
# problem = 'Baird'
# algorithms = ['tdc', 'htd', 'regh_tdc']
# stepsize = 'constant'

bestBy = 'auc'
show_unconst = False

stderr = False

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

    if 'ReghTDC' in exp.agent or 'TDRCC' in exp.agent:
        const = whereParameterEquals(const, 'ratio', 1)
        # const = whereParameterEquals(const, 'reg_h', 0.8)

    elif 'TDC' in exp.agent:
        const = whereParameterGreaterEq(const, 'ratio', 1)

    if show_unconst:
        b = plotSensitivity(unconst, param, ax, stderr=stderr, color=color, label=label + '_unc', bestBy=bestBy, dashed=True)
        bounds.append(b)

    b = plotSensitivity(const, param, ax, stderr=stderr, color=color, label=label, bestBy=bestBy)
    bounds.append(b)

def generatePlot(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, errorfile)

    color = colors[exp.agent]
    label = exp.agent

    b = plotSensitivity(results, param, ax, color=color, label=label, bestBy=bestBy)
    bounds.append(b)

def baseline(ax, exp_path, values, bounds):
    exp = loadExperiment(path)
    results = loadResults(exp, errorfile)

    if 'Regh' in exp.agent:
        results = whereParameterEquals(results, 'reg_h', 0.8)
        results = whereParameterEquals(results, 'ratio', 1)
    elif 'TDC' in exp.agent or 'GTD2' in exp.agent:
        results = whereParameterEquals(results, 'ratio', 1)

    if bestBy == 'end':
        metric = lambda m: np.mean(m[-int(m.shape[0] * .1):])
        best = getBestEnd(results)
    elif bestBy == 'auc':
        metric = np.mean
        best = getBest(results)

    color = colors[exp.agent]
    label = exp.agent

    m = metric(best.mean())
    low = min(values)
    high = max(values)
    ax.hlines(m, low, high, color=color, label=label, linewidth=4, linestyle=':')

    bounds.append((m, m))

if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()

    bounds = []

    for alg in algorithms:
        if stepsize == 'constant':
            a = alg.replace('replay_', '')
            exp_path = f'experiments/icml_2020/{problem}/{alg}/{a}.json'
        else:
            exp_path = f'experiments/icml_2020/{problem}/{alg}/{alg}{stepsize}.json'

        if alg != 'td' and alg != 'vtrace':
            generatePlotTTA(ax, exp_path, bounds)
        else:
            generatePlot(ax, exp_path, bounds)

    tmp = loadExperiment(exp_path)
    param_values = tmp._d['metaParameters'][param]

    for base in baselines:
        if stepsize == 'constant':
            path = f'experiments/icml_2020/{problem}/{base}/{base}.json'
        else:
            path = f'experiments/icml_2020/{problem}/{base}/{base}{stepsize}.json'

        baseline(ax, path, param_values, bounds)

    lower = min(map(lambda x: x[0], bounds)) * 0.9
    upper = max(map(lambda x: x[1], bounds)) * 1.05

    if lower < 0.01:
        lower = -0.01

    ax.set_ylim([lower, upper])
    # ax.set_ylim([lower, 1])
    ax.set_xscale("log", basex=2)

    plt.legend()
    plt.show()
    exit()

    save_path = 'experiments/icml_2020/plots'
    os.makedirs(save_path, exist_ok=True)

    # merp. backwards compatibility with old file names
    if param == 'ratio':
        param = 'eta'

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{param}-sensitivity_{error}_{stepsize}_{problem}_{bestBy}.pdf', bbox_inches='tight', dpi=100)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, getBestEnd, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment


error = 'rmspbe'

problem = 'SmallChainDependent4060'
# algorithms = ['tdc', 'htd']
# algorithms = ['tdc', 'gtd2', 'regh_tdc', 'tdrcc', 'gaussiankernel', 'linearkernel']
algorithms = ['tdc', 'td', 'gtd2', 'regh_tdc', 'tdrcc', 'gaussiankernel', 'linearkernel']
# algorithms = ['tdc', 'td', 'gtd2', 'regh_tdc']
stepsize = ''


bestBy = 'auc'

stderr = False

SMALL = 8
MEDIUM = 16
BIGGER = 20
BIGGEST = 25

plt.rc('font', size=SMALL)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGEST)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM)    # legend fontsize
plt.rc('figure', titlesize=BIGGER)  # fontsize of the figure title
if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def rename(name):
    return {
        'ReghTDC': 'TDRC',
        'TDRCC': 'TDC++',
    }.get(name, name)

def addUpdateParam(results):
    for r in results:
        for i in list(range(2, 20)) + list(range(20, 300, 25)):
            new = r.clone()
            new.params['updates'] = i + 1
            new.reducer(lambda m, i=i: m[0 : i + 1] if type(m) is not float else np.nan)
            yield new

def generatePlot(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, errorfile)
    results = addUpdateParam(results)

    color = colors[exp.agent]
    label = rename(exp.agent)

    b = plotSensitivity(results, "updates", ax, color=color, label=label, bestBy=bestBy)
    bounds.append(b)

if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()

    bounds = []

    for alg in algorithms:
        if stepsize == 'constant':
            a = alg.replace('replay_', '')
            exp_path = f'experiments/batch/{problem}/{alg}/{a}.json'
        else:
            exp_path = f'experiments/batch/{problem}/{alg}/{alg}{stepsize}.json'

        generatePlot(ax, exp_path, bounds)

    lower = min(map(lambda x: x[0], bounds)) * 0.9
    upper = max(map(lambda x: x[1], bounds)) * 1.05

    if lower < 0.01:
        lower = -0.01

    ax.set_ylim([lower, upper])
    # ax.set_ylim([lower, 1])
    ax.set_xscale("log", basex=10)

    plt.xlabel('Number of Updates')
    # plt.ylabel('RMSPBE')


    # plt.legend()
    # plt.show()
    # exit()

    save_path = 'experiments/batch/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/update-sensitivity_{error}_{stepsize}_{problem}_{bestBy}.pdf', bbox_inches='tight', dpi=100)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest
from src.utils.model import loadExperiment
from src.analysis.colormap import colors


error = 'rmspbe'

name = 'all'
problems = ['SmallChainTabular4060', 'SmallChainInverted4060', 'SmallChainDependent4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc', 'vtrace', 'htd', 'td', 'regh_tdc', 'tdrcc']
stepsize = ''

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'


def indexOf(arr, e):
    for i, a in enumerate(arr):
        if a == e:
            return i

    return None

if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()
    table = np.zeros((len(algorithms), len(problems), 2))

    for i, alg in enumerate(algorithms):
        for j, problem in enumerate(problems):
            exp_path = f'experiments/icml_2020/{problem}/{alg}/{alg}{stepsize}.json'
            try:
                exp = loadExperiment(exp_path)
            except:
                continue

            results = loadResults(exp, errorfile)

            if alg == 'regh_tdc':
                results = whereParameterEquals(results, 'ratio', 1)
                results = whereParameterEquals(results, 'reg_h', 0.8)

            if alg == 'tdc':
                results = whereParameterGreaterEq(results, 'ratio', 1)

            if alg == 'tdrcc':
                results = whereParameterGreaterEq(results, 'reg_h', 0.1)

            best = getBest(results)
            metric = np.mean

            # best = getBestEnd(results)
            # metric = lambda m: np.mean(m[-(int(len(m))):])

            mean = metric(best.mean())
            stderr = metric(best.stderr())
            print(alg, problem, mean, best.params)

            table[i, j] = [mean, stderr]

    htd_idx = indexOf(algorithms, 'htd')
    vtrace_idx = indexOf(algorithms, 'vtrace')
    td_idx = indexOf(algorithms, 'td')
    our_idx = indexOf(algorithms, 'regh_tdc')
    for j, problem in enumerate(problems):
        table[:, j, 0] = table[:, j, 0] / table[our_idx, j, 0]

        if htd_idx is not None and table[htd_idx, j, 0] == 0:
            table[htd_idx, j] = table[td_idx, j]

        if vtrace_idx is not None and table[vtrace_idx, j, 0] == 0:
            table[vtrace_idx, j] = table[td_idx, j]

    offset = -3
    for i, p in enumerate(problems):
        offset += 3
        for j, a in enumerate(algorithms):
            x = i * len(algorithms) + j + offset
            ax.bar(x, table[j, i, 0], yerr=table[j, i, 1], color=colors[a.upper()], tick_label='')

    ax.set_ylim([.6, 2.5])
    # ax.set_ylim([-.7, .2])

    plt.show()
    exit()

    save_path = 'experiments/icml_2020/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 32
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_bar-plot_{stepsize}_{error}.pdf', bbox_inches='tight', dpi=100)

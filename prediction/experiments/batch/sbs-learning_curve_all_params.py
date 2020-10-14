import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, where, whereParameterEquals, whereParameterGreaterEq, whereParameterLesserEq, getBest, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

problems = ['SmallChainTabular4060', 'SmallChainInverted4060', 'SmallChainDependent4060', 'Baird', 'Boyan']

algorithms = ['td', 'regh_tdc', 'tdc', 'gtd2', 'tdrcc', 'gaussiankernel', 'linearkernel']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

bounds = {
    'SmallChainTabular4060': 0.07,
    'SmallChainDependent4060': 0.03,
    'SmallChainInverted4060': 0.07,
    'Boyan': 8,
    'Baird': 4,
}

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)

        color = colors[exp.agent]
        label = exp.agent

        results = whereParameterEquals(results, 'batch_size', 4)
        results = whereParameterLesserEq(results, 'ratio', 8)
        results = whereParameterLesserEq(results, 'alpha', 0.5)
        results = where(results, lambda r: r.params.get('ratio', 1) * r.params['alpha'] <= 1)

        if 'ReghTDC' in label:
            results = whereParameterEquals(results, 'reg_h', 1)
            results = whereParameterEquals(results, 'ratio', 1)

        elif 'TDRCC' in label:
            results = whereParameterEquals(results, 'reg_h', 0.8)
            results = whereParameterEquals(results, 'ratio', 1)

        elif 'TDC' in label:
            results = whereParameterGreaterEq(results, 'ratio', 1)

        left, right = tee(results)

        best_line = getBest(right).mean()
        best = np.mean(best_line)

        for result in left:
            # print(label, result.params)
            shade = 0.12
            line = result.mean()
            if np.mean(line) == best:
                shade = 1

            plotBest(result, ax, label=label, color=color, alphaMain=shade, dashed=False)

        bounds.append(best_line[0])


if __name__ == "__main__":
    f, axes = plt.subplots(len(problems), len(algorithms))

    for i, problem in enumerate(problems):
        for j, alg in enumerate(algorithms):

            exp_paths = glob.glob(f'experiments/batch/{problem}/{alg}/{alg}.json')
            if len(exp_paths) == 0:
                continue

            generatePlot(axes[i, j], exp_paths, [])

            if i == 0:
                axes[i, j].set_title(f'{problem}\n{alg}')
            else:
                axes[i, j].set_title(f'{problem}')


        for j, _ in enumerate(algorithms):
            axes[i, j].set_ylim([0, bounds[problem]])


    # plt.show()
    # exit()

    save_path = 'experiments/batch/plots'
    os.makedirs(save_path, exist_ok=True)

    height = len(problems) * (24/5)
    width = len(algorithms) * 8
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/sbs-learning_curve_all_params_{error}.png', dpi=100)

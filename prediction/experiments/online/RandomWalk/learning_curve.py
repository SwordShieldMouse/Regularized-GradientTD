import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from src.analysis.learning_curve import plotBest
from src.analysis.colormap import colors
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults
from PyExpUtils.utils.arrays import first
from analysis.learning_curve import getBest

SAVE_PATH=os.path.join(os.getcwd(),"figures/RandomWalk/")

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_PATHS = list(map(lambda alg: os.path.join(FILE_DIR, f'{alg}.json'),['pfgtd','gtd2','tdc','tdrc', 'td','htd','vtrace']))

def generatePlot(ax, exp_paths, bounds, feats):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)
        alg = exp.agent

        results = loadResults(exp, 'rmspbe_summary.npy')
        sub_results = filter(lambda r: r.params["representation"]==feats, results)

        best = getBest(sub_results)
        print(f'best parameters ({feats}):', exp_path)
        print(best.params)

        b = plotBest(best, ax, label=alg, color=colors[alg], dashed=False)
        bounds.append(b)

if __name__ == "__main__":
    for feats in ['tabular','inverted','dependent']:
        f, axes = plt.subplots(1)

        bounds = []

        generatePlot(axes, EXP_PATHS, bounds, feats)

        os.makedirs(SAVE_PATH, exist_ok=True)

        width = 8
        height = (24/5)
        f.set_size_inches((width, height), forward=False)
        axes.set_title(f"RandomWalk ({feats})")
        plt.savefig(f'{SAVE_PATH}/{feats}-learning-curve.png', dpi=100)

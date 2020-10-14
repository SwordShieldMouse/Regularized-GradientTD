import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from functools import partial
from itertools import tee
from multiprocessing.pool import Pool

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals
from src.utils.model import loadExperiment

def columnReducer(col):
    return lambda m: m[:, col]

def generatePlot(exp_paths):
    b_fig = plt.figure("Mean Update Variance")
    w_fig = plt.figure("W Update Variance")
    h_fig = plt.figure("H Update Variance")
    b_ax = b_fig.add_subplot(1, 1, 1)
    w_ax = w_fig.add_subplot(1, 1, 1)
    h_ax = h_fig.add_subplot(1, 1, 1)

    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'variance_summary.npy')

        both, w, h = tee(results, 3)

        both = map(lambda r: r.reducer(columnReducer(0)), both)
        w = map(lambda r: r.reducer(columnReducer(1)), w)
        h = map(lambda r: r.reducer(columnReducer(2)), h)

        plot(both, b_ax)
        plot(w, w_ax)
        plot(h, h_ax)

    plt.show()
    exit()

    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}'
    os.makedirs(save_path, exist_ok=True)
    b_fig.savefig(f'{save_path}/both_variance-curve.pdf')
    w_fig.savefig(f'{save_path}/w_variance-curve.pdf')
    h_fig.savefig(f'{save_path}/h_variance-curve.pdf')

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plotBest, save
from src.analysis.results import loadResults, whereParameterEquals, getBest
from src.utils.model import loadExperiment
from src.utils.arrays import partition

LAMBDA = 0.1

exp_paths = sys.argv[1:]

ax = plt.gca()

exps = map(loadExperiment, exp_paths)
tdc, gtd2 = partition(exps, lambda exp: exp.agent.startswith('tdc'))

def getResultsAndBest(exps):
    all_results = []
    best_result = None
    for exp in exps:
        results = loadResults(exp)
        results = whereParameterEquals(results, 'lambda', LAMBDA)

        best = getBest(results)
        all_results.append(best)

        if best_result is None:
            best_result = best
        elif np.mean(best.mean()) < np.mean(best_result.mean()):
            best_result = best

    return all_results, best_result

tdc_results, best_tdc = getResultsAndBest(tdc)
gtd2_results, best_gtd2 = getResultsAndBest(gtd2)

for r in tdc_results:
    if r == best_tdc:
        plotBest(r, ax, 'blue', r.exp.agent, stderr=False)
    else:
        plotBest(r, ax, 'blue', '_', alphaMain=0.15, stderr=False)

for r in gtd2_results:
    if r == best_gtd2:
        plotBest(r, ax, 'red', r.exp.agent, stderr=False)
    else:
        plotBest(r, ax, 'red', '_', alphaMain=0.15, stderr=False)

# save(exp, 'lambda')
# ax.set_ylim([0, 1])
plt.show()

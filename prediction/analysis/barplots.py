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

def getBairdConfigs():
    return map(lambda alg: os.path.join("./experiments/online/Baird",f"{alg}.json"),["cwpfgtdsh","pfgtd","pfcombined_cwsh","gtd2","tdc","tdrc"])

def getBoyanConfigs():
    return map(lambda alg: os.path.join("./experiments/online/Boyan",f"{alg}.json"),["cwpfgtd","pfgtd","pfcombined_cw","gtd2","tdc","tdrc","td"])

def getRWConfigs():
    return map(lambda alg: os.path.join("./experiments/online/RandomWalk",f"{alg}.json"),["cwpfgtd","pfgtd","pfcombined_cw","gtd2","tdc","tdrc","td"])

def getMDPData(exp_paths,fltr):
    data = {}

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

    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)
        alg = exp.agent

        results = loadResults(exp, 'rmspbe_summary.npy')
        if fltr is not None:
            results = filter(fltr, results)

        best = _getBest(results)
        m, s, _ = best.load()
        data[alg] = (np.mean(m), np.mean(s))

    return data

def getRWData(exp_paths, fltr):
    data = {}

    def _getData(exp_paths, feats):
        for exp_path in exp_paths:
            exp = ExperimentModel.load(exp_path)
            alg = exp.agent

            results = loadResults(exp, 'rmspbe_summary.npy')
            sub_results = filter(lambda r: r.params["representation"]==feats, results)

            if fltr is not None:
                sub_results = filter(fltr, sub_results)

            best = getBest(sub_results)
            m,s,_ = best.load()
            data[alg] = (np.mean(m), np.mean(s))
        return data

    alldata = {}
    for feats in ['tabular','inverted','dependent']:
        alldata[feats] = _getData(exp_paths,  feats)
    return alldata

if __name__ == "__main__":
    fltrs = [
        (None, "bar_plots.pdf"),
        (lambda r: r.params.get('eta', 1) == 1, "bar_plots_allParams.pdf")
    ]

    for fltr, filename in fltrs:

        f, ax = plt.subplots(1)

        print("RW...")
        rwConfigs = getRWConfigs()
        data=getRWData(rwConfigs, fltr)

        print("Boyan...")
        boyanConfigs = getBoyanConfigs()
        data["Boyan"] = getMDPData(boyanConfigs, fltr)

        print("Baird...")
        bairdConfigs = getBairdConfigs()
        data["Baird"] = getMDPData(bairdConfigs, fltr)


        ref_alg = 'PFCombined'
        offset = -3
        prev = 0
        for i, problem in enumerate(data.keys()):
            offset+=3

            learner_data = data[problem]
            ref, _ = learner_data[ref_alg]
            for j, learner in enumerate(learner_data.keys()):
                x = prev + j + offset
                val, stderr = learner_data[learner]
                val, stderr = val / ref, stderr / ref
                ax.bar(x, val, yerr=stderr, color = colors[learner], tick_label=problem)
            prev += len(learner_data.keys())

        savepath = "figures/"
        width = 8
        height = (24/5)
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}.pdf")

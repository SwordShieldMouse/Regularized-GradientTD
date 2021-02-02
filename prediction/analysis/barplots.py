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

#ALL_ALGS = ["cwpfgtd","pfgtd","pfcombined_cw","pfgtd+","gtd2","tdc","tdrc", "td"]
#ALL_ALGS = ["cwpfgtd","pfgtd","pfgtd+","gtd2","tdc","tdrc", "td"]
ALL_ALGS = ["cwpfgtd","pfgtd","pfgtd+","gtd2","tdc","tdrc", "td"]
BAIRD_ALGS = ALL_ALGS[:-1]

def getBairdConfigs():
    return list(map(lambda alg: os.path.join("./experiments/online/Baird",f"{alg}.json"), BAIRD_ALGS))

def getBoyanConfigs():
    return list(map(lambda alg: os.path.join("./experiments/online/Boyan",f"{alg}.json"), ALL_ALGS))

def getRWConfigs():
    return list(map(lambda alg: os.path.join("./experiments/online/RandomWalk",f"{alg}.json"), ALL_ALGS))

def getMDPData(exp_paths,fltr,measure):

    def _getBest(results):
        best = first(results)
        bestVal = measure(best.load()[0])

        for r in results:
            if not np.isfinite(bestVal):
                best = r
                bestVal = measure(r.load()[0])
                continue
            a = r.load()[0]
            am = measure(a)
            if am < bestVal:
                best = r
                bestVal = am

        print(f"{bestVal} <= {best.params}")
        return best

    data = {}
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)
        alg = exp.agent

        results = loadResults(exp, 'rmspbe_summary.npy')
        if fltr is not None:
            results = filter(fltr, results)

        best = _getBest(results)
        m, s, _ = best.load()
        data[alg] = (measure(m), measure(s))

    return data

def getRWData(exp_paths, fltr,measure):

    def _getRWBest(results):
        best = first(results)
        bestVal = measure(best.load()[0])

        for r in results:
            if not np.isfinite(bestVal):
                best = r
                bestVal = measure(r.load()[0])
                continue
            a = r.load()[0]
            am = measure(a)
            if am < bestVal:
                best = r
                bestVal = am

        print(f"{bestVal} <= {best.params}")
        return best

    def _getRWData(exp_paths, feats):
        data={}
        for exp_path in exp_paths:
            exp = ExperimentModel.load(exp_path)
            alg = exp.agent

            results = loadResults(exp, 'rmspbe_summary.npy')
            sub_results = filter(lambda r: r.params["representation"]==feats, results)

            if fltr is not None:
                sub_results = filter(fltr, sub_results)

            best = _getRWBest(sub_results)
            m,s,_ = best.load()
            data[alg] = (measure(m), measure(s))
        return data

    alldata = {}
    for feats in ['tabular','dependent','inverted']:
        alldata[feats] = _getRWData(exp_paths,  feats)
    return alldata

def main():
    fltrs = [
        (lambda r: r.params.get('eta', 1) == 1, "barplots"),
        (None, "barplots_allParams"),
    ]
    measures = [
        (np.mean, "AUC"),
        (lambda d: d[-1], "FinalPerformance")
    ]

    for fltr, filename in fltrs:
        for measure, measurename in measures:

            f, ax = plt.subplots(1)

            print("RW...")
            rwConfigs = getRWConfigs()
            data=getRWData(rwConfigs, fltr, measure)

            print("Boyan...")
            boyanConfigs = getBoyanConfigs()
            data["Boyan"] = getMDPData(boyanConfigs, fltr, measure)

            # print("Baird...")
            # bairdConfigs = getBairdConfigs()
            # data["Baird"] = getMDPData(bairdConfigs, fltr, measure)


            ref_alg = 'GTD2'
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
                    ax.bar(x, val, yerr=stderr, color = colors[learner], tick_label='')
                prev += len(learner_data.keys())

            savepath = "figures/"
            width = 8
            height = (24/5)
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(f"{savepath}/{filename}_{measurename}.png")
            plt.savefig(f"{savepath}/{filename}_{measurename}.pdf")

if __name__ == "__main__":
    main()

import os
import sys
import glob
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

# from src.analysis.learning_curve import plotBest
from src.analysis.colormap import colors
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults
from PyExpUtils.utils.arrays import first



def collate_results(results):
    collated_m = []
    for r in results:
        # for Boyan's chain, rows are the runs, columns are the mspbe over time
        m = r.load().flatten()[-1]

        # get the auc
        collated_m.append(m)

    arr = np.array(collated_m)
    broken = np.argwhere(filter(lambda a: not np.isfinite(a) or np.isnan(a) , arr))
    arr[broken] = 1
    arr = np.minimum(arr,1)
    return  arr


# for each alg, collate all the results
def generate_cdf_plot(ax, exp_paths, stat_name, fltr=None):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)
        alg = exp.agent

        print(f"alg = {alg}")

        # results = loadResults(exp, 'rmspbe.npy')
        results = loadResults(exp, f'{stat_name}.npy')

        if fltr is not None:
            results = filter(fltr, results)

        collated = collate_results(results)
        # plot a cdf where the y-axis is proportion of runs and x-axis is the error
        curve = generate_cdf(collated)    
        
        ax.plot(curve[:, 1], curve[:, 0],  label=alg, color=colors[alg])

        # confidence intervals
        # from eq 4 of appendix C of https://arxiv.org/pdf/2006.16958.pdf
        delta = 0.05
        bound = np.sqrt(np.log(2 / delta) / 2 / collated.shape[0])
        lower =curve[:, 0] - bound
        upper =curve[:, 0] + bound
        ax.fill_between(curve[:, 1], lower, upper, color = colors[alg], alpha = 0.2)



def generate_cdf(arr):
    # input is an array of avg errors
    # output is a list of tuples of the form (proportion_of_runs, error_obtained)

    # sort by increasing error and then do a cumulative sum
    cdf = sorted(arr)
    n_total_runs = arr.shape[0]
    proportions = (np.arange(n_total_runs) + 1) / n_total_runs
    return np.array(list(zip(proportions, cdf)))

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    exp_name = "Nexting"

    for stat_name in ["smape_summary", "mse_summary"]:
        f, axes = plt.subplots(1)
        print(f"stat = {stat_name}")
        generate_cdf_plot(axes, exp_paths, stat_name)

        save_path = 'figures'
        os.makedirs(save_path, exist_ok=True)

        width = 8
        height = (24/5)
        f.set_size_inches((width, height), forward=False)
        axes.set_ylabel("Cumulative Probability")
        axes.set_xlabel("Average Error")
        axes.set_title(f"{exp_name} {stat_name}")
        axes.legend()
        plt.savefig(f'{save_path}/{exp_name}-{stat_name}-cdf-allRuns.pdf')
        plt.clf()

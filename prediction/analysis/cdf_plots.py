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


plt.rcParams.update({'font.size': 20,'legend.fontsize':16})

def collate_results(results):
    collated_m = []
    for r in results:
        # for Boyan's chain, rows are the runs, columns are the mspbe over time
        m = r.load()

        # get the auc
        collated_m.append(m)

    arr = np.concatenate(collated_m)
    broken = np.argwhere(filter(lambda a: not np.isfinite(a) or np.isnan(a) , arr))
    arr[broken] = 1e6
    return  arr


# for each alg, collate all the results
def generate_cdf_plot(ax, exp_paths, stat_name, fltr=None):
    bounds=[]
    for i,exp_path in enumerate(exp_paths):
        exp = ExperimentModel.load(exp_path)
        alg = exp.agent

        print(f"alg = {alg}")

        # results = loadResults(exp, 'rmspbe.npy')
        results = loadResults(exp, f'{stat_name}.npy')

        if fltr is not None:
            results = filter(fltr, results)

        collated = collate_results(results)

        lt1 = np.argwhere(collated>=10000)
        collated[lt1] = 10000
        #acceptable_data = collated[lt1].flatten()

        # plt.scatter(i + 0.5*(np.random.rand(len(acceptable_data))-0.5), acceptable_data, label=alg, facecolors='none',edgecolors=colors[alg])
        # print(f"{alg}: {1.0 - len(acceptable_data)/len(collated)}")

        # plot a cdf where the y-axis is proportion of runs and x-axis is the error
        curve = generate_cdf(collated)
        probs = curve[:,0]
        values = curve[:,1]

        finite = np.argwhere(np.isfinite(values))
        probs = probs[finite].flatten()
        values = values[finite].flatten()

        ax.plot(values, probs,  label=alg, color=colors[alg], linewidth=5.0)


        # confidence intervals
        # from eq 4 of appendix C of https://arxiv.org/pdf/2006.16958.pdf
        delta = 0.05
        bound = np.sqrt(np.log(2 / delta) / 2 / collated.shape[0])
        lower = probs - bound
        #lower = np.minimum(np.maximum(lower,0),1)
        upper = probs + bound
        #upper = np.minimum(np.maximum(probs + bound,0),1)
        ax.fill_between(values, lower, upper, color = colors[alg], alpha = 0.2)

        bounds.append(np.min(values))
    return min(bounds)



def generate_cdf(arr):
    # input is an array of avg errors
    # output is a list of tuples of the form (proportion_of_runs, error_obtained)

    # sort by increasing error and then do a cumulative sum
    cdf = sorted(arr)
    n_total_runs = arr.shape[0]
    proportions = (np.arange(n_total_runs) + 1) / n_total_runs
    return np.array(list(zip(proportions, cdf)))



def experiment_is(name, exp_paths):
    return reduce(lambda prev, path: prev and name in path, exp_paths)

def get_exp(exp_paths):
    for name in ["Baird","Boyan", "RandomWalk"]:
        if experiment_is(name, exp_paths):
            return name
    raise Exception("get_exp( ... ): experiment name not found!")

XLIM_U = {
    "Baird": 2,
    "Boyan": 1.0,
    "RandomWalk": 0.2
}

ALL_STATS = ["auc", "half_auc", "final_rmspbe", "median"]

def has_grid_exp(exp_paths):
    for p in exp_paths:
        if "experiments/cdf_grid"  in p:
            return True
    return False

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    exp_name = get_exp(exp_paths)
    for stat_name in ["auc","final_rmspbe"]:
        if exp_name != "RandomWalk":
            f, axes = plt.subplots(1)
            print(f"stat = {stat_name}")
            minx = generate_cdf_plot(axes, exp_paths, stat_name)

            save_path = 'figures'
            os.makedirs(save_path, exist_ok=True)

            width = 8
            height = (24/5)
            f.set_size_inches((width, height), forward=False)
            axes.set_ylabel("Cumulative Probability")
            axes.set_xlabel("Average Error")
            axes.set_title(f"{exp_name} {stat_name}")
            #axes.set_yscale("log")
            #axes.set_xscale('log')
            axes.set_xlim([minx,XLIM_U[exp_name]])
            axes.legend()

            outname = f'{save_path}/{exp_name}-{stat_name}-cdf'
            outname+='-grid' if has_grid_exp(exp_paths) else ''
            plt.savefig(f'{outname}.png')
            plt.savefig(f'{outname}.pdf')
            plt.clf()
        else:
            for feats in ["tabular", "dependent", "inverted"]:
                f, axes = plt.subplots(1)
                print(f"stat = {stat_name}")

                fltr = lambda r: r.params['representation'] == feats
                xmin=generate_cdf_plot(axes, exp_paths, stat_name, fltr = fltr)

                save_path = 'figures'
                os.makedirs(save_path, exist_ok=True)

                width = 8
                height = (24/5)
                f.set_size_inches((width, height), forward=False)
                axes.set_ylabel("Cumulative Probability")
                axes.set_title(f"{exp_name} ({feats}) {stat_name}")
                axes.set_xlim([xmin,XLIM_U[exp_name]])
                axes.legend()

                outname = f'{save_path}/{exp_name}-{feats}-{stat_name}-cdf'
                outname+='-grid' if has_grid_exp(exp_paths) else ''
                plt.savefig(f'{outname}.png')
                plt.savefig(f'{outname}.pdf')
                plt.clf()

import numpy as np

# source found at: https://github.com/andnp/RlGlue
from RlGlue import RlGlue
from utils.Collector import Collector
from utils.policies import actionArrayToPolicy, matrixToPolicy
from utils.rl_glue import RlGlueCompatWrapper
from utils.errors import buildRMSPBE

from environments.RandomWalk import RandomWalk, TabularRep, DependentRep, InvertedRep
from environments.Boyan import Boyan, BoyanRep
from environments.Baird import Baird, BairdRep

from agents.TD import TD
from agents.TDC import TDC, BatchTDC
from agents.HTD import HTD
from agents.GTD2 import GTD2, BatchGTD2
from agents.TDRC import TDRC
from agents.Vtrace import Vtrace

from agents.ParameterFree import ParameterFree, PFGTD, PFTDC, CWPFGTD, PFGTDUntrunc
from agents.GTD2MP import GTD2MP

import os

# --------------------------------
# Set up parameters for experiment
# --------------------------------

RUNS = 10
LEARNERS = [PFGTD, GTD2MP, GTD2, TDC, TDRC]
#LEARNERS = [PFGTD, PFGTDUntrunc,BatchGTD2, BatchTDC, GTD2, TDC, TDRC]#, GTD2, TDC]
#LEARNERS = [PFGTD,GTD2, TDC]

PROBLEMS = [
    # # 5-state random walk environment with tabular features
    # {
    #     'env': RandomWalk,
    #     'representation': TabularRep,
    #     # go LEFT 40% of the time
    #     'target': actionArrayToPolicy([0.4, 0.6]),
    #     # take each action equally
    #     'behavior': actionArrayToPolicy([0.5, 0.5]),
    #     'gamma': 1.0,
    #     'steps': 3000,
    #     # hardcode stepsizes found from parameter study
    #     'stepsizes': {
    #         'TD': 0.03125,
    #         'TDRC': 0.03125,
    #         'TDC': 0.0625,
    #         'BatchTDC': 0.0625,
    #         'GTD2': 0.03125,
    #         'GTD2MP': 0.03125,
    #         'BatchGTD2': 0.03125,
    #         'HTD': 0.03125,
    #         'Vtrace': 0.03125,
    #     }
    # },
    # # 5-state random walk environment with dependent features
    # {
    #     'env': RandomWalk,
    #     'representation': DependentRep,
    #     # go LEFT 40% of the time
    #     'target': actionArrayToPolicy([0.4, 0.6]),
    #     # take each action equally
    #     'behavior': actionArrayToPolicy([0.5, 0.5]),
    #     'gamma': 1.0,
    #     'steps': 3000,
    #     # hardcode stepsizes found from parameter study
    #     'stepsizes': {
    #         'TD': 0.03125,
    #         'TDRC': 0.03125,
    #         'TDC': 0.0625,
    #         'BatchTDC': 0.0625,
    #         'GTD2': 0.0625,
    #         'GTD2MP': 0.0625,
    #         'BatchGTD2': 0.0625,
    #         'HTD': 0.03125,
    #         'Vtrace': 0.03125,
    #     }
    # },
    # # 5-state random walk environment with inverted features
    # {
    #     'env': RandomWalk,
    #     'representation': InvertedRep,
    #     # go LEFT 40% of the time
    #     'target': actionArrayToPolicy([0.4, 0.6]),
    #     # take each action equally
    #     'behavior': actionArrayToPolicy([0.5, 0.5]),
    #     'gamma': 1.0,
    #     'steps': 3000,
    #     # hardcode stepsizes found from parameter study
    #     'stepsizes': {
    #         'TD': 0.125,
    #         'TDRC': 0.125,
    #         'TDC': 0.125,
    #         'BatchTDC': 0.125,
    #         'GTD2': 0.125,
    #         'GTD2MP': 0.125,
    #         'BatchGTD2': 0.125,
    #         'HTD': 0.125,
    #         'Vtrace': 0.125,
    #     }
    # },
    # # Boyan's chain
    # {
    #     'env': Boyan,
    #     'representation': BoyanRep,
    #     # go LEFT 40% of the time
    #     'target': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
    #     # take each action equally
    #     'behavior': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
    #     'gamma': 1.0,
    #     'steps': 10000,
    #     # hardcode stepsizes found from parameter study
    #     'stepsizes': {
    #         'TD': 0.0625,
    #         'TDRC': 0.0625,
    #         'TDC': 0.5,
    #         'BatchTDC': 0.5,
    #         'GTD2': 0.5,
    #         'GTD2MP': 0.5,
    #         'BatchGTD2': 0.5,
    #         'HTD': 0.0625,
    #         'Vtrace': 0.0625,
    #     }
    # },
    # Baird's Counter-example domain
    {
        'env': Baird,
        'representation': BairdRep,
        'target': actionArrayToPolicy([6/7, 1/7]),
        'behavior': actionArrayToPolicy([0., 1.]),
        'starting_condition': np.array([1, 1, 1, 1, 1, 1, 1, 10]),
        'gamma': 0.99,
        'steps': 20000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.00390625,
            'TDRC': 0.015625,
            'TDC': 0.00390625,
            'BatchTDC': 0.00390625,
            'GTD2': 0.00390625,
            'GTD2MP': 0.00390625,
            'BatchGTD2': 0.00390625,
            'HTD': 0.00390625,
            'Vtrace': 0.00390625,
        }
    },
]


COLORS = {
    'PFGTD': 'red',
    'PFGTDUntrunc':'blue',
    'PFTDC': 'pink',
    #'CWPFGTD': 'pink',
    'GTD2MP': 'black',
    'TD': 'blue',
    'BatchTDC': 'cyan',
    'TDC': 'green',
    'TDRC': 'orange',
    'BatchGTD2': 'black',
    'GTD2': 'grey',
    'Vtrace': 'red',
    'HTD': 'purple',
}

# -----------------------------------
# Collect the data for the experiment
# -----------------------------------

# a convenience object to store data collected during runs
collector = Collector()

def log_err(learner, err_fn, data_key):
    w = learner.getWeights()
    err = err_fn(w)

    # store the data in the "collector" until we need it for plotting
    collector.collect(data_key, err)


for run in range(RUNS):
    for problem in PROBLEMS:
        for Learner in LEARNERS:
            # for reproducibility, set the random seed for each run
            # also reset the seed for each learner, so we guarantee each sees the same data
            np.random.seed(run)

            # build a new instance of the environment each time
            # just to be sure we don't bleed one learner into the next
            Env = problem['env']
            env = Env()

            target = problem['target']
            behavior = problem['behavior']

            Rep = problem['representation']
            rep = Rep()

            data_key = f'{Env.__name__}-{Rep.__name__}-{Learner.__name__}'
            print(run, *data_key.split('-'))

            # build the X, P, R, and D matrices for computing RMSPBE
            X, P, R, D = env.getXPRD(target, rep)
            RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

            # build a new instance of the learning algorithm
            if issubclass(Learner, ParameterFree):
                learner = Learner(rep.features(), {
                    'gamma': problem['gamma'],
                    'wealth': 1.0,
                    'hint': 1.0,
                    'beta': 0.0
                })
            else:
                learner = Learner(rep.features(), {
                    'gamma': problem['gamma'],
                    'alpha': problem['stepsizes'][Learner.__name__],
                    'beta': 1,
                })

            # build an "agent" which selects actions according to the behavior
            # and tries to estimate according to the target policy
            agent = RlGlueCompatWrapper(learner, behavior, target, rep.encode)


            # for Baird's counter-example, set the initial value function manually
            if problem.get('starting_condition') is not None:
                u = np.array(problem['starting_condition'].copy(),dtype='float64')
                learner.initWeights(u)

            # Log initial error
            log_err(learner, RMSPBE, data_key)

            # build the experiment runner
            # ties together the agent and environment
            # and allows executing the agent-environment interface from Sutton-Barto
            glue = RlGlue(agent, env)


            # start the episode (env produces a state then agent produces an action)
            glue.start()
            for step in range(problem['steps']):
                # interface sends action to env and produces a next-state and reward
                # then sends the next-state and reward to the agent to make an update
                _, _, _, terminal = glue.step()

                # when we hit a terminal state, start a new episode
                if terminal:
                    glue.start()

                # evaluate the RMPSBE
                # subsample to reduce computational cost
                if (step+1) % 100 == 0:
                    log_err(learner, RMSPBE, data_key)

            # tell the data collector we're done collecting data for this env/learner/rep combination
            collector.reset()

for i, problem in enumerate(PROBLEMS):
    for j, Learner in enumerate(LEARNERS):
        env = problem['env'].__name__
        agent = problem['learner'].__name__
        data = collector.getState(f'{env}-{rep}-{learner}')

        save_context = exp



# ---------------------
# Plotting the bar plot
# ---------------------
import matplotlib.pyplot as plt

ax = plt.gca()
f = plt.gcf()

# get TDRC's baseline performance for each problem
baselines = [None] * len(PROBLEMS)
for i, problem in enumerate(PROBLEMS):
    env = problem['env'].__name__
    rep = problem['representation'].__name__

    mean_curve, _, _ = collector.getStats(f'{env}-{rep}-TDRC')

    # compute TDRC's AUC
    baselines[i] = mean_curve.mean()

# how far from the left side of the plot to put the bar
offset = -3
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    offset += 3
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__
        env = problem['env'].__name__
        rep = problem['representation'].__name__

        x = i * len(LEARNERS) + j + offset

        mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')
        auc = mean_curve.mean()
        auc_stderr = stderr_curve.mean()

        relative_auc = auc / baselines[i]
        relative_stderr = auc_stderr / baselines[i]

        ax.bar(x, relative_auc, yerr=relative_stderr, color=COLORS[learner], tick_label='')

# plt.show()
fig_dir = "figures/prediction/"
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(f"{fig_dir}auc.png")

# =========================
# --- FINAL PERFORMANCE ---
# =========================
f, ax = plt.subplots()

# get TDRC's baseline performance for each problem
baselines = [None] * len(PROBLEMS)
for i, problem in enumerate(PROBLEMS):
    env = problem['env'].__name__
    rep = problem['representation'].__name__

    mean_curve, _, _ = collector.getStats(f'{env}-{rep}-TDRC')

    # compute TDRC's AUC
    baselines[i] = mean_curve.mean()

# how far from the left side of the plot to put the bar
offset = -3
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    offset += 3
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__
        env = problem['env'].__name__
        rep = problem['representation'].__name__

        x = i * len(LEARNERS) + j + offset

        mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')
        auc = mean_curve[-1]
        auc_stderr = stderr_curve[-1]

        relative_auc = auc / baselines[i]
        relative_stderr = auc_stderr / baselines[i]

        ax.bar(x, relative_auc, yerr=relative_stderr, color=COLORS[learner], tick_label='')

# plt.show()
fig_dir = "figures/prediction/"
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(f"{fig_dir}final.png")

# =======================
# --- LEARNING CURVES ---
# =======================
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    fig, ax = plt.subplots()

    env = problem['env'].__name__
    rep = problem['representation'].__name__
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__

        mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')

        ax.plot(np.arange(len(mean_curve)), mean_curve, color=COLORS[learner], label=learner)
        ax.fill_between(np.arange(len(mean_curve)), mean_curve - stderr_curve, mean_curve + stderr_curve, alpha=0.2, color=COLORS[learner])

    ax.set_title(f"{env} {rep}")
    ax.legend()

    fig_dir = "figures/prediction/learning_curves/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}{env}_{rep}.png")

# ========================================
# --- LEARNING CURVES (INDIVIDUAL RUNS)---
# ========================================
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    fig, ax = plt.subplots()

    env = problem['env'].__name__
    rep = problem['representation'].__name__
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__

        all_data = collector.all_data[f'{env}-{rep}-{learner}']

        for d in all_data:
            data = np.array(d)
            ax.plot(np.arange(len(d)), d,  color=COLORS[learner], alpha=0.2)

        mean_curve = np.mean(np.array(all_data), axis=0)
        ax.plot(np.arange(len(mean_curve)), mean_curve,  color=COLORS[learner], label=learner, linewidth=2.0)

    ax.legend()
    ax.set_title(f"{env} {rep}")

    fig_dir = "figures/prediction/learning_curves/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}{env}_{rep}_allRuns.png")
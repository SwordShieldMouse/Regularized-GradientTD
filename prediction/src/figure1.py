import numpy as np

from RlGlue import RlGlue
from src.utils.Collector import Collector
from src.utils.policies import actionArrayToPolicy, matrixToPolicy
from src.utils.rl_glue import RlGlueCompatWrapper
from src.utils.errors import buildRMSPBE

from src.experiment import ExperimentModel
from src.problems.registry import getProblem

import os

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

# -----------------------------------
# Collect the data for the experiment
# -----------------------------------

# a convenience object to store data collected during runs
collector = Collector()
num_params = exp.permutations()
for run in range(RUNS):
    # for reproducibility, set the random seed for each run
    # also reset the seed for each learner, so we guarantee each sees the same data
    np.random.seed(run)

    # build a new instance of the environment each time
    # just to be sure we don't bleed one learner into the next
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx + run * num_params)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    agent_wrapper = problem.getAgent()
    glue = RlGlue(agent_wrapper, env)

    # build the experiment runner
    # ties together the agent and environment
    # and allows executing the agent-environment interface from Sutton-Barto
    glue = RlGlue(agent, env)

    rmsve, rmspbe = problem.evaluateStep({
        'step': step,
        'reward': r,
    })

    collector.collect('errors', rmsve)
    collector.collect('rmspbe', rmspbe)

    # start the episode (env produces a state then agent produces an action)
    glue.start()
    for step in range(problem.getSteps()):
        # interface sends action to env and produces a next-state and reward
        # then sends the next-state and reward to the agent to make an update
        _, _, _, terminal = glue.step()

        rmsve, rmspbe = problem.evaluateStep({
            'step': step,
            'reward': r,
        })

        collector.collect('errors', rmsve)
        collector.collect('rmspbe', rmspbe)

        # when we hit a terminal state, start a new episode
        if terminal:
            glue.start()

        # evaluate the RMPSBE
        # subsample to reduce computational cost
        if (step+1) % 100 == 0:
            rmsve, rmspbe = prob
            collector.collect('rmspbe', RMSPBE(agent.getWeights()))

    # tell the data collector we're done collecting data for this env/learner/rep combination
    collector.reset()

data = collector.getStats('rmspbe')

save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('rmspbe_summary.npy'), return_data)

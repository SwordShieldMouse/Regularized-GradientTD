import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.problems.registry import getProblem
from src.experiment import ExperimentModel
from src.utils.Collector import Collector
# ====================================================================================

RUNS = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

print(sys.argv[2])

# -----------------------------------
# Collect the data for the experiment
# -----------------------------------

# a convenience object to store data collected during runs
collector = Collector()
for run in range(RUNS):
    # for reproducibility, set the random seed for each run
    # also reset the seed for each learner, so we guarantee each sees the same data
    np.random.seed(run)

    # build a new instance of the environment each time
    # just to be sure we don't bleed one learner into the next
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    env = problem.getEnvironment()
    rep = problem.getRepresentation()
    agent_wrapper = problem.getAgent()

    glue = RlGlue(agent_wrapper, env)

    # eval first step
    _, rmspbe = problem.evaluateStep()
    #collector.collect('rmsve',rmsve)
    collector.collect('rmspbe', rmspbe)

    # start the episode (env produces a state then agent produces an action)
    glue.start()
    for step in range(problem.getSteps()):
        # interface sends action to env and produces a next-state and reward
        # then sends the next-state and reward to the agent to make an update
        _, _, _, terminal = glue.step()

        # when we hit a terminal state, start a new episode
        if terminal:
            glue.start()

        # evaluate the RMPSBE
        # subsample to reduce computational cost
        rmsve, rmspbe = problem.evaluateStep()
        #collector.collect('rmsve',rmsve)
        collector.collect('rmspbe', rmspbe)

    # tell the data collector we're done collecting data for this env/learner/rep combination
    collector.reset()

save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('rmspbe_summary.npy'), np.array(collector.getStats('rmspbe')))
#np.save(save_context.resolve('rmsve_summary.npy'), np.array(collector.getStats('rmsve')))

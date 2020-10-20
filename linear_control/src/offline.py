import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
from src.utils.rlglue import OfflineOneStepWrapper, PolicyWrapper
from src.utils.policies import Policy

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

EVERY = 5000

collector = Collector()

def evalPolicy(agent, env, rep, nsteps):

    # Wrap agent in policy wrapper so that we don't
    # update the params during evaluation
    wrapper = PolicyWrapper(agent,rep)
    glue = RlGlue(wrapper, env)

    # Run the experiment
    last_steps = 0
    last_rewards = 0
    ncomplete = 0
    running = True
    while running:
        last_steps = glue.num_steps
        glue.total_reward = 0
        running = glue.runEpisode(nsteps)
        ncomplete+=1

        # do this to avoid underestimating the last episode
        collect = glue.total_reward
        if not running:
            collect = last_rewards

        # rewards = [collect] * (glue.num_steps - last_steps)
        # collector.concat('return', rewards)
        # collector.collect('steps', glue.num_steps - last_steps)

        last_rewards = glue.total_reward
        print(f"{ncomplete} episodes completed")


for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    max_steps = problem.max_steps

    agent = problem.getAgent()
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    wrapper = OfflineOneStepWrapper(agent, problem.behavior, problem.getGamma(), rep)

    for updates in range(EVERY, max_steps, EVERY):
        # Run the experiment
        glue = RlGlue(wrapper, env)
        running = True
        last_steps=0
        while running:
            running = glue.runEpisode(updates)
        print(f"done {updates} updates")

        evalPolicy(agent, env, rep, max_steps)
    collector.reset()

return_data = np.array(collector.getStats('return'), dtype='object')
step_data = np.array(collector.getStats('steps'), dtype='object')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('return_summary.npy'), return_data)
np.save(save_context.resolve('step_summary.npy'), step_data)

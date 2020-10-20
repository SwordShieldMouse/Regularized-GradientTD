import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
from src.utils.rlglue import PolicyWrapper
from src.utils.SampleGenerator import SampleGenerator
from src.agents.BaseAgent import PolicyWrapper

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

collector = Collector()
for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    max_steps = problem.max_steps
    evalEpisodes = problem.evalEpisodes
    evalSteps = problem.evalSteps

    agent = problem.getAgent()
    env = problem.getEnvironment()

    wrapper = OneStepWrapper(agent, problem.getGamma(), problem.getRepresentation())

    glue = RlGlue(wrapper, env)

    if run % 50 == 0:
        generator = SampleGenerator(problem)
        generator.generate(num=1e6)

    # Run the experiment
    glue.start()
    broke = False
    for step in range(steps):
        agent.batch_update(generator)

        if step in evalSteps:
            av_rewards = 0.0
            av_steps = 0.0
            for n in evalEpisodes:
                glue.runEpisode(max_steps)
                av_rewards += 1.0/(n+1) * (glue.total_reward - av_rewards)
                av_steps += 1.0/(n+1) * (glue.num_steps - av_steps)
            collector.collect("return", av_rewards)
            collector.collect("steps", av_steps)
    collector.reset()

return_data = np.array(collector.getStats('return'), dtype='object')
step_data = np.array(collector.getStats('steps'), dtype='object')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('return_summary.npy'), return_data)
np.save(save_context.resolve('step_summary.npy'), step_data)

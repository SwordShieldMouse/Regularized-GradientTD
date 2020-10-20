import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
from src.utils.rlglue import PolicyWrapper, OneStepWrapper
from src.utils.SampleGenerator import SampleGenerator
from src.utils.policies import Policy

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
    epochs = problem.epochs

    agent = problem.getAgent()
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    if run % 50 == 0:
        generator = SampleGenerator(problem)
        generator.target = Policy(lambda s: agent.policy(rep.encode(s)))
        generator.generate(num=1e4)


    # Run the experiment
    prev = 0
    for epoch in range(epochs):

        generator = SampleGenerator(problem)
        generator.target = Policy(lambda s: agent.policy(rep.encode(s)))
        generator.generate(num=1e4)

        agent.batch_update(generator, evalSteps)

        print(f"Epoch {epoch+1}/{epochs}")

        wrapper = PolicyWrapper(agent, rep)
        glue = RlGlue(wrapper, env)

        last_steps = 0
        last_rewards = 0
        running=True
        while running:
            last_steps = glue.num_steps
            glue.total_reward = 0
            running = glue.runEpisode(50000)

            # do this to avoid underestimating the last episode
            collect = glue.total_reward
            if not running:
                collect = last_rewards

            rewards = [collect] * (glue.num_steps - last_steps)
            collector.concat('return', rewards)
            collector.collect('steps', glue.num_steps - last_steps)

            last_rewards = glue.total_reward
            print(run, last_rewards)
        # av_rewards = 0.0
        # av_steps = 0.0
        # wrapper = PolicyWrapper(Policy(lambda s: problem.getAgent().policy(rep.encode(s))))
        # glue = RlGlue(wrapper, env)
        # for n in range(evalEpisodes):
        #     glue.runEpisode(max_steps)
        #     av_rewards += 1.0/(n+1) * (glue.total_reward - av_rewards)
        #     av_steps += 1.0/(n+1) * (glue.num_steps - av_steps)
        # collector.concat("return", [av_rewards]*num)
        # collector.concat("steps", [av_steps]*num)
    collector.reset()

return_data = np.array(collector.getStats('return'), dtype='object')
step_data = np.array(collector.getStats('steps'), dtype='object')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('return_summary.npy'), return_data)
np.save(save_context.resolve('step_summary.npy'), step_data)

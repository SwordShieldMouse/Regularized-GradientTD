import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
from src.utils.rlglue import PolicyWrapper, OneStepWrapper
from src.utils.SampleGenerator import SequentialSampleGenerator
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
    evalSteps = problem.evalSteps
    epochs = problem.epochs

    agent = problem.getAgent()
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    if run % 50 == 0:
        if not os.path.isfile("batchData.npy"):
            print("generating data...", end=' ');
            sys.stdout.flush()

            generator = SequentialSampleGenerator(problem)
            generator.generate(num=1e6)
            np.save('batchData.npy', generator._generated)
        else:
            print("Loading data...", end=' '); sys.stdout.flush()
            generator = SequentialSampleGenerator(problem)
            generator._generated = np.load('batchData.npy', allow_pickle=True)

        print("done!")



    # Run the experiment
    prev = 0
    Rperepoch = []
    for epoch in range(epochs):
        agent.batch_update(generator, evalSteps)

        print(f"Epoch {epoch+1}/{epochs}")

        wrapper = PolicyWrapper(agent, rep)
        glue = RlGlue(wrapper, env)

        last_steps = 0
        last_rewards = 0
        running=True
        Rperep = []
        while running:
            last_steps = glue.num_steps
            glue.total_reward = 0
            running = glue.runEpisode(max_steps)

            # do this to avoid underestimating the last episode
            collect = glue.total_reward
            if not running:
                collect = last_rewards
            print(f'collect: {collect}, running: {running}')
            Rperep.append(collect)

            last_rewards = glue.total_reward

        print(Rperep)
        avg = np.mean(Rperep)
        print(f'run {run}, epoch {epoch}: {avg}')
        Rperepoch.append(avg)

    collector.collect('return', Rperepoch)

    collector.reset()

return_data = np.array(collector.getStats('return'), dtype='object')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('return_summary.npy'), return_data)

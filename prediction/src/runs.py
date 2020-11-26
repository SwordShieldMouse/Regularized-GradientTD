import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel

from src.utils.rlglue import OffPolicyWrapper

from src.problems.registry import getProblem
from src.utils.errors import partiallyApplyMSPBE, MSPBE
from src.utils.Collector import Collector

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    sys.exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

EVERY = 100

collector = Collector()
for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    env = problem.getEnvironment()
    rep = problem.getRepresentation()
    agent = problem.getAgent()

    mu = problem.behavior
    pi = problem.target

    # takes actions according to mu and will pass the agent an importance sampling ratio
    # makes sure the agent only sees the state passed through rep.encode.
    # agent does not see raw state
    agent_wrapper = OffPolicyWrapper(agent, problem.getGamma(), mu, pi, rep.encode)

    X = rep.buildFeatureMatrix()
    P = env.buildTransitionMatrix(pi)
    R = env.buildAverageReward(pi)
    d = env.getSteadyStateDist(mu)

    # precompute matrices for cheaply computing MSPBE
    AbC = partiallyApplyMSPBE(X, P, R, d, problem.getGamma())

    # log initial err
    mspbe = MSPBE(agent.getWeights(), *AbC)
    collector.collect('rmspbe', np.sqrt(mspbe))

    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    glue.start()
    for step in range(exp.steps):
        # call agent.step and environment.step
        r, o, a, t = glue.step()

        mspbe = MSPBE(agent.getWeights(), *AbC)
        collector.collect('rmspbe', np.sqrt(mspbe))
        if step % EVERY == 0:
            print(np.sqrt(mspbe))

        # if terminal state, then restart the interface
        if t:
            glue.start()

    # tell the collector to start a new run
    collector.reset()

rmspbe_data = np.array(collector.getAllData('rmspbe'), dtype='object')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('rmspbe.npy'), rmspbe_data)
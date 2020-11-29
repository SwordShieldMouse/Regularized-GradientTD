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

if len(sys.argv) < 2:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    sys.exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = 0

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

    if 'alpha' in problem.params.keys():
        # HACK: the alpha in the config specifies the lower bound on
        # stepsize 2.0^-N, which the agent is instantiated with when Problem(exp,idx)
        # is called. Here we sample the step-size for algorithms which require tuning.
        agent.alpha = 2.0 ** np.random.uniform(agent.alpha, 0.0)

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

# Get matrix of learning curves. rows = runs, cols = MSPBE curve
rmspbe_data = np.array(collector.getAllData('rmspbe'), dtype='object')

# summary stats
auc = np.mean(rmspbe_data, axis=1)
half_auc = np.mean(rmspbe_data[:,rmspbe_data.shape[1]//2:], axis=1)
median = np.median(rmspbe_data, axis=1)
final = rmspbe_data[:,-1]

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('auc.npy'), auc)
np.save(save_context.resolve('half_auc.npy'), half_auc)
np.save(save_context.resolve('median.npy'), median)
np.save(save_context.resolve('final_rmspbe.npy'), final)

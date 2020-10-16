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
from src.utils.SampleGenerator import SampleGenerator

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    sys.exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

EVERY = 10

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
   
    # determine effective number of samples
    # we need 300 at most
    steps = exp.steps
    samples = steps // EVERY
    if samples > 300:
        EVERY = int(steps // 300)

    # TODO: why was this there?
    if run % 50 == 0:
        generator = SampleGenerator(problem)
        generator.generate(num=1e6)

    # Run the experiment
    glue.start()
    broke = False
    for step in range(steps):
        agent.batch_update(generator)

        if step % EVERY != 0:
            continue

        mspbe = MSPBE(agent.getWeights(), *AbC)


        collector.collect('rmspbe', np.sqrt(mspbe))

        # if we've diverged, just go ahead and give up
        # saves some computation and these runs are useless to me anyways
        if np.isnan(mspbe):
            collector.fillRest(np.nan, int(problem.getSteps() / EVERY))
            broke = True
            break

    # tell the collector to start a new run
    collector.reset()

    if broke:
        break

rmspbe_data = collector.getStats('rmspbe')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('rmspbe_summary.npy'), rmspbe_data)

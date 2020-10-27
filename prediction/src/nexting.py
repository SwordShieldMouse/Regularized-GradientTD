import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel

from src.utils.rlglue import NextingWrapper

from src.problems.registry import getProblem
from src.utils.errors import partiallyApplyMSPBE, MSPBE
from src.utils.Collector import Collector
from src.utils import Averages
from src.utils.Critterbot import loadReturns, computeReturns

EVERY = 100

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    sys.exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])


collector = Collector()
for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    env = problem.getEnvironment()
    rep = problem.getRepresentation()
    agent = problem.getAgent()

    sensorIdx = env.sensorIdx
    G = computeReturns(problem)
    var = np.var(G, ddof=1)

    if var == 0:
        # don't normalize zero-variance sensors
        var = 1.0


    # takes actions according to mu and will pass the agent an importance sampling ratio
    # makes sure the agent only sees the state passed through rep.encode.
    # agent does not see raw state
    agent_wrapper = NextingWrapper(agent, problem.getGamma(), rep.encode)

    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    s=glue.start()
    v = agent_wrapper.V(s)
    mse = Averages.Uniform((G[0]-v)**2)
    smape = Averages.Uniform(np.abs(v-G[0])/ (np.abs(v) + np.abs(G[0])))

    # log initial err/pred
    m = mse.get()
    collector.collect("nmse", m / var)
    collector.collect("mse", m)
    collector.collect("smape", smape.get())
    collector.collect("prediction", v)

    for step in range(1, exp.steps+1):
        # call agent.step and environment.step
        r, s, a, t = glue.step()

        v = agent_wrapper.V(s)
        mse.update((G[step]- v)**2)
        smape.update(np.abs(v-G[step])/ (np.abs(v) + np.abs(G[step])))

        if step % EVERY == 0:
            m= mse.get()
            err = m/ var
            smp = smape.get()
            collector.collect("mse", m)
            collector.collect("nmse", err)
            collector.collect("smape", smp)

            if runs == 1:
                collector.collect("prediction", v)
                collector.collect("Gt", G[step])
            print(f"Step {step} | mse: {m:0.5f} | nmse: {err:0.5f} | smape: {smp:0.5f}")

    # tell the collector to start a new run
    collector.reset()

if runs == 1:
    mse_data = np.array(collector.getRunData('mse'), dtype='object')
    nmse_data = np.array(collector.getRunData('nmse'), dtype='object')
    smape_data = np.array(collector.getRunData('smape'), dtype='object')
    prediction_data = np.array(collector.getRunData('prediction'), dtype='object')
    return_data = np.array(collector.getRunData('Gt'), dtype='object')
else:
    mse_data = np.array(collector.getStats('mse'), dtype='object')
    nmse_data = np.array(collector.getStats('nmse'), dtype='object')
    smape_data = np.array(collector.getStats('smape'), dtype='object')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('mse_summary.npy'), mse_data)
np.save(save_context.resolve('nmse_summary.npy'), nmse_data)
np.save(save_context.resolve('smape_summary.npy'), smape_data)

if runs == 1:
    np.save(save_context.resolve('prediction_summary.npy'), prediction_data)
    np.save(save_context.resolve('returns_summary.npy'), return_data)

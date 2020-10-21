import itertools
import numpy as np

from src.agents.ParameterFree import ParameterFree, PFGQ, PFGQ2

class PFCombination(BaseAgent):
    def __init__(self, features, actions, params):
        params['alpha'] = None
        super().__init__(features, actions, params)

    def applyUpdate(self, x, a, xp, r, gamma):
        for subagent in self.subagents:
            subagent.applyUpdate(x,a,xp,r,gamma)

    def getWeights(self):
        return sum([subagent.getWeights() for subagent in self.subagents])

class PFEnsemble(PFCombination):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        lambdas = params.get('lambdas', [])
        averages = params.get('averages', [])
        settings = itertools.product(lambdas, averages)

        W0 = params["wealth"] / length(settings)
        subparams = params.copy()
        subparams["wealth"] = W0 / length(settings)

        self.subagents = []
        for (lmda, av) in settings:
            subparams['lambda'] = lmda
            subparams['averaging'] = av
            self.subagents.append(PFGQ(features, actions, params))

class BootstrapPFGQ(PFCombination):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        self.N = params["N"]
        self.subagents = [PFGQ(features,actions,params) for _ in range(N)]
        self.actingAgent = np.random.randint(N)

    def applyUpdate(self, x, a, xp, r, gamma):
        super().applyUpdate(x, a, xp, r, gamma)

        if gamma==0:
            # change the acting agent at the end of an episode
            self.actingAgent = np.random.randint(self.N)

    def policy(self, x):
        return self.subagents[self.actingAgent].policy(x)

    def selectAction(self, x):
       return self.subagents[self.actingAgent].selectAction(x)

   def getWeights(self):
       return self.subaents[self.actingAgent].getWeights()

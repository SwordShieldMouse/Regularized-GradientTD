import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.representations.RandomCluster import RandomCluster, RandomOuterCluster
from src.problems.Chain import BaseChain, Policy5050, Policy4060, Policy1090, RepDependent, RepInverted, RepTabular
from src.environments.Chain import Chain as ChainEnv
from src.utils.rlglue import OffPolicyWrapper
from src.utils.policies import Policy

class SmallChain(BaseChain):
    def _getSize(self):
        return 5

    def _getdb(self):
        return np.array([0.111111, 0.222222, 0.333333, 0.222222, 0.111111, 0])

    def getSteps(self):
        return 3000

class RepRandomCluster:
    def _getRepresentation(self, n):
        size = getattr(self, 'metaParameters').get('size', int(n // 2) + 1)
        return RandomCluster(size, getattr(self, 'v_star'))

class RepRandomOuterCluster:
    def _getRepresentation(self, n):
        size = getattr(self, 'metaParameters').get('size', int(n // 2) + 1)
        return RandomOuterCluster(size, getattr(self, 'v_star'))

# ---------------
# -- Resultant --
# ---------------

class SmallChainDependent5050(Policy5050, RepDependent, SmallChain):
    pass

class SmallChainInverted5050(Policy5050, RepInverted, SmallChain):
    pass

class SmallChainTabular5050(Policy5050, RepTabular, SmallChain):
    pass

class SmallChainDependent4060(Policy4060, RepDependent, SmallChain):
    pass

class SmallChainInverted4060(Policy4060, RepInverted, SmallChain):
    pass

class SmallChainTabular4060(Policy4060, RepTabular, SmallChain):
    pass

class SmallChainRandomCluster5050(Policy5050, RepRandomCluster, SmallChain):
    def getSteps(self):
        return 10000

class SmallChainRandomCluster4060(Policy4060, RepRandomCluster, SmallChain):
    def getSteps(self):
        return 10000

class SmallChainRandomCluster1090(Policy1090, RepRandomCluster, SmallChain):
    def getSteps(self):
        return 10000

class SmallChainOuterRandomCluster1090(Policy1090, RepRandomOuterCluster, SmallChain):
    def getSteps(self):
        return 10000

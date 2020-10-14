import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.Chain import BaseChain, Policy2575, Policy4060, Policy5050, RepInverted, RepDependent, RepTabular
from src.environments.Chain import Chain as ChainEnv
from src.utils.rlglue import OffPolicyWrapper
from src.utils.policies import Policy

class MediumChain(BaseChain):
    def _getSize(self):
        return 9

    def _getdb(self):
        return np.array([0.04, 0.08, 0.12, 0.16, 0.2 , 0.16, 0.12, 0.08, 0.04, 0])

    def getSteps(self):
        return 5000


# ---------------
# -- Resultant --
# ---------------

class MediumChainInverted4060(Policy4060, RepInverted, MediumChain):
    pass

class MediumChainInverted2575(Policy2575, RepInverted, MediumChain):
    pass

class MediumChainTabular4060(Policy4060, RepTabular, MediumChain):
    pass

class MediumChainTabular2575(Policy2575, RepTabular, MediumChain):
    pass

class MediumChainDependent4060(Policy4060, RepDependent, MediumChain):
    pass

class MediumChainDependent2575(Policy2575, RepDependent, MediumChain):
    pass

class MediumChainTabular5050(Policy5050, RepTabular, MediumChain):
    pass

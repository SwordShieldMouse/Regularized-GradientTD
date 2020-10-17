from agents.GQ import GQ
from agents.QLearning import QLearning
from agents.QRC import QRC
from agents.QRC2 import QRC2
from src.agents.SARSA import SARSA
from src.agents.ESARSA import ESARSA
from src.agents.QC import QC
from src.agents.EQC import EQC
from src.agents.QC2 import QC2
from src.agents.ParameterFree import *

def getAgent(name):
    if name == "GQ":
        return GQ

    if name == 'SARSA':
        return SARSA

    if name == 'ESARSA':
        return ESARSA

    if name == 'QC':
        return QC

    if name == 'EQC':
        return EQC

    if name == 'QC2':
        return QC2

    if name == 'QRC':
        return QRC

    if name == 'QRC2':
        return QRC2

    if name == 'PFGQ':
        return PFGQ

    if name == 'PFGQ2':
        return PFGQ2

    if name == 'PFGQUntrunc':
        return PFGQUntrunc

    if name == 'PFGQScaledGrad':
        return PFGQScaledGrad

    if name == 'QLearning':
        return QLearning

    raise NotImplementedError()

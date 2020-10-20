from agents.GQ import GQ
from agents.QLearning import QLearning
from agents.QRC import QRC
from src.agents.ESARSA import ESARSA
from src.agents.QC import QC
from src.agents.EQC import EQC
from src.agents.ParameterFree import *

def getAgent(name):
    if name == "GQ":
        return GQ

    if name == 'ESARSA':
        return ESARSA

    if name == 'QC':
        return QC

    if name == 'EQC':
        return EQC

    if name == 'QRC':
        return QRC

    if name == 'PFGQ':
        return PFGQ

    if name == 'EPFGQ':
        return EPFGQ

    if name == 'PFGQ2':
        return PFGQ2

    if name == 'PFGQUntrunc':
        return PFGQUntrunc

    if name == 'PFGQScaledGrad':
        return PFGQScaledGrad

    if name == 'QLearning':
        return QLearning

    raise NotImplementedError()

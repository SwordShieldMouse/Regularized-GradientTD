from src.agents.GQ import GQ
from src.agents.QLearning import QLearning
from src.agents.QRC import QRC
from src.agents.ESARSA import ESARSA
from src.agents.QC import QC
from src.agents.EQC import EQC
from src.agents.ParameterFree import *
from src.agents.PFEnsembles import PFEnsemble, PFRobust, PFRobust2, BootstrapPFGQ, BootstrapPFGQ2

def getAgent(name):
    if name == "GQ":
        return GQ

    if name == "BootstrapPFGQ":
        return BootstrapPFGQ

    if name == "BootstrapPFGQ2":
        return BootstrapPFGQ2


    if name == "PFEnsemble":
        return PFEnsemble

    if name == "PFRobust":
        return PFRobust

    if name == "PFRobust2":
        return PFRobust2

    if name == "PFEnsembleTraces":
        return PFEnsemble

    if name == "PFEnsembleAverages":
        return PFEnsemble

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

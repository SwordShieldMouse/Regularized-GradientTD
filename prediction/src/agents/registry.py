from src.agents.TD import TD
from src.agents.TDC import TDC, BatchTDC
from src.agents.HTD import HTD
from src.agents.GTD2 import GTD2, BatchGTD2
from src.agents.TDRC import TDRC
from src.agents.Vtrace import Vtrace

from src.agents.ParameterFree import *
from src.agents.GTD2MP import GTD2MP

def getAgent(name):
    if name == 'TD':
        return TD
    elif name == 'TDC':
        return TDC
    elif name == 'BatchTDC':
        return BatchTDC
    elif name == 'HTD':
        return HTD
    elif name == 'GTD2':
        return GTD2
    elif name == 'BatchGTD2':
        return BatchGTD2
    elif name == 'Vtrace':
        return Vtrace
    elif name == 'TDRC':
        return TDRC
    elif name == 'GTD2MP':
        return GTD2MP

    elif name == 'PFGTD':
        return PFGTD
    elif name == "PFGTD+":
        return PFGTDPlus
    elif name == 'CWPFGTD':
        return CWPFGTD
    raise Exception(f'Unexpected agent {name} given')

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

    elif name == 'PFGTD':
        return PFGTD
    elif name == 'PFGTDMP':
        return PFGTDMP
    elif name == 'PFGTDHalfCW':
        return PFGTDHalfCW
    elif name == 'MultiDiscountPFGTD':
        return MultiDiscountPFGTD
    elif name == 'DiscountedPFGTD':
        return DiscountedPFGTD
    elif name == 'PFTDC':
        return PFTDC
    elif name == 'COCOBPFGTD':
        return COCOBPFGTD
    elif name == 'CWPFGTD':
        return CWPFGTD
    elif name == 'PFCombined':
        return PFCombined
    elif name == 'PFResidual':
        return PFResidual
    elif name == 'PFResidualV2':
        return PFResidual
    elif name == 'PFGTDUntrunc':
        return PFGTDUntrunc
    elif name == 'GTD2MP':
        return GTD2MP
    raise Exception(f'Unexpected agent {name} given')

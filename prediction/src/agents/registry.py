from TD import TD
from TDC import TDC, BatchTDC
from HTD import HTD
from GTD2 import GTD2, BatchGTD2
from TDRC import TDRC
from Vtrace import Vtrace

from ParameterFree import PFGTD, PFTDC, CWPFGTD, PFGTDUntrunc
from GTD2MP import GTD2MP

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

    elif name == 'PFGTD':
        return PFGTD
    elif name == 'PFTDC':
        return PFTDC
    elif name == 'CWPFGTD':
        return CWPFGTD
    elif name == 'PFGTDUntrunc':
        return PFGTDUntrunc
    elif name == 'GTD2MP':
        return GTD2MP
    raise Exception('Unexpected agent given')

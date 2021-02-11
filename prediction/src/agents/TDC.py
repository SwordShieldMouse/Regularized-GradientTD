import numpy as np
from src.agents.TDRC import TDRC, BatchTDRC
from src.utils.dict import merge

class TDC(TDRC):
    def __init__(self, features, actions, params):
        # TDC is just an instance of TDRC where beta = 0
        super().__init__(features, actions, merge(params, { 'beta': 0 }))

class BatchTDC(BatchTDRC):
    def __init__(self, features, actions, params):
        # TDC is just an instance of TDRC where beta = 0
        super().__init__(features, actions, merge(params, { 'beta': 0 }))

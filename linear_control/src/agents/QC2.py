from PyExpUtils.utils.dict import merge
from src.agents.QRC2 import QRC2

class QC2(QRC2):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, merge(params, { 'beta': 0 }))

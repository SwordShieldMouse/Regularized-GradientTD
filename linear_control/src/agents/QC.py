from PyExpUtils.utils.dict import merge
from src.agents.QRC import QRC

class QC(QRC):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, merge(params, { 'beta': 0 }))

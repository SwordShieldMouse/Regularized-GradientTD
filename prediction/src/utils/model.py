import sys
import json
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

class Experiment(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.agent = d['agent']
        self.problem = d['problem']
        self.subsample = d.get('subsample', 10)
        self.steps = d.get('steps', 10000)

def loadExperiment(path = None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = Experiment(d, path)
    return exp

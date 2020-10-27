import numpy as np
from RlGlue import BaseEnvironment
from src.utils.Critterbot import loadSensor


class Critterbot(BaseEnvironment):
    def __init__(self, sensorIdx):
        self.idx = 0
        self.sensorIdx = sensorIdx
        self.data = loadSensor(sensorIdx)

    def start(self):
        self.idx = 0
        return self.idx

    def step(self, a):
        self.idx+=1

        reward = self.data[self.idx]

        return (reward, self.idx, False)

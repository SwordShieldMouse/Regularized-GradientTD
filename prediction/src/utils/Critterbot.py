import numpy as np
from PyExpUtils.results.results import loadResults
from src.experiment import ExperimentModel
from PyExpUtils.utils.arrays import first

def loadSensor(sensorIdx):
    return np.load(f'./src/data/Critterbot/sensors/sensor{sensorIdx}.npy')

def loadTiles():
    return np.load('./src/data/Critterbot/tiles.npy')

def loadReturns(sensorIdx=None):
    path = "./src/data/Critterbot/returns.npy"
    return np.load(path) if sensorIdx is None else np.load(path)[:,sensorIdx]

def getNMSEs(exp_path):
    exp = ExperimentModel.load(exp_path)
    results = loadResults(exp, "nmse_summary.npy")

    r = first(results)
    data, std, _ = r.load()

    # predictions for one sensor in each row
    allData = {}
    allStd = {}
    allData[r.params["sensorIdx"]] = data
    allStd[r.params["sensorIdx"]] = std
    for r in results:
        data, std, _ = r.load()
        allData[r.params['sensorIdx']] = data
        allStd[r.params['sensorIdx']] = std
    return allData, allStd

def loadSensorNames():
    return np.load("./src/data/Critterbot/sensorNames.npy")

def getSensorNum(name):
    return list(loadSensorNames()).index(name)

def getSensorName(num):
    return loadSensorNames[num]

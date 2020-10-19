from src.problems.MountainCar import MountainCar, OfflineMountainCar

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    if name == 'OfflineMountainCar':
        return OfflineMountainCar

    raise NotImplementedError()

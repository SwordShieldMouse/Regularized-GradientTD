from src.problems.MountainCar import MountainCarTC, OfflineMountainCar
from src.problems.CartPole import CartPole

def getProblem(name):
    if name == 'MountainCar':
        return MountainCarTC

    if name == 'MountainCarRBF':
        return MountainCarRBF
    
    if name == 'OfflineMountainCar':
        return OfflineMountainCar

    if name == 'CartPole':
        return CartPole

    raise NotImplementedError()

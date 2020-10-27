from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.Boyan import Boyan
from src.problems.RandomWalk import RandomWalk
from src.problems.Critterbot import Critterbot

def getProblem(name):
    if name == 'RandomWalk':
        return RandomWalk

    if name == 'Baird':
        return BairdCounterexample

    if name == 'Boyan':
        return Boyan

    if name == 'Critterbot':
        return Critterbot

    raise NotImplementedError()

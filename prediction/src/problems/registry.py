from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.Boyan import Boyan
from src.problems.Chain import ChainInverted5050, ChainInverted4060, ChainInverted2575, ChainTabular5050, ChainTabular4060, ChainTabular2575, ChainDependent5050, ChainDependent4060, ChainDependent2575
from src.problems.MediumChain import MediumChainTabular5050
from src.problems.MediumChainTC import MediumChainTC5050
from src.problems.SmallChain import SmallChainTabular5050, SmallChainInverted5050, SmallChainDependent5050, SmallChainTabular4060, SmallChainInverted4060, SmallChainDependent4060, SmallChainRandomCluster5050, SmallChainRandomCluster4060, SmallChainRandomCluster1090, SmallChainOuterRandomCluster1090
from src.problems.SmallChainLeftZero import SmallChainTabular5050LeftZero, SmallChainInverted5050LeftZero, SmallChainDependent5050LeftZero
from src.problems.Collision import Collision
from src.problems.CollisionJournalPaper import CollisionJournalPaper

def getProblem(name):
    if name == 'Collision':
        return Collision

    if name == 'CollisionJournalPaper':
        return CollisionJournalPaper

    if name == 'Baird':
        return BairdCounterexample

    if name == 'Boyan':
        return Boyan

    if name == 'ChainInverted5050':
        return ChainInverted5050

    if name == 'ChainInverted4060':
        return ChainInverted4060

    if name == 'ChainInverted2575':
        return ChainInverted2575

    if name == 'ChainTabular5050':
        return ChainTabular5050

    if name == 'ChainTabular4060':
        return ChainTabular4060

    if name == 'ChainTabular2575':
        return ChainTabular2575

    if name == 'ChainDependent5050':
        return ChainDependent5050

    if name == 'ChainDependent4060':
        return ChainDependent4060

    if name == 'ChainDependent2575':
        return ChainDependent2575

    if name == 'MediumChainTabular5050':
        return MediumChainTabular5050

    if name == 'MediumChainTC5050':
        return MediumChainTC5050

    if name == 'SmallChainTabular5050':
        return SmallChainTabular5050

    if name == 'SmallChainDependent5050':
        return SmallChainDependent5050

    if name == 'SmallChainInverted5050':
        return SmallChainInverted5050

    if name == 'SmallChainTabular4060':
        return SmallChainTabular4060

    if name == 'SmallChainDependent4060':
        return SmallChainDependent4060

    if name == 'SmallChainInverted4060':
        return SmallChainInverted4060

    if name == 'SmallChainDependent5050LeftZero':
        return SmallChainDependent5050LeftZero

    if name == 'SmallChainInverted5050LeftZero':
        return SmallChainInverted5050LeftZero

    if name == 'SmallChainTabular5050LeftZero':
        return SmallChainTabular5050LeftZero

    if name == 'SmallChainRandomCluster5050':
        return SmallChainRandomCluster5050

    if name == 'SmallChainRandomCluster4060':
        return SmallChainRandomCluster4060

    if name == 'SmallChainRandomCluster1090':
        return SmallChainRandomCluster1090

    if name == 'SmallChainOuterRandomCluster1090':
        return SmallChainOuterRandomCluster1090

    raise NotImplementedError()



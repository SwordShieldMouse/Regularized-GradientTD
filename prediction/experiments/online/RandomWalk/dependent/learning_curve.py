import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from analysis.learning_curve import generatePlot

TITLE="RandomWalk (Dependent Features)"
FEATS="dependent"
SAVE_PATH=os.path.join(os.getcwd(),"figures/RandomWalk")

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_PATHS = list(map(lambda alg: os.path.join(FILE_DIR, f'{alg}.json'),['pfgtd','gtd2','tdc','tdrc', 'td','htd','vtrace']))

if __name__ == "__main__":
    f, axes = plt.subplots(1)

    bounds = []

    generatePlot(axes, EXP_PATHS, bounds)

    os.makedirs(SAVE_PATH, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    axes.set_title(TITLE)
    plt.savefig(f'{SAVE_PATH}/{FEATS}-learning-curve.png', dpi=100)

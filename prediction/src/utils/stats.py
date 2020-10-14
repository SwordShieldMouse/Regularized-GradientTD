import numpy as np

def exponentialSmoothing(a, beta=0.9):
    if len(a.shape) == 1:
        mean = 0
    else:
        mean = np.zeros(a.shape[1])

    r = np.zeros(a.shape)

    for i in range(a.shape[0]):
        mean = beta * mean + (1 - beta) * a[i]
        r[i] = mean


    return r

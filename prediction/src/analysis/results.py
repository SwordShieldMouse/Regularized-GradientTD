import numpy as np
from PyExpUtils.results.paths import listResultsPaths
from src.utils.arrays import first
from src.utils.dict import equal

class Result:
    def __init__(self, path, exp, idx):
        self.path = path
        self.exp = exp
        self.idx = idx
        self.params = exp.getPermutation(idx)['metaParameters']
        self._data = None
        self._reducer = lambda m: m

    def _lazyLoad(self):
        if self._data is not None:
            return self._data

        try:
            self._data = np.load(self.path, allow_pickle=True)
            return self._data
        except:
            print('Result not found :: ' + self.path)
            return (np.NaN, np.NaN, 0)

    def reducer(self, lm):
        self._reducer = lm
        return self

    def mean(self):
        return self._reducer(self._lazyLoad()[0])

    def stderr(self):
        return self._reducer(self._lazyLoad()[1])

    def runs(self):
        return self._reducer(self._lazyLoad()[2])

    def clone(self):
        return Result(self.path, self.exp, self.idx)

def splitOverParameter(results, param):
    parts = {}
    for r in results:
        param_value = r.params[param]

        if param_value not in parts:
            parts[param_value] = []

        parts[param_value].append(r)

    return parts

def getBestOverParameter(results, param, bestBy='end'):
    parts = splitOverParameter(results, param)

    best = {}
    for k in parts:
        if bestBy == 'auc':
            best[k] = getBest(parts[k])
        elif bestBy == 'end':
            best[k] = getBestEnd(parts[k])

    return best

def sliceOverParameter(results, slicer, param):
    parts = splitOverParameter(results, param)

    sl = {}
    for k in parts:
        sl[k] = find(parts[k], slicer, ignore=[param])

    return sl

def getBest(results, bestBy='auc'):
    if bestBy == 'end':
        return getBestEnd(results)

    low = first(results)
    for r in results:
        am = np.mean(r.mean())
        bm = np.mean(low.mean())
        if np.isnan(bm) or am < bm:
            low = r

    return low

def getBestEnd(results):
    low = first(results)
    for r in results:
        a = r.mean()
        b = low.mean()
        steps = int(len(a) * 0.1)
        am = np.mean(a[-steps:])
        bm = np.mean(b[-steps:])
        if np.isnan(bm) or am < bm:
            low = r

    return low

def find(stream, other, ignore=[]):
    params = other.params
    for res in stream:
        if equal(params, res.params, ignore):
            return res

def whereParameterEquals(results, param, value):
    return filter(lambda r: r.params.get(param, value) == value, results)

def whereParameterGreaterEq(results, param, value):
    return filter(lambda r: r.params.get(param, value) >= value, results)

def whereParameterLesserEq(results, param, value):
    return filter(lambda r: r.params.get(param, value) <= value, results)

def where(results, pred):
    return filter(pred, results)

def loadResults(exp, summary='errors_summary.npy'):
    for i, path in enumerate(listResultsPaths(exp)):
        summary_path = path + '/' + summary
        yield Result(summary_path, exp, i)

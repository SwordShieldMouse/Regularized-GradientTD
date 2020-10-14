def isIterable(thing):
    try:
        _ = (x for x in thing)
        return True
    except:
        return False

def flatMap(f, gen):
    for x in gen:
        r = f(x)
        if isIterable(r):
            for y in r:
                yield y

        else:
            yield r

def argmax(vals):
    ties = []
    top = vals[0]
    for i, v in enumerate(vals):
        if v > top:
            top = v
            ties = [i]
        elif v == top:
            ties.append(i)

    if len(ties) == 0:
        return list(range(len(vals)))

    return ties

def equal(d1, d2, ignore=[]):
    for k in d1:
        if k in ignore:
            continue

        if k not in d2:
            return False

        if d1[k] != d2[k]:
            return False

    return True

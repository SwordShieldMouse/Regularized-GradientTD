from src.utils.arrays import last

def up(path):
    return '/'.join(path.split('/')[:-1])

def fileName(path):
    parts = path.split('/')
    f = last(parts)
    return f

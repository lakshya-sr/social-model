import random

def clamp(x, minv, maxv):
    return max(minv, min(maxv, x))


def interval(start, stop, n):
    return [start+((stop-start)*i/n) for i in range(n)]

def random_array(n):
    return [random.random() for i in range(n)]

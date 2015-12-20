from itertools import chain, product
import numpy as np
import random

def makeWalls(rows, cols, density):
    if density > .8:
        print "WARN: Wall map with excessive density of %f" % density
    grid = list(product(np.arange(rows), np.arange(cols)))
    random.shuffle(grid)
    walls = {}
    for idx, el in enumerate(grid):
        walls[el] = True
        if float(idx) / (rows*cols) >= density:
            break
    return walls

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def manhattan(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return abs(y1-y2) + abs(x1-x2)

def connected(visited, first = True):
    xsorted = sorted(visited, key = lambda x: x[int(not(first))])
    out = {}
    seen = None
    maxSize = 0
    for i, j in xsorted:
        if seen != i:
            out[i] = [(i,j)]
            seen = i
        else:
            out[i] = out[i] + [(i,j)]
        size = len(out[i])
        if size > maxSize:
            maxSize = size
    return maxSize
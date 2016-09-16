from itertools import chain, product
from collections import OrderedDict as OD
from collections import deque
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

def get_path(prevs, goal, start):
    """Gets the path from start to goal using prev"""
    path = OD({goal: 0})
    cur = goal
    while cur != start:
        (cost, node) = prevs.get(cur)
        if node == None or node in path:
            print("ERROR: No path found from %s -> %s" % (start, goal))
            return (0, None)
        path[node] = path[cur] + cost
        cur = node
    return (path[start], path.keys()[::-1])

def bisect_left(a, x, lo=0, hi=None, key=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    if key is None:
        key = lambda x:x
    xval = key(x)
    while lo < hi:
        mid = (lo+hi)//2
        if key(a[mid]) < xval: lo = mid+1
        else: hi = mid
    return lo

def queue_insert(q, value, key=lambda x: x):
    new_idx = bisect_left(q, value, key=key)
    length = len(q)
    if new_idx < length // 2:
        q.rotate(-new_idx)
        q.appendleft(value)
        q.rotate(new_idx)
    else:
        q.rotate(length-new_idx)
        q.append(value)
        q.rotate(-(length-new_idx))

def queue_remove(q, value, key=lambda x: x):
    idx = bisect_left(q, value, key=key)
    length = len(q)
    if idx < length // 2:
        q.rotate(-idx)
        q.popleft()
        q.rotate(idx)
    else:
        q.rotate((length-1)-idx)
        q.pop()
        q.rotate(-(length-1-idx))
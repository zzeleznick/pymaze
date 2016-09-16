from itertools import combinations
from termcolor import colored # coloring yay
import time
import re
from copy import deepcopy
from collections import defaultdict, deque
from heapq import heappush, heappop
# internals #
from utils import *

class Maze(object):
    def __init__(self, rows, cols, start = (0,0), end = None, verbose = True):

        assert rows > 0 and cols > 0, "Must have non-zero rows and cols"
        self.rows = rows
        self.cols = cols
        self.walls = {}
        self.validSqFn =  lambda x: x[0] >= 0 and x[0] < self.rows  \
                                 and x[1] >= 0 and x[1] < self.cols
        if self.validSqFn(start):
            self.start = start
        else:
            print "WARN: Start of grid. Placing at (0,0)"
            self.start = (0,0)
        self.verbose = verbose
        if not end or not self.validSqFn(end):
            self.end = (rows-1, cols-1)
        else:
            self.end = end

        def genRep(rows, cols):
            row = ['| '] * cols + ['|\n']
            out = []
            [ out.append(row[:]) for i in xrange(rows) ]
            i,j = self.start
            p,q = self.end
            out[i][j] = out[i][j].replace(" ", 'S')
            out[p][q] = out[p][q].replace(" ", 'E')
            return out

        self.genRep = genRep   # establish for children
        self.rep = self.genRep(rows, cols)  # establish for children

    def __repr__(self):
        head = '<Maze r:%d | c:%d>' % (self.rows, self.cols)
        return head

    def __str__(self):
        return ''.join(flatten(self.rep))

    def showPath(self, explored, path = None, flat = True):
        if path and type(path[0]) == int:
            path = path[1]
        out = deepcopy(self.rep)
        if explored:
            for a,b in explored:
                if path and (a,b) in path:
                    continue
                else:
                    bar, val = out[a][b]
                    out[a][b] = colored(bar + '-', "blue")
        digit = re.compile(r'\d')
        lower = re.compile(r'[a-z]')
        if path:
            for i,j in path:
                bar, val = out[i][j].split('|')
                val = val[0]
                if digit.match(val):
                    if int(val) == 9:
                        out[i][j] = "|" + 'a'
                    else:
                        out[i][j] =  colored( "|" + str(int(val)+1), "green")
                elif lower.match(val):
                    val = chr(ord(val) + 1)
                    out[i][j] = "|" + val
                else:
                    out[i][j] = colored("|" + '1', "green")
        if flat:
            return ''.join(flatten(out))
        else:
            return out

    def getNeighbors(self, pos):
        r,c = pos
        potential = [ (r-1,c), (r,c+1),  # top, right,
                      (r+1,c), (r,c-1) ] # bottom, left
        return filter(self.validSqFn, potential)

    def costFunction(self, method, fnc):
        if method.upper() == 'ASTAR':
            return lambda x, y: fnc(x, self.end)
        else: # BFS, DFS
            return lambda x, y: fnc(self.start, self.start)

    def DFSsolve(self):
        return self.solve(method = 'DFS')

    def BFSsolve(self):
        return self.solve(method = 'BFS')

    def ASTARsolve(self):
        return self.solve(method = 'ASTAR')

    def goalCompleted(self, path1, path2):
        res = set(path1).intersection(set(path2))
        return res

    def twowaysolve(self, method = 'BFS', costfn = manhattan):
        costfn = self.costFunction(method, costfn)
        notDFS = method != 'DFS' # reverse flag
        st1 = self.start
        st2 = self.end
        goals = [st2, st1]
        if len(self.getNeighbors(st1)) == 0 or len(self.getNeighbors(st2)) == 0:
            # print "No solution. Exit Blocked"
            return None
        explored = {}
        fringe = [ [2, [[st1], [st2]] ] ]
        counter = 0
        path1 = []
        while fringe:
            # path1, path2 = fringe[-1][1]
            if self.goalCompleted(*fringe[-1][1]):
                # print "Solved"
                if self.verbose: print "Remaining fringe: %s" % fringe
                if self.verbose: print "Solved: %s" % node
                path1, path2 = fringe[-1][1]
                break
            elif counter > 800000:
                print "error"
                return None
            else:
                turn = counter % 2 # 0 or 1
                fringe = sorted(fringe, key = lambda x: len(x[1][turn]), reverse = True)
                num = len(fringe[-1][1][turn])
                iterations = 1
                for x in range(len(fringe) - 2, -1, -1):
                     if len(fringe[x][1][turn]) == num:
                        iterations += 1
                newNodes = []
                for idx in xrange(iterations):
                    node = fringe.pop()
                    value = node[0]
                    paths = node[1]
                    myvisits = paths[turn]
                    sibvisits = paths[turn == 0]
                    position = myvisits[-1]
                    sibpos = sibvisits[-1]
                    if turn == 0:
                        nextState = (position, sibpos)
                    else:
                        nextState = (sibpos, position)
                    counter += 1
                    if nextState not in explored:
                        explored[nextState] = True
                        neighbors = self.getNeighbors(position)
                        # print neighbors
                        spentcost = len(myvisits) + len(sibvisits) + 1
                        sibHeur = costfn(sibpos, goals[turn == 0])
                        visits = [ [], [] ]
                        visits[turn == 0] = sibvisits
                        for idx, n in enumerate(neighbors):
                            cost = spentcost + min(sibHeur, costfn(n, goals[turn]))
                            visits[turn] = myvisits + [n]
                            newNodes += [ [cost, deepcopy(visits)] ]
                        if self.verbose: print newNodes
                        # print 'New Nodes:', newNodes
                [fringe.append(el) for el in newNodes]

        def proccessExplored(explored):
            allKeys = explored.keys()
            exp = {}
            for k1, k2 in allKeys:
                exp[k1] = True
                exp[k2] = True
            return exp

        if not path1 or not path2:
            if not fringe: print "fringe is empty with %s" % fringe
            print "this is a hard puzzle..."
            return proccessExplored(explored)
        else:
            print "Exiting"
            return proccessExplored(explored), list(set(path1).union(set(path2)))

    def solve(self, method = 'BFS', costfn = manhattan):
        costfn = self.costFunction(method, costfn)
        notDFS = method != 'DFS' # reverse flag
        pos = self.start
        goal = self.end
        if len(self.getNeighbors(pos)) == 0 or len(self.getNeighbors(goal)) == 0:
            # print "No solution."
            return None
        explored = {}
        fringe = [ [0, [pos]]   ]  # e.g. [ [1, [(0, 0)] ], ... ]
        path = []
        counter = 0
        while fringe:
            # print "counter: %d, f: %s" % (counter, fringe)
            if pos == goal:
                if self.verbose: print "Remaining fringe: %s" % fringe
                # print "solved."
                if self.verbose: print "Solved: %s" % node
                path = node
                break
            elif counter > 800000:
                print "error"
                return {}, None
            else:
                node = fringe.pop()  # lowest cost node
                counter += 1
                if self.verbose: print "Popped Node: %s at idx: %d" % (node, idx)
                pos = node[1][-1]
                val = node[0]
                nextState = pos
                if self.verbose: print "Next state: (%d, %d)" % (pos[0], pos[1])
                if nextState not in explored:
                    explored[nextState] = val
                    visited = node[1]
                    neighbors = self.getNeighbors(pos)
                    # print "neighbors for idx: %d are %s" % (counter, neighbors)
                    # cost := distance + heuristic
                    newNodes = [ [ (len(visited)-1) + 1 + costfn(n, goal),
                                  node[1] + [n] ] for n in neighbors ]
                    if self.verbose: print newNodes
                    fringe += newNodes
                    fringe = sorted(fringe, key = lambda x: x[0], reverse = notDFS)
                    # since pop takes from the end, we sort to have the smallest at end

        print "Visited %s nodes" % len(explored)
        if not path:
            # print "No solution."
            return explored, None
        return explored, path

    def solve2(self, method = 'BFS', costfn = manhattan):
        costfn = self.costFunction(method, costfn)
        notDFS = method != 'DFS' # reverse flag
        start, goal = self.start, self.end
        if len(self.getNeighbors(start)) == 0 or len(self.getNeighbors(goal)) == 0:
            return None
        visited = {}
        costs = defaultdict(lambda: float("inf"))
        prevs = defaultdict(lambda: (0, None))
        # fringe = []
        # heappush(fringe, (0, start))
        fringe = deque([(0, start)])
        while fringe:
            # cost, node = heappop(fringe) # pop the min element
            cost, node = fringe.popleft()
            if node == goal:
                print("Visited %s nodes" % len(visited))
                return get_path(prevs, goal, start)
            if node in visited:
                continue
            visited[node] = True
            neighbors = self.getNeighbors(node)
            for v in neighbors:
                # cost := distance + heuristic
                next_cost = (cost + 1) + costfn(v, goal)
                if next_cost < costs[v]:
                    costs[v] = next_cost
                    prevs[v] = (1, node)
                    # heappush(fringe, (next_cost, v))
                    # fringe.append((next_cost, v))
                    # fringe = sorted(fringe, key = lambda x: x[0], reverse = notDFS)
                    queue_insert(fringe, (next_cost, v), key=lambda x: x[0])
        print "No solution."
        return (costs, prevs)

class WalledMaze(Maze):
    def __init__(self, rows, cols, walls = [], verbose = True):

        Maze.__init__(self, rows, cols, verbose = verbose)

        def genWalledRep(rows, cols, walls):
            out = self.genRep(rows, cols) # call from parent
            match = re.compile(r"[SE\. ]")
            assert type(walls) == list, "walls malformed"
            for i,j in walls:
                # x = colored("X", "red")
                x = "~"
                out[i][j] = match.sub(x, out[i][j])
            return out

        self.walls = {}
        [self.walls.update({w: True}) for w in walls if w != self.start and w != self.end]
        self.rep = genWalledRep(rows, cols, self.walls.keys())

        self.validSqFn =  lambda x: x[0] >= 0 and x[0] < self.rows  \
                                 and x[1] >= 0 and x[1] < self.cols \
                                 and x not in self.walls.keys()
    def __repr__(self):
        head = '<WalledMaze r:%d | c:%d | w: %d>' % (self.rows, self.cols, len(self.walls))
        return head

    def showPath(self, explored, path):
        return Maze.showPath(self, explored = explored, path = path)

    @classmethod
    def generate(cls, rows, cols, wallDensity = .5):
        walls = makeWalls(rows, cols, wallDensity)
        return cls(rows, cols, walls = walls, verbose = False)

class SpecialMaze(Maze):
    def __init__(self, rows, cols, goals = [(0,0)], allowDiag = False, verbose = True):
        if len(goals) == 1:
            goals += [(rows-1, cols-1)]
        self.goals = goals
        Maze.__init__(self, rows, cols, start = goals[0], end = goals[-1], verbose = verbose)

        def genSpecialRep(rows, cols, goals):
            out = self.genRep(rows, cols) # call from parent
            assert type(goals) == list, "goals malformed"
            match = re.compile(r"[SE\. ]")
            for i,j in goals:
                out[i][j] = match.sub('*', out[i][j])
            return out

        self.rep = genSpecialRep(rows, cols, goals)
        self.getUnvisited = lambda history: tuple([goal in history for goal in self.goals])
        self.unvisitedCount = lambda hst: len([goal for goal in self.goals if goal not in hst])
        self.allowDiag = allowDiag
        dists = [ ( abs(s1[0] - s2[0]), abs(s1[1] - s2[1]) ) \
                 for s1, s2 in list(combinations(self.goals, 2)) ]
        self.xd = sorted(dists, key = lambda x: x[0] )[-1][0]
        self.yd = sorted(dists, key = lambda x: x[1] )[-1][1]

    def __repr__(self):
        head = '<SpecialMaze r:%d | c:%d | g: %d>' % (self.rows, self.cols, len(self.goals))
        return head

    def getNeighbors(self, pos):
        r,c = pos
        potential = [ (r-1,c), (r,c+1),  # top, right,
                      (r+1,c), (r,c-1) ] # bottom, left
        if self.allowDiag:
            potential += [ (r-1,c-1), (r-1,c+1),  # top-left, top-right
                          (r+1,c-1), (r+1,c+1) ] # bot-left, bot-right
        return filter(self.validSqFn, potential)

    def showPath(self, explored, path):
        return Maze.showPath(self, explored = explored, path = path)

    @classmethod
    def generate(cls, rows, cols, goalDensity = .05, allowDiag = False):
        goals = makeWalls(rows, cols, goalDensity)
        return cls(rows, cols, goals = goals.keys(), allowDiag = allowDiag, verbose = False)

    def goalCompleted(self, visited):
        flags = [(goal in visited) for goal in self.goals]
        return not(False in flags)

    def farthestDist(self, visited):
        unvisited = self.getUnvisited(visited)
        if len(unvisited) >= 2:
            dists = [ manhattan(s1,s2) for s1, s2 in list(combinations(unvisited, 2)) ]
            return max(dists)
        else:
            return 0

    def travelerHeuristic(visited):
        # steps = len(visited)  # keys in dict
        # return steps
        pass

    def t2(self, visited):
        steps = len(visited)  # keys in dict
        return steps + len(self.getUnvisited(visited))

    def t3(self, visited):
        xd = self.xd
        yd = self.yd
        xv = connected(visited, first = True)
        yv = connected(visited, first = False)
        return len(visited) + max(xd-xv, 0) + max(yd-yv, 0) # + len(self.getUnvisited(visited))

    def specialSolve(self, method ='BFS', costfn = None):
        if not costfn:
            costfn = self.unvisitedCount
        if len(self.goals) == 2:
            visited, path = self.solve()
            return visited, path
        else:
            notDFS = method != 'DFS' # reverse flag
            position = self.start
            goals = self.goals
            visited = {position: True}
            left = self.unvisitedCount(visited)
            flagsPulled = self.getUnvisited(visited)
            explored = {}  # flags [False, False...] ,position (0,0)
            fringe = [ [1, [position] ] ]
            path = []
            idx = 0
            keyGen = lambda x: (x, True)
            while fringe:
                if self.goalCompleted(visited):
                    if self.verbose: print "Remaining fringe: %s" % fringe
                    if self.verbose: print "Solved: %s" % node
                    path = node
                    break
                elif idx > 500000:
                    print "error"
                    break
                else:
                    node = fringe.pop()  # lowest cost node
                    if self.verbose: print "Popped Node: %s at idx: %d" % (node, idx)
                    pos = node[1][-1] # next position to explore
                    visited = node[1]
                    unique = dict(map(keyGen, visited))
                    unique[pos] = True
                    flagsPulled = self.getUnvisited(unique.keys())
                    nextState = (flagsPulled, pos)
                    if self.verbose: print "Next state: %s" % list(nextState)
                    # print nextState
                    if nextState not in explored:
                        idx += 1
                        explored[nextState] = True
                        neighbors = self.getNeighbors(pos)
                        trueCost = len(unique)
                        newH = costfn(unique.keys())  # unexplored goals
                        newNodes = [ [trueCost + newH,
                                      node[1] + [n] ] for n in neighbors ]
                        if self.verbose: print newNodes
                        fringe += newNodes
                        fringe = sorted(fringe, key = lambda x: x[0], reverse = notDFS)
                        # since pop takes from the end, we sort to have the smallest at end
        if not path:
            # print "No solution."
            return None
        else:
            return path

def test1():
    maze = Maze(4,4)
    print repr(maze)
    print maze
    mazeSp2 = SpecialMaze(7,7, verbose = False, goals = [(0,0), (6,0), (3,4), (4,4), (6,6)])
    print repr(mazeSp2)
    print mazeSp2
    mazeSp = WalledMaze(7,7, walls = [(6,0), (3,4), (4,4), (6,6)], verbose = False)
    print repr(mazeSp)
    print mazeSp

def test2():
    mazeSp = WalledMaze.generate(80, 40, .3)
    print repr(mazeSp)
    print mazeSp
    start = time.time()
    explored, path = mazeSp.solve()
    print "Time Elapsed %0.3f ms" % (1000 * (time.time() - start))
    print "Solved with path (%s)\n%s" % (path[0], path[1])
    print mazeSp.showPath(explored = None, path = path)
    start = time.time()
    cost, path = mazeSp.solve2()
    print "Time Elapsed %0.3f ms" % (1000 * (time.time() - start))
    print "Solved with path (%s)\n%s" % (cost, path)
    print mazeSp.showPath(explored = None, path = path)

if __name__ == '__main__':
    # test1()
    test2()
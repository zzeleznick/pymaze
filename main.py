from maze import Maze, SpecialMaze, WalledMaze
import sys
import re
import time

from colorama import init as initColor # use init from Colorama to make Termcolor work on Windows too
from termcolor import colored

# internals #
from utils import makeWalls

def generateWalledMaze(rows, cols, wallDensity = .5):
    walls = makeWalls(rows, cols, wallDensity)
    maze = WalledMaze(rows, cols, walls = walls, verbose = False)
    return maze

def generateGoalMaze(rows, cols, goalDensity = .05, allowDiag = False):
    goals = makeWalls(rows, cols, goalDensity)
    maze = SpecialMaze(rows, cols, goals = goals.keys(), allowDiag = allowDiag, verbose = False)
    return maze

def test_Steiner_Maze():
    # maze = generateGoalMaze(10, 10, .05)
    # maze = generateGoalMaze(20, 20, .0125)
    maze = generateGoalMaze(25, 25, .006, allowDiag = True)
    path = maze.specialSolve()
    if path:
        # print "Goals at ", maze.goals
        print repr(maze)
        # print "Length: %d | Path: %s" %  ( len(set(path[1])), path )
        print maze
        print maze.showPath(explored = None, path = path)

def test_Steiner_Maze_Fixed(rows, cols, goals):
    mazeSp2 = SpecialMaze(rows, cols, verbose = False, goals = goals)
    print repr(mazeSp2)
    print mazeSp2
    start = time.time()
    path = mazeSp2.specialSolve()
    end1 = time.time()
    # path2 = mazeSp2.specialSolve(costfn = mazeSp2.t2)
    # path2 = mazeSp2.specialSolve(costfn = mazeSp2.t3)
    path2 = None
    end2 = time.time()
    print "Naive Search: %0.7f" % (end1 - start)
    if path2: print "Better Search: %0.7f" % (end2 - end1)
    if path:
        print "Naive Length: %d | Path: %s" %  ( len(set(path[1])), path )
    if path2:
        print "Better Length: %d | Path: %s" % ( len(set(path2[1])), path2)
    if path:
        print mazeSp2.showPath(explored = None, path = path)
    if path2:
        print mazeSp2.showPath(explored =  None,  path = path2)

def test_simple_special_maze():
    mazeSp = SpecialMaze(7,7, verbose = False)
    maze = Maze(7,7, verbose = False)
    print repr(mazeSp)
    print mazeSp
    print repr(maze)
    print maze
    explored, path = mazeSp.specialSolve()
    explored2, path2 = maze.solve()
    print mazeSp.showPath(explored = explored.keys(), path = path)
    print maze.showPath(explored = explored2.keys(), path = path2)

def test_simple_walled_maze():
    maze = WalledMaze(7,7, walls = [(1,0), (1,1)], verbose = False)
    print repr(maze)
    print maze
    explored, path = maze.solve()
    print maze.showPath(path)

def run_all_searches(bfs = True):
    maze = generateWalledMaze(40, 40, .3)
    # print maze.walls.keys()
    print repr(maze)
    print maze
    out = maze.BFSsolve()
    out2 = maze.DFSsolve()
    out3 = maze.ASTARsolve()
    # out = maze.twowaysolve()
    if not out:
        print "Unsolvable Maze"
    elif type(out) == dict:
        print maze.showPath(explored = out.keys(), path = None)
    else:
        print maze.showPath(explored = out[0].keys(), path = out[1])
        print maze.showPath(explored = out2[0].keys(), path = out2[1])
        print maze.showPath(explored = out3[0].keys(), path = out3[1])


def main(sizes, verbose = False):
    if type(sizes) == tuple: sizes = [sizes]
    for size in sizes:
        r,c = size
        maze = Maze(r, c, verbose = verbose)
        print repr(maze)
        print maze
        start = time.time()
        e1, p1 = maze.BFSsolve()
        end1 = time.time()
        e2, p2 = maze.DFSsolve()
        end2 = time.time()
        e3, p3 = maze.ASTARsolve()
        end3 = time.time()
        print "BFS Search: %0.7f" % (end1 - start)
        print "Number Explored: %d | Path: %s" % (p1[0], p1[1])
        print "DFS Search: %0.7f" % (end2 - end1)
        print "Number Explored: %d | Path: %s" % (p2[0], p2[1])
        print "A*  Search: %0.7f" % (end3 - end2)
        print "Number Explored: %d | Path: %s" % (p3[0], p3[1])
        print maze.showPath(p1)

def parser():
    args = sys.argv[1:]
    print args
    nondigit = re.compile(r'\D')
    scrub = lambda x: nondigit.sub('', x)
    vals = [int(scrub(arg)) for arg in args if scrub(arg)]
    sizes = []
    for x in range(1, len(vals), 2):
        sizes += [ (vals[x-1], vals[x]) ]
    print sizes
    main(sizes)


if __name__ == '__main__':
    # print colored('hello', 'red'), colored('world', 'green')
    args = sys.argv[1:]
    if len(args) == 0:
        run_all_searches()
    elif len(args) == 1:
        goals = [(0,0), (6,0), (3,4), (4,4), (6,6)]
        # goals = [(0,0), (0,4), (4,2), (4,4), (0,9), (9,9)]
        # goals =  [(0,0), (0,4), (4,2), (4,4), (0,9), (6,6), (2,4), (4,9), (3,9), (8,1)]
        r,c  = 10, 10
        test_Steiner_Maze_Fixed(r,c, goals)
    else:
        test_Steiner_Maze()

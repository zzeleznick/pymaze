#!/usr/bin/env python
# coding=UTF-8
import sys
from blessings import Terminal
import time
import numpy as np
import argparse
from random import gauss

def process_input():
    out = None
    args = sys.argv[1:]
    if args:
        out = args[0]
    return out

class EggTimer(object):
    """
    Dings when the timer has expired
    """
    def __init__(self, term, dingFnc = lambda: 'Done', expiry = 0.3):
        self.home = term
        self.ding = dingFnc
        self.last = time.time()
        self.expiry = expiry

    def ding_dong(self):
        '''
        i.e. print on release
        Prints each time now - last > threshold.
        '''
        now = time.time()
        last, exp = self.last, self.expiry
        term, ding = self.home, self.ding
        if now - last > exp:
            # print term.move(h, 0) + str(graphic)
            print term.move(1, 0) + str(ding())
            self.last = time.time()

class ProgressBar(object):
    """
    Creates a configurable progress bar that displays
    the percentage and fraction of completion for a task.
    e.g. >>> Progress: (100%)  [========================>]    100000 | 100000 bits
    """
    def __init__(self, width, limit, title = 'Progress', sep = '|', units = 'bits'):
        self.width = width
        # max width of the terminal display
        self.limit = limit
        # max value of the progress bar
        self.title = title
        # text title of the progress bar to display
        self.sep = sep
        # character to separate the bar from the values
        self.units = units
        # units of the value
        self.current = 0
        # current value
        self.determine_padding()

    def determine_padding(self):
        self.padding = ' ' * max(3, int(self.width * 0.025))
        # internal padding between each element
        self.rhs = 2 * len(str(self.limit)) + len(self.sep) + len(self.units)
        # space for the ratio of current / total units
        # >>> <lhs> <bar>  100000 | 100000 bits
        self.lhs = len(self.title) + 5
        # space for the title and percentage completion
        # >>> Progress: (100%) <bar>
        self.mid = max(0, int(self.width * .9) - self.rhs - self.lhs - len(self.padding))
        # max size of the bar

    def makeLine(self, idx):
            title, limit, sep, units = self.title, self.limit, self.sep, self.units
            padding, rhs, maxbarsize = self.padding, self.rhs, self.mid
            self.current = idx
            n = ('%0.0f' % (float(idx)/limit * 100)).zfill(2)
            pct = '(%s%s)' % (n, '%')
            right = ('%d %s %d %s' % (idx, sep, limit, units)).rjust(rhs)
            bar = '=' * int(float(idx) / limit * maxbarsize)
            unfilled = ' ' * (maxbarsize - len(bar))
            mid = '%s[%s>%s]%s' % (' ', bar, unfilled, padding)
            return ' '.join([title, pct, mid, right])

    def __repr__(self):
        return self.makeLine(self.current)

def proccess_values(i, values):
    delay = 2.0 / (len(values))
    delta = gauss(0, delay)
    delay = max(0, delay + delta )
    time.sleep(delay)
    if i < len(values):
        i += values[i]
    else:
        i += np.random.randint(10)
    return i

def main():
    sys.stderr.write("\x1b[2J\x1b[H")
    # clear the terminal
    term = Terminal() # activate the new terminal
    task = process_input()
    if not task: task = 'Important stuff'
    print term.bold(task)
    h = 1
    w = term.width
    limit = 10**5
    vals = np.random.random(limit) * 10

    title = "Progress:"
    sep = '|'
    units = 'bits'

    graphic = ProgressBar(w, limit, title, sep, units)

    i = 0
    print graphic.makeLine(i)
    ding = lambda: str(graphic)
    timer = EggTimer(term, ding)

    while i < limit:
        i = proccess_values(i, vals)
        graphic.current = i
        timer.ding_dong()


    print term.move(h, 0) + graphic.makeLine(limit)
    print term.bold('Success!')

if __name__ == '__main__':
    main()
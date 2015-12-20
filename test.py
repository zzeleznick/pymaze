import sys
from blessings import Terminal
import time

# import subprocess;
# subprocess.call(["printf", "\033c"]);
sys.stderr.write("\x1b[2J\x1b[H")

term = Terminal()

print term.bold('Important stuff')

if term.does_styling:
    # h = term.height
    w = min(40, int(term.width * .6))
    h = 1
    for i in xrange(w//2):
        # with term.location(0, h):
        bar = "=" * i
        space = ' ' * (w - i)
        time.sleep(.05)
        print term.move(h, 0) + 'Progress: [%s>%s]' % (bar, space)

    for i in xrange(w//2, -1, -1):
        with term.location(0, h):
            if i == 10:
                print term.move(h, 0)
                print term.bold('Hahaha')
            bar = "=" * i
            space = ' ' * (w - i)
            time.sleep(.05)
            print term.move(h, 0) + 'Progress: [%s>%s]' % (bar, space)

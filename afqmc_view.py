import numpy as np

import subprocess


def pwrite(g, data):
    g.stdin.write(data.encode('utf-8'))

def make_plot(x, y):
    pass

def equil_curve():
    #1. get data
    pass

def test():
    x = np.linspace(0,2*np.pi,100)
    y = np.sin(x)

    gplot = subprocess.Popen(["/usr/bin/gnuplot"],
                             stdin=subprocess.PIPE)

    # function alias
    gflush = gplot.stdin.flush

    pwrite(gplot,"set term dumb 60 20\n")
    pwrite(gplot,"plot '-' using 1:2 title 'E(Beta)' with linespoints \n")
    for i,j in zip(x,y):
        pwrite(gplot,f"{i} {j}\n")
    pwrite(gplot,"e\n")
    gflush()

if __name__ == '__main__':
    test()

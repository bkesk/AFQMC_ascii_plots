#!/usr/bin/env python3
import numpy as np
import argparse
import logging

import subprocess

from regressions import reblock

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def pwrite(g, data):
    g.stdin.write(data.encode('utf-8'))

def make_plot(x, y, dy):
    '''
    make an ascii plot with x,y and with errorbars y

    inputs:
    x - plot domain data
    y - plot data
    dy - errorbars for plot data

    '''
    gplot = subprocess.Popen(["/usr/bin/gnuplot"],
                             stdin=subprocess.PIPE)

    # function alias
    gflush = gplot.stdin.flush

    
    pwrite(gplot,"set term dumb 80 20\n")
    pwrite(gplot,"plot '-' using 1:2 title 'E(Beta)' with linespoints \n")
    for a,b in zip(x,y):
        pwrite(gplot,f"{a} {b}\n")
    pwrite(gplot,"e\n")
    gflush()
    
    # plot errorbars separately
    #if dy is not None:
    #    pwrite(gplot,"set term dumb 80 20\n")
    #    pwrite(gplot,"plot '-' using 1:3 title 'dE (Beta)' with yerrorlines \n")
    #    for a,b,c in zip(x,y,dy):
    #        pwrite(gplot,f"{a} {b} {c}\n")
    #    pwrite(gplot,"e\n")
    #    gflush()

def equil_curve(fname, block=None, weights=None):

    #1. get data
    beta,Ereal,Eimag = np.loadtxt(fname,unpack=True)
    
    E = np.abs(Ereal + 1j*Eimag)
    #E = Ereal
    dE = np.zeros(E.shape) # TODO: this is temporary, need to get dE for each block

    #2. (opt) reblock data
    if block is not None:
        print(f"  [{bcolors.OKGREEN}+{bcolors.ENDC}] reblocking with block size {bcolors.OKGREEN}{block}{bcolors.ENDC} ")        
        if weights is None:
            weights,_ = np.loadtxt("den.dat",unpack=True)
        beta,E,dE,W=reblock(beta,E,dE,weights,block)

    #3. plot data
    make_plot(beta,E,dE)

def get_args():
    '''
    gets and parses terminal arguments

    returns:
    args - an object containing the parsed arguments - see argparse documentation
    '''
    
    parser = argparse.ArgumentParser(description='Plot AFQMC E(beta) curve')
    parser.add_argument('--block_size','-b', metavar='block', type=int,
                        action='store',
                        help='block size for reblocking')
    parser.add_argument('--name','-n', metavar='name', type=str,
                        action='store',
                        default='energyVersusBeta.dat',
                        help='name of file containing AFQMC data')
    parser.add_argument('--verbose', '-v', action='count',
                        help='set verbosity level, use more \'v\' chars to make output more verbose (i.e. -vv ) max. level is 3')


    args = parser.parse_args()
    return args

def main():
    # get options
    args = get_args()

    block = args.block_size#[0]
    fname = args.name
    verbose = args.verbose


    #TODO: use logger for these
    #print("Options are: ")
    #print(" block_size ", block)
    #print(" name ", fname)
    #print(" verbose ", verbose)

    ##############
    #
    #  debug: example
    #
    ##############
    #print("Contents of args's 'dir'")
    #for r in dir(args):
    #    print(f"{r}")
    
    if verbose:
        print(f"  [{bcolors.OKGREEN}+{bcolors.ENDC}] using verbosity level {bcolors.OKGREEN}{verbose}{bcolors.ENDC} ")

    # run analysis
    equil_curve(fname,block)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import numpy as np
import argparse

import subprocess # careful! no user input

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
    pwrite(gplot,"plot '-' using 1:2 title 'E(beta)' with linespoints \n")
    for a,b in zip(x,y):
        pwrite(gplot,f"{a} {b}\n")
    pwrite(gplot,"e\n")
    gflush()
    

def avg_block(beta, E, err, W):
    assert W.shape == E.shape
    assert beta.shape == E.shape
    assert err.shape == E.shape
    
    Wblk = np.sum(W)
    Eblk = np.dot(E,W)/Wblk

    # the error is simply the rms error, acounting for the walker weights
    errblk = np.sqrt(np.average(err**2, weights=W))
    betablk = np.mean(beta)
    
    return betablk, Eblk, errblk, Wblk
    
def reblock(beta, E, err, W, blk_size):
    assert W.shape == E.shape
    assert beta.shape == E.shape
    assert err.shape == E.shape
        
    N = E.shape[0] // blk_size

    Beta_rb = np.zeros((N))
    Erb = np.zeros((N))
    err_rb = np.zeros((N))
    Wrb = np.zeros((N))
    
    for n in range(N):
        Beta_rb[n], Erb[n], err_rb[n], Wrb[n] = avg_block(beta[n*blk_size:(n+1)*blk_size],
                                     E[n*blk_size:(n+1)*blk_size],
                                     err[n*blk_size:(n+1)*blk_size],
                                     W[n*blk_size:(n+1)*blk_size])

    return Beta_rb, Erb, err_rb, Wrb

def equil_curve(fname, block=None, ignore=0, weights=None, imag=False):

    #1. get data
    beta,Ereal,Eimag = np.loadtxt(fname,unpack=True)
    
    if ignore > 0:
        beta = beta[ignore:]
        Ereal = Ereal[ignore:]
        Eimag = Eimag[ignore:]

    if imag:
        print(f"\n\n{bcolors.OKCYAN}======== Plotting Imag(E) vs. Beta ========{bcolors.ENDC}")
        E = Eimag
    else:
        print(f"\n\n{bcolors.OKBLUE}======== Plotting Real(E) vs. Beta ========{bcolors.ENDC}")
        E = Ereal
    dE = np.zeros(E.shape) # TODO: this is temporary, need to get dE for each block

    #2. (opt) reblock data
    if block is not None:
        print(f"  [{bcolors.OKGREEN}+{bcolors.ENDC}] reblocking with block size {bcolors.OKGREEN}{block}{bcolors.ENDC} ")        
        if weights is None:
            weights,_ = np.loadtxt("den.dat",unpack=True)
        if ignore > 0:
            weights = weights[ignore:]
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
    parser.add_argument('--ignore','-I', metavar='num_ignore', type=int,
                        action='store',default=0,
                        help='number of entries to ignore (from beginning of data)')

    parser.add_argument('--name','-n', metavar='name', type=str,
                        action='store',
                        default='energyVersusBeta.dat',
                        help='name of file containing AFQMC data')

    parser.add_argument('--verbose', '-v', action='count',
                        default=0,
                        help='set verbosity level, use more \'v\' chars to make output more verbose (i.e. -vv ) max. level is 3')

    parser.add_argument('--imag','-i',action='store_true',
                        help='plot imaginary part only (by default, the real part is plotted)')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    block = args.block_size
    fname = args.name
    verbose = args.verbose
    imag = args.imag
    ignore = args.ignore
    
    print(f"  [{bcolors.OKGREEN}+{bcolors.ENDC}] using verbosity level {bcolors.OKGREEN}{verbose}{bcolors.ENDC} ")

    equil_curve(fname,
                block=block,
                ignore=ignore,
                imag=imag)

if __name__ == '__main__':
    main()

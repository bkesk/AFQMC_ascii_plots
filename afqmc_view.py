#!/usr/bin/env python3

import logging # https://stackoverflow.com/questions/20240464/python-logging-file-is-not-working-when-using-logging-basicconfig/63868063

import numpy as np
import argparse
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

##############################################################
#                                                            #
#  Example: Q: what if we want to change our input settings? #
#                                                            #
##############################################################

def log_init(verbose):
    # see https://stackoverflow.com/questions/20240464/python-logging-file-is-not-working-when-using-logging-basicconfig/63868063
    level = { 0 : logging.ERROR, 
              1 : logging.WARNING,
              2 : logging.INFO,
              3 : logging.DEBUG}

    print(level[verbose])
    # version 1. (my preference) save to a file
    logging.basicConfig(filename='example.log',
                        filemode='w', 
                        level=level[verbose])

    # version 2. send to stdout
    #logging.basicConfig(level=level[verbose])

    # note: more recent versions of python may need the 'encoding'
    #           argument to be set (to 'utf-8' probably)


def pwrite(g, data):
    g.stdin.write(data.encode('utf-8'))

    ################
    #
    # EX: the dir() function
    #
    ##############
    logging.debug("what does the argument 'g' contain?")
    for d in dir(g):
        logging.debug(str(d))

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
    
    # plot errorbars separately
    #if dy is not None:
    #    pwrite(gplot,"set term dumb 80 20\n")
    #    pwrite(gplot,"plot '-' using 1:3 title 'dE (Beta)' with yerrorlines \n")
    #    for a,b,c in zip(x,y,dy):
    #        pwrite(gplot,f"{a} {b} {c}\n")
    #    pwrite(gplot,"e\n")
    #    gflush()

def equil_curve(fname, block=None, ignore=0, weights=None, imag=False):

    #1. get data
    beta,Ereal,Eimag = np.loadtxt(fname,unpack=True)
    
    if ignore > 0:
        beta = beta[ignore:]
        Ereal = Ereal[ignore:]
        Eimag = Eimag[ignore:]

    #E = np.abs(Ereal + 1j*Eimag)
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
    
    # EX: quickly add a new option

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

    #ex: logging.error(f"{bcolors.WARNING}KE: intentional error, as an example!{bcolors.ENDC}") # set default to 0
    parser.add_argument('--verbose', '-v', action='count',
                        default=0,
                        help='set verbosity level, use more \'v\' chars to make output more verbose (i.e. -vv ) max. level is 3')

    parser.add_argument('--imag','-i',action='store_true',
                        help='plot imaginary part only (by default, the real part is plotted)')

    args = parser.parse_args()
    return args

def main():
    # EX: the dir() function
    #print(dir(logging))

    # EX: careful not to call the logging before setting it up! (otherwise, it uses defaults!)
    args = get_args()

    block = args.block_size
    fname = args.name
    verbose = args.verbose
    imag = args.imag
    ignore = args.ignore
    
    # EX: logging module - good for debugging
    log_init(verbose)

    logging.info("Options are: ")
    logging.info(f" block_size={block} ")
    logging.info(f" file name={fname} ")
    logging.info(f" verbose={verbose} ")
    logging.info(f" plot imaginary part? {imag}")
    logging.info(f" num. entries to ignore (starting with lowest index): {ignore}")

    ##############
    #
    #  Ex: inspecting the "contents" of any python object
    #
    ##############

    #print("Contents of args's 'dir'")
    #for r in dir(args):
    #    print(f"{r}")
    
    
    print(f"  [{bcolors.OKGREEN}+{bcolors.ENDC}] using verbosity level {bcolors.OKGREEN}{verbose}{bcolors.ENDC} ")

    # run analysis
    equil_curve(fname,
                block=block,
                ignore=ignore,
                imag=imag)

if __name__ == '__main__':
    main()

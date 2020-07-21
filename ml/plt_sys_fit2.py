import os
import argparse

import numpy as np
np.random.seed(1234)

import csv
from csv import reader

import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_from_csv(workdir, method):
    with open(os.path.join(workdir, 'scan_data_{}.csv'.format(method)), "rU") as file:
        counts = []
        for line in file:
            lines = []
            elements = line.split()
            for i in range(0, len(elements)):
                lines.append(float(elements[i]))
            counts.append(lines)
    r = counts[0]
    deltaNLL = counts[1]
    return(r, deltaNLL)

def main(args):
    r, deltaNLL = load_from_csv(args.workdir1, 'nosys')
    r_sys, deltaNLL_sys = load_from_csv(args.workdir2, 'sys')

    plt.figure()
    plt.plot(r, deltaNLL, color='k')
    plt.plot(r_sys, deltaNLL_sys, color='r')
    plt.savefig(os.path.join('/home/gristo/', 'test_{}.png'.format(args.binning)), bbox_inches="tight")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir1', help='Working directory for nonsys outputs')
    parser.add_argument('workdir2', help='Working directory for sys outputs')
    parser.add_argument('binning', help='binning')
    args = parser.parse_args()
    main(args)
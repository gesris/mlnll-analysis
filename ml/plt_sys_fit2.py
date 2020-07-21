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
    r_sys_nosysimpl, deltaNLL_sys_nosysimpl = load_from_csv(args.workdir1, 'sys_nosysimpl')
    r_nosys_nosysimpl, deltaNLL_nosys_nosysimpl = load_from_csv(args.workdir2, 'nosys_nosysimpl')
    r_sys_sysimpl, deltaNLL_sys_sysimpl = load_from_csv(args.workdir3, 'sys_sysimpl')
    r_nosys_sysimpl, deltaNLL_nosys_sysimpl = load_from_csv(args.workdir4, 'nosys_sysimpl')

    x = np.linspace(0, 2, 100)
    hline = np.ones(len(x))

    plt.figure()
    plt.xlabel("mu")
    plt.xlim((0, 2))
    plt.ylabel("-2 Delta NLL")
    plt.ylim((0, 9))
    plt.plot(r_sys_nosysimpl, deltaNLL_sys_nosysimpl, color='k')
    plt.plot(r_nosys_nosysimpl, deltaNLL_nosys_nosysimpl, color='b')
    plt.plot(x, hline, color='r')
    idx = np.argwhere(np.diff(np.sign(deltaNLL_sys_nosysimpl - hline))).flatten()
    plt.plot(x[idx], hline[idx], 'ro')
    #plt.axhline(y=1., color='r')
    plt.plot([0], [0], color='k', label="stat + sys")
    plt.plot([0], [0], color='b', label="stat")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.savefig(os.path.join('/home/gristo/', 'scan_nosysimpl_{}.png'.format(args.binning)), bbox_inches="tight")
    
    plt.figure()
    plt.xlabel("mu")
    plt.xlim((0, 2))
    plt.ylabel("-2 Delta NLL")
    plt.ylim((0, 9))
    plt.plot(r_sys_sysimpl, deltaNLL_sys_sysimpl, color='k')
    plt.plot(r_nosys_sysimpl, deltaNLL_nosys_sysimpl, color='b')
    plt.axhline(y=1., color='r')
    plt.plot([0], [0], color='k', label="stat + sys")
    plt.plot([0], [0], color='b', label="stat")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.savefig(os.path.join('/home/gristo/', 'scan_sysimpl_{}.png'.format(args.binning)), bbox_inches="tight")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir1', help='Working directory for sys_nosysimpl outputs')
    parser.add_argument('workdir2', help='Working directory for nosys_nosysimpl outputs')
    parser.add_argument('workdir3', help='Working directory for sys_sysimpl outputs')
    parser.add_argument('workdir4', help='Working directory for nosys_sysimpl outputs')
    parser.add_argument('binning', help='binning')
    args = parser.parse_args()
    main(args)
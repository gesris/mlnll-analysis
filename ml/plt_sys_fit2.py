import os
import argparse

import numpy as np
np.random.seed(1234)
from scipy import interpolate

import csv
from csv import reader

import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc("font", size=16, family="serif")


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
    deltaNLL = np.array(counts[1]) * 2
    return(r, deltaNLL)

def main(args):
    r_sys_nosysimpl, deltaNLL_sys_nosysimpl = load_from_csv(args.workdir1, 'nosysimpl_sys')
    r_nosys_nosysimpl, deltaNLL_nosys_nosysimpl = load_from_csv(args.workdir2, 'nosysimpl_nosys')
    r_sys_sysimpl, deltaNLL_sys_sysimpl = load_from_csv(args.workdir3, 'wsysimpl_sys')
    r_nosys_sysimpl, deltaNLL_nosys_sysimpl = load_from_csv(args.workdir4, 'wsysimpl_nosys')


    ## Preprocessing SYS NOSYSIMPL
    f_deltaNLL_sys_nosysimpl = interpolate.UnivariateSpline(r_sys_nosysimpl, deltaNLL_sys_nosysimpl, s=0)
    x_r_sys_nosysimpl = np.arange(0.0, 2.0, 0.02)
    
    y_sys_nosysimpl = 1
    yreduced_sys_nosysimpl = np.array(deltaNLL_sys_nosysimpl) - y_sys_nosysimpl
    freduced_sys_nosysimpl = interpolate.UnivariateSpline(r_sys_nosysimpl, yreduced_sys_nosysimpl, s=0)
    constraints_sys_nosysimpl = [ 1 - freduced_sys_nosysimpl.roots()[0], freduced_sys_nosysimpl.roots()[1] - 1 ]


    ## Preprocessing NOSYS NOSYSIMPL
    f_deltaNLL_nosys_nosysimpl = interpolate.UnivariateSpline(r_nosys_nosysimpl, deltaNLL_nosys_nosysimpl, s=0)
    x_r_nosys_nosysimpl = np.arange(0.0, 2.0, 0.02)
    
    y_nosys_nosysimpl = 1
    yreduced_nosys_nosysimpl = np.array(deltaNLL_nosys_nosysimpl) - y_nosys_nosysimpl
    freduced_nosys_nosysimpl = interpolate.UnivariateSpline(r_nosys_nosysimpl, yreduced_nosys_nosysimpl, s=0)
    constraints_nosys_nosysimpl = [ 1 - freduced_nosys_nosysimpl.roots()[0], freduced_nosys_nosysimpl.roots()[1] - 1 ]


    ## Preprocessing SYS SYSIMPL
    f_deltaNLL_sys_sysimpl = interpolate.UnivariateSpline(r_sys_sysimpl, deltaNLL_sys_sysimpl, s=0)
    x_r_sys_sysimpl = np.arange(0.0, 2.0, 0.02)
    
    y_sys_sysimpl = 1
    yreduced_sys_sysimpl = np.array(deltaNLL_sys_sysimpl) - y_sys_sysimpl
    freduced_sys_sysimpl = interpolate.UnivariateSpline(r_sys_sysimpl, yreduced_sys_sysimpl, s=0)
    constraints_sys_sysimpl = [ 1 - freduced_sys_sysimpl.roots()[0], freduced_sys_sysimpl.roots()[1] - 1 ]


    ## Preprocessing NOSYS SYSIMPL
    f_deltaNLL_nosys_sysimpl = interpolate.UnivariateSpline(r_nosys_sysimpl, deltaNLL_nosys_sysimpl, s=0)
    x_r_nosys_sysimpl = np.arange(0.0, 2.0, 0.02)
    
    y_nosys_sysimpl = 1
    yreduced_nosys_sysimpl = np.array(deltaNLL_nosys_sysimpl) - y_nosys_sysimpl
    freduced_nosys_sysimpl = interpolate.UnivariateSpline(r_nosys_sysimpl, yreduced_nosys_sysimpl, s=0)
    constraints_nosys_sysimpl = [ 1 - freduced_nosys_sysimpl.roots()[0], freduced_nosys_sysimpl.roots()[1] - 1 ]

    y_limit = [0.0, 4.5]
    x_limit = [0.5, 1.5]

    plt.figure()
    plt.xlabel("$\mu$")
    plt.xlim((x_limit[0], x_limit[1]))
    plt.ylabel("-2 $\Delta$NLL")
    plt.ylim((y_limit[0], y_limit[1]))
    plt.plot(x_r_sys_nosysimpl, f_deltaNLL_sys_nosysimpl(x_r_sys_nosysimpl), color='#ff7f0e')
    plt.plot(x_r_nosys_nosysimpl, f_deltaNLL_nosys_nosysimpl(x_r_nosys_nosysimpl), color='#1f77b4')
    vscale = 1 / y_limit[1]
    plt.plot([x_limit[0], 1 - constraints_sys_nosysimpl[0]], [1, 1], '#d62728')
    plt.plot([1 + constraints_sys_nosysimpl[1], x_limit[1]], [1, 1], '#d62728')
    plt.plot([x_limit[0], 1 - constraints_nosys_nosysimpl[0]], [1, 1], '#d62728')
    plt.plot([1 + constraints_nosys_nosysimpl[1], x_limit[1]], [1, 1], '#d62728')
    plt.axvline(x=1 - constraints_sys_nosysimpl[0], ymax=1. * vscale, color='#d62728')
    plt.axvline(x=1 + constraints_sys_nosysimpl[1], ymax=1. * vscale, color='#d62728')
    plt.axvline(x=1 - constraints_nosys_nosysimpl[0], ymax=1. * vscale, color='#d62728')
    plt.axvline(x=1 + constraints_nosys_nosysimpl[1], ymax=1. * vscale, color='#d62728')
    plt.plot([0], [0], color='#ff7f0e', label="stat + sys:                           +{:.3f} / -{:.3f}".format(constraints_sys_nosysimpl[0], constraints_sys_nosysimpl[1]))
    plt.plot([0], [0], color='#1f77b4', label="stat:                                     +{:.3f} / -{:.3f}".format(constraints_nosys_nosysimpl[0], constraints_nosys_nosysimpl[1]))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.savefig(os.path.join('/home/gristo/', 'scan_{}_nosysimpl_{}.png'.format(args.systematic, args.binning)), bbox_inches="tight")
    
    plt.figure()
    plt.xlabel("$\mu$")
    plt.xlim((x_limit[0], x_limit[1]))
    plt.ylabel("-2 $\Delta$NLL")
    plt.ylim((y_limit[0], y_limit[1]))
    plt.plot(x_r_sys_sysimpl, f_deltaNLL_sys_sysimpl(x_r_sys_sysimpl), color='#ff7f0e')
    plt.plot(x_r_nosys_sysimpl, f_deltaNLL_nosys_sysimpl(x_r_nosys_sysimpl), color='#1f77b4')
    vscale = 1 / y_limit[1]
    plt.plot([x_limit[0], 1 - constraints_sys_sysimpl[0]], [1, 1], '#d62728')
    plt.plot([1 + constraints_sys_sysimpl[1], x_limit[1]], [1, 1], '#d62728')
    plt.plot([x_limit[0], 1 - constraints_nosys_sysimpl[0]], [1, 1], '#d62728')
    plt.plot([1 + constraints_nosys_sysimpl[1], x_limit[1]], [1, 1], '#d62728')
    plt.axvline(x=1 - constraints_sys_sysimpl[0], ymax=1. * vscale, color='#d62728')
    plt.axvline(x=1 + constraints_sys_sysimpl[1], ymax=1. * vscale, color='#d62728')
    plt.axvline(x=1 - constraints_nosys_sysimpl[0], ymax=1. * vscale, color='#d62728')
    plt.axvline(x=1 + constraints_nosys_sysimpl[1], ymax=1. * vscale, color='#d62728')
    plt.plot([0], [0], color='#ff7f0e', label="stat + sys:                           +{:.3f} / -{:.3f}".format(constraints_sys_sysimpl[0], constraints_sys_sysimpl[1]))
    plt.plot([0], [0], color='#1f77b4', label="stat:                                     +{:.3f} / -{:.3f}".format(constraints_nosys_sysimpl[0], constraints_nosys_sysimpl[1]))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.savefig(os.path.join('/home/gristo/', 'scan_{}_sysimpl_{}.png'.format(args.systematic, args.binning)), bbox_inches="tight")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir1', help='Working directory for sys_nosysimpl outputs')
    parser.add_argument('workdir2', help='Working directory for nosys_nosysimpl outputs')
    parser.add_argument('workdir3', help='Working directory for sys_sysimpl outputs')
    parser.add_argument('workdir4', help='Working directory for nosys_sysimpl outputs')
    parser.add_argument('binning', help='binning')
    parser.add_argument('systematic', help='systematic')
    args = parser.parse_args()
    main(args)
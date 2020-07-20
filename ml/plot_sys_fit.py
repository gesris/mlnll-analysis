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

def main(args):
    path = os.path.join(args.workdir, 'higgsCombine.Scan.MultiDimFit.mH125.root')
    #path = '/home/gristo/mlnll-analysis/output/4_bins_wsys_shapes/higgsCombine.Scan.MultiDimFit.mH125.root'
    df = ROOT.RDataFrame('limit', path)
    data = df.AsNumpy(['r', 'deltaNLL'])
    for i in range(len(data['r'])):
        print(data['r'][i])
    #plt.plot(data['r'], 2 * data['deltaNLL'], '+')
    #plt.savefig('plot.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    main(args)
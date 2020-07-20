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

def save_to_csv(args):
    # Load from ROOT file
    path = os.path.join(args.workdir, 'higgsCombine.Scan.MultiDimFit.mH125.root')
    df = ROOT.RDataFrame('limit', path)
    data = df.AsNumpy(['r', 'deltaNLL'])

    # Save as .csv
    open(os.path.join(args.workdir, 'scan_data_{}.csv'.format(args.method)), "w").close()
    with open(os.path.join(args.workdir, 'scan_data_{}.csv'.format(args.method)), "ab") as file:
        np.savetxt(file, [data['r']])
        np.savetxt(file, [data['deltaNLL']])

def main(args):
    save_to_csv(args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('method', help='Method with or without Sys')
    args = parser.parse_args()
    main(args)
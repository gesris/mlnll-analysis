import os
import argparse

import numpy as np
np.random.seed(1234)

import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
path = '/home/gristo/mlnll-analysis/output/4_bins_wsys_shapes/higgsCombine.Scan.MultiDimFit.mH125.root'
df = ROOT.RDataFrame('limit', path)
data = df.AsNumpy(['r', 'deltaNLL'])
for i in len(data['r'][:]):
    print(data['r'][i])
#plt.plot(data['r'], 2 * data['deltaNLL'], '+')
#plt.savefig('plot.png', bbox_inches='tight')
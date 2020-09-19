import ROOT
from root_numpy import array2tree
import numpy as np
from utils import config as cfg
import array

import os
import csv
from csv import reader

home_basepath = '/home/gristo/workspace/htautau/deeptau_02-20/2018/'

def save_to_csv(nparray, path, filename):
    data = np.asarray(nparray)
    np.savetxt(path + filename, data, delimiter=',')

def load_from_csv(path, filename):
    data = np.loadtxt(path + filename, delimiter=',')
    return data

for filename in cfg.files:
    #if filename in ['ggh']:
    for file_ in cfg.files[filename]:
        binning = load_from_csv(home_basepath + file_ , '/binning.csv')
        weights_up = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_up.csv'.format(file_))
        weights_down = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_down.csv'.format(file_))
        a = np.array(weights_up, dtype=[('jpt_1_weights_up', np.float32)])
        b = np.array(weights_down, dtype=[('jpt_1_weights_down', np.float32)])

        ## Make tree with two branches upweights and downweights
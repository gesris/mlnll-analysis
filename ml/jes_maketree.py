import ROOT
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
    print(filename)
    #if filename in ['ggh']:
    for file_ in cfg.files[filename]:
        binning = load_from_csv(home_basepath + file_ , '/binning.csv')
        print(binning)
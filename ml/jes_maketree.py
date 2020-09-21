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
    if filename in ['ggh']:
        for file_ in cfg.files[filename]:
            if file_ in ['GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2']:
                binning = load_from_csv(home_basepath + file_ , '/binning.csv')
                weights_up = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_up.csv'.format(file_))
                weights_down = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_down.csv'.format(file_))

                
                ## Make new root file with new tree with two branches upweights and downweights
                root_file = ROOT.TFile(home_basepath + file_ + '/jpt_1_weights.root', 'RECREATE')
                tree = ROOT.TTree('tree', 'jpt_1_weights')

                ## create 1 dimensional float arrays as fill variables, in this way the float
                ## array serves as a pointer which can be passed to the branch
                x = np.zeros(1, dtype=float)
                y = np.zeros(1, dtype=float)

                ## create the branches and assign the fill-variables to them as floats (F)
                tree.Branch('jpt_1_weights_up', x, 'jpt_1_weights_up/F')
                tree.Branch('jpt_1_weights_down', y, 'jpt_1_weights_down/F')
                
                
                ## Fill tree
                for i in range(len(weights_down)):
                    x[0] = weights_up[i]
                    y[0] = weights_down[i]
                    tree.Fill()
                #tree.Fill()

                root_file.Write()
                root_file.Close()
                

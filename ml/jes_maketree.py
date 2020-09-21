import ROOT
import numpy as np
from utils import config as cfg
from array import array

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
    if filename in 'ggh':
        for file_ in cfg.files[filename]:
            if file_ in 'GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2':
                binning = load_from_csv(home_basepath + file_ , '/binning.csv')
                weights_up = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_up.csv'.format(file_))
                weights_down = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_down.csv'.format(file_))

                
                ## Make new root file with new tree with two branches upweights and downweights
                root_file = ROOT.TFile(home_basepath + file_ + '/' + file_ + '.root', 'RECREATE')
                tree = ROOT.TTree('ntuple', 'ntuple')

                ## create 1 dimensional float arrays as fill variables, in this way the float
                ## array serves as a pointer which can be passed to the branch
                x = array('f', [0])
                y = array('f', [0])

                ## create the branches and assign the fill-variables to them as floats (F)
                tree.Branch('jpt_1_weights_up', x, 'jpt_1_weights_up/F')
                tree.Branch('jpt_1_weights_down', y, 'jpt_1_weights_down/F')

                ## Loading root files
                path = cfg.basepath + 'ntuples/' + file_ + '/' + file_ + '.root'
                nominal = ROOT.TFile(path)
                tree = nominal.Get("mt_nominal/ntuple")
                
                for event in tree:
                    if event.jpt_1 > binning[-1]:
                        ## fill tree with weight = 1
                        x[0] = 1.
                        y[0] = 1.
                        tree.Fill()
                    else:
                        left_binedge = binning[binning <= event.jpt_1]
                        index = np.where(binning==left_binedge)
                        x[0] = weights_up[index]
                        y[0] = weights_down[index]
                        tree.Fill()
                
                root_file.Write()
                root_file.Close()

                
                """
                ## Fill tree
                for i in range(len(weights_down)):
                    x[0] = weights_up[i]
                    y[0] = weights_down[i]
                    tree.Fill()

                root_file.Write()
                root_file.Close()
                """

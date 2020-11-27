import ROOT
import numpy as np
from utils import config as cfg
from array import array

import os
import csv
from csv import reader

import multiprocessing.dummy as mp

home_basepath = '/home/gristo/workspace_met/htautau/deeptau_02-20/2018/ntuples/'

def save_to_csv(nparray, path, filename):
    data = np.asarray(nparray)
    np.savetxt(path + filename, data, delimiter=',')

def load_from_csv(path, filename):
    data = np.loadtxt(path + filename, delimiter=',')
    return data

foldernames = [
        'mt_jerUncUp',
        'mt_jerUncDown',
        'mt_jecUncAbsoluteYearUp',
        'mt_jecUncAbsoluteYearDown',
        'mt_jecUncAbsoluteUp',
        'mt_jecUncAbsoluteDown',
        'mt_jecUncBBEC1YearUp',
        'mt_jecUncBBEC1YearDown',
        'mt_jecUncBBEC1Up',
        'mt_jecUncBBEC1Down',
        'mt_jecUncEC2YearUp',
        'mt_jecUncEC2YearDown',
        'mt_jecUncEC2Up',
        'mt_jecUncEC2Down',
        'mt_jecUncHFYearUp',
        'mt_jecUncHFYearDown',
        'mt_jecUncHFUp',
        'mt_jecUncHFDown',
        'mt_jecUncFlavorQCDUp',
        'mt_jecUncFlavorQCDDown',
        'mt_jecUncRelativeSampleYearUp',
        'mt_jecUncRelativeSampleYearDown',
        'mt_jecUncRelativeBalUp',
        'mt_jecUncRelativeBalDown',
        'mt_tauEsThreeProngUp',
        'mt_tauEsThreeProngDown',
        'mt_tauEsThreeProngOnePiZeroUp',
        'mt_tauEsThreeProngOnePiZeroDown',
        'mt_tauEsOneProngUp',
        'mt_tauEsOneProngDown',
        'mt_tauEsOneProngOnePiZeroUp',
        'mt_tauEsOneProngOnePiZeroDown',
        ]

for folder in foldernames:
    for filename in cfg.files:
        if filename in ['singlemuon']:
            for file_ in cfg.files[filename]:
                #if file_ in ['GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2']:
                ## Loadng TDirectory needet to clone
                f = ROOT.TFile(home_basepath + file_ + '/' + file_ + '.root', 'UPDATE')
                t = f.Get("mt_nominal/ntuple")

                ## Making new TDirectory
                d_new = ROOT.TDirectoryFile(folder, folder)
                d_new.cd()

                ## Cloning and Saving changes
                tree_clone = t.Clone()
                d_new.Write()
                f.Close()
            
        


"""
#for filename in cfg.files:
    #if filename in 'ggh':
    # pass

def job(filename):
    for file_ in cfg.files[filename]:
        if file_ in 'SingleMuon_Run2018A_17Sep2018v2_13TeV_MINIAOD':
            binning = load_from_csv(home_basepath + file_ , '/binning.csv')
            weights_up = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_up.csv'.format(file_))
            weights_down = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_down.csv'.format(file_))

            
            ## Make new root file with new tree with two branches upweights and downweights
            root_file = ROOT.TFile(home_basepath + file_ + '/' + file_ + '.root', 'RECREATE')
            tdirectory = ROOT.TDirectoryFile('mt_nominal', 'mt_nominal')
            tdirectory.cd()
            tree = ROOT.TTree('ntuple', 'ntuple')

            ## create 1 dimensional float arrays as fill variables, in this way the float
            ## array serves as a pointer which can be passed to the branch
            x = array('f', [0])
            y = array('f', [0])

            ## create the branches and assign the fill-variables to them as floats (F)
            tree.Branch('jpt_1_weights_up', x, 'jpt_1_weights_up/F')
            tree.Branch('jpt_1_weights_down', y, 'jpt_1_weights_down/F')

            ## Loading basepath root files
            path = cfg.basepath + 'ntuples/' + file_ + '/' + file_ + '.root'
            nominal = ROOT.TFile(path)
            tree_2 = nominal.Get("mt_nominal/ntuple")
            
            ## assigning specific weight to each event
            for event in tree_2:
                if event.jpt_1 > binning[-1]:
                    ## assign weight 1 to entries out of bounds
                    x[0] = 1.
                    y[0] = 1.
                    tree.Fill()
                else:
                    left_binedge = binning[binning <= event.jpt_1][-1]
                    index = np.where(binning==left_binedge)
                    x[0] = weights_up[index][0]
                    y[0] = weights_down[index][0]
                    tree.Fill()
            
            root_file.Write()
            root_file.Close()

## With multiprozessing with 1 core per category
if __name__=="__main__":
    filenames = []
    for filename in cfg.files:
        filenames.append(filename)
    #p = mp.Pool(len(filenames))
    p = mp.Pool(1)
    p.map(job, filenames)
    p.close()
    p.join()
"""

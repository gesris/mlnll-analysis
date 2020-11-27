from numpy.lib.twodim_base import tri
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

def job(filename):
    for file_ in cfg.files[filename]:
        print(file_)
        if file_ in ['VBFHToTauTauHTXSFilterSTXS1p1Bin203to205M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1']:
            met_binning = load_from_csv(home_basepath + file_ , '/met_binning.csv')
            met_weights_up = load_from_csv(home_basepath + file_ , '/{}_met_weights_up.csv'.format(file_))
            met_weights_down = load_from_csv(home_basepath + file_ , '/{}_met_weights_down.csv'.format(file_))

            
            ## Make new root file with new tree with two branches upweights and downweights
            root_file = ROOT.TFile(home_basepath + file_ + '/' + file_ + '.root', 'RECREATE')
            tdirectory = ROOT.TDirectoryFile('mt_nominal', 'mt_nominal')
            tdirectory.cd()
            tree = ROOT.TTree('ntuple', 'ntuple')


            ## create 1 dimensional float arrays as fill variables, in this way the float
            ## array serves as a pointer which can be passed to the branch
            met_x = array('f', [0])
            met_y = array('f', [0])


            ## create the branches and assign the fill-variables to them as floats (F)
            tree.Branch('met_weights_up', met_x, 'met_weights_up/F')
            tree.Branch('met_weights_down', met_y, 'met_weights_down/F')


            ## Loading basepath root files to match weight with event
            path = cfg.basepath + 'ntuples/' + file_ + '/' + file_ + '.root'
            nominal = ROOT.TFile(path)
            tree_2 = nominal.Get("mt_nominal/ntuple")
            

            ## assigning specific weight to each event
            ## MET
            for event in tree_2:
                if event.met > met_binning[-1]:
                    met_x[0] = 1.
                    met_y[0] = 1.
                else:
                    left_binedge = met_binning[met_binning <= event.met][-1]
                    index = np.where(met_binning==left_binedge)
                    print(left_binedge)
                    met_x[0] = met_weights_up[index][0]
                    met_y[0] = met_weights_down[index][0]                    
                tree.Fill()
            root_file.Write()
            root_file.Close()


def clone_to_all_tdirectories(tdirectories):
    for folder in tdirectories:
        for filename in cfg.files:
            print(filename)
            for file_ in cfg.files[filename]:
                #if file_ in 'VBFHToTauTauHTXSFilterSTXS1p1Bin203to205M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1':
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

    # clone_to_all_tdirectories(foldernames)

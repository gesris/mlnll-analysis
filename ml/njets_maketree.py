from numpy.lib.twodim_base import tri
import ROOT
import numpy as np
from utils import config as cfg
from array import array

import os
import csv
from csv import reader

import multiprocessing.dummy as mp

home_basepath = '/home/gristo/workspace/htautau/deeptau_02-20/2018/ntuples/'

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
        if file_ in ['SingleMuon_Run2018A_17Sep2018v2_13TeV_MINIAOD', 'W1JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2', 'DY1JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2', 'TTTo2L2Nu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1', 'WW_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_pythia8_v2', 'GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2', 'VBFHToTauTauHTXSFilterSTXS1p1Bin203to205M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1']:
            jpt1_binning = load_from_csv(home_basepath + file_ , '/jpt1_binning.csv')
            njets_binning = load_from_csv(home_basepath + file_ , '/njets_binning.csv')
            jpt1_weights_up = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_up.csv'.format(file_))
            jpt1_weights_down = load_from_csv(home_basepath + file_ , '/{}_jpt1_weights_down.csv'.format(file_))
            njets_weights_up = load_from_csv(home_basepath + file_ , '/{}_njets_weights_up.csv'.format(file_))
            njets_weights_down = load_from_csv(home_basepath + file_ , '/{}_njets_weights_down.csv'.format(file_))

            
            ## Make new root file with new tree with two branches upweights and downweights
            root_file = ROOT.TFile(home_basepath + file_ + '/' + file_ + '.root', 'RECREATE')
            tdirectory = ROOT.TDirectoryFile('mt_nominal', 'mt_nominal')
            tdirectory.cd()
            tree = ROOT.TTree('ntuple', 'ntuple')


            ## create 1 dimensional float arrays as fill variables, in this way the float
            ## array serves as a pointer which can be passed to the branch
            jpt1_x = array('f', [0])
            jpt1_y = array('f', [0])
            njets_x = array('f', [0])
            njets_y = array('f', [0])


            ## create the branches and assign the fill-variables to them as floats (F)
            tree.Branch('jpt_1_weights_up', jpt1_x, 'jpt_1_weights_up/F')
            tree.Branch('jpt_1_weights_down', jpt1_y, 'jpt_1_weights_down/F')
            tree.Branch('njets_weights_up', njets_x, 'njets_weights_up/F')
            tree.Branch('njets_weights_down', njets_y, 'njets_weights_down/F')


            ## Loading basepath root files to match weight with event
            path = cfg.basepath + 'ntuples/' + file_ + '/' + file_ + '.root'
            nominal = ROOT.TFile(path)
            tree_2 = nominal.Get("mt_nominal/ntuple")
            

            ## assigning specific weight to each event
            ## NJETS
            for event in tree_2:
                if event.njets > njets_binning[-2]:   #all entries over value of left bin edge of last bin are ignored
                    ## assign weight 1 to entries out of bounds
                    njets_x[0] = 1.
                    njets_y[0] = 1.

                    ## consider jpt_1 weights aswell
                    if event.jpt_1 > jpt1_binning[-1]:
                        jpt1_x[0] = 1.
                        jpt1_y[0] = 1.
                    else:
                        left_binedge = jpt1_binning[jpt1_binning <= event.jpt_1][-1]
                        index = np.where(jpt1_binning==left_binedge)
                        print(left_binedge)
                        jpt1_x[0] = jpt1_weights_up[index][0]
                        jpt1_y[0] = jpt1_weights_down[index][0]
                        
                    tree.Fill()
                else:
                    left_binedge = njets_binning[njets_binning <= event.njets][-1]
                    index = np.where(njets_binning==left_binedge)
                    print(left_binedge)
                    njets_x[0] = njets_weights_up[index][0]
                    njets_y[0] = njets_weights_down[index][0]

                    ## consider jpt_1 weights aswell
                    if event.jpt_1 > jpt1_binning[-1]:
                        jpt1_x[0] = 1.
                        jpt1_y[0] = 1.
                    else:
                        left_binedge = jpt1_binning[jpt1_binning <= event.jpt_1][-1]
                        index = np.where(jpt1_binning==left_binedge)
                        print(left_binedge)
                        jpt1_x[0] = jpt1_weights_up[index][0]
                        jpt1_y[0] = jpt1_weights_down[index][0]

                    tree.Fill()

            root_file.Write()
            root_file.Close()


def clone_to_all_tdirectories(tdirectories):
    for folder in tdirectories:
        for filename in cfg.files:
            print(filename)
            for file_ in cfg.files[filename]:
                #if file_ in 'W1JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2':
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
    p = mp.Pool(len(filenames))
    #p = mp.Pool(1)
    p.map(job, filenames)
    p.close()
    p.join()

    clone_to_all_tdirectories(foldernames)

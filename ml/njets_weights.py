import ROOT
import numpy as np
from utils import config as cfg
import array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#mpl.rc("font", size=16, family="serif")

import os
import csv
from csv import reader


home_basepath = '/home/gristo/workspace_njets/htautau/deeptau_02-20/2018/ntuples/'

def save_to_csv(nparray, path, filename):
    data = np.asarray(nparray)
    np.savetxt(path + filename, data, delimiter=',')

def load_from_csv(path, filename):
    data = np.loadtxt(path + filename, delimiter=',')
    return data


for filename in cfg.files:
    #print(filename)
    #if filename in 'ggh':
    for file_ in cfg.files[filename]:
        print(file_)
        #if file_ in ['GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2']:
        ## Make directory for Hist and .csv with weights
        if os.path.exists(home_basepath + file_):
            print("Directory {} exists".format(home_basepath + file_))
        else:
            try:
                os.mkdir(home_basepath + file_)
            except OSError:
                print("Creating directory {} failed".format(home_basepath + file_))
            else:
                print("Successfully created directory %s " % [home_basepath + file_])
        
        ## Loading root files
        path = cfg.basepath + 'ntuples/' + file_ + '/' + file_ + '.root'
        nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["njets"])
        
        ## Prepatre for Hist
        nbins = 10
        minrange = 0
        maxrange = 10
        binning = np.linspace(minrange, maxrange, nbins + 1)
        heights_nom, bins = np.histogram(nominal["njets"], bins=nbins, range=(minrange, maxrange))
        
        ## Calculate shifts
        ## Upshift: add 20% of first bin on first bin and substract same amount from second bin
        ## Downshift: addsubtract 20% of first bin from first bin and add same amount on second bin
        # diff = heights_nom[0] * 0.2
        # heights_up = np.append([heights_nom[0] + diff, heights_nom[1] - diff], heights_nom[2:])
        # heights_down = np.append([heights_nom[0] - diff, heights_nom[1] + diff], heights_nom[2:])

        ## New more advanced shift: add/subtract 1 to every event, except edges
        ## Upshift
        upshift = np.zeros(nbins)
        for i, element in enumerate(heights_nom):
            step = element * 0.1
            upshift[i] = upshift[i] + element - step
            upshift[i + 1] = upshift[i + 1] + step
            if i == 8:
                upshift[i + 1] = heights_nom[i + 1]
                break
        heights_up = upshift

        ## Downshift
        downshift = np.zeros(nbins)
        for i, element in enumerate(heights_nom):
            if i == 0:
                downshift[i] = element
            else:
                step = element * 0.1
                downshift[i] = downshift[i] + element - step
                downshift[i - 1] = downshift[i - 1] + step
        heights_down = downshift

        print(heights_nom, np.sum(heights_nom))
        print(heights_up, np.sum(heights_up))
        print(heights_down, np.sum(heights_down))


        ## Calculate weights
        epsilon = 1e-6
        heights_nom = heights_nom.astype(float)
        heights_up = heights_up.astype(float)
        heights_down = heights_down.astype(float)
        heights_nom[heights_nom <= 0] = epsilon
        heights_up[heights_up <= 0] = epsilon
        heights_down[heights_down <= 0] = epsilon

        weights_up = heights_up / heights_nom
        weights_down = heights_down / heights_nom

        weights_up[weights_up <= 0] = epsilon
        weights_down[weights_down <= 0] = epsilon

        ## Save weights to .csv
        #save_to_csv(weights_up, home_basepath + file_, '/{}_njets_weights_up.csv'.format(file_))
        np.savetxt(home_basepath + file_ + '/{}_njets_weights_up.csv'.format(file_), np.asarray(weights_up), delimiter=',')
        np.savetxt(home_basepath + file_ + '/{}_njets_weights_down.csv'.format(file_), np.asarray(weights_down), delimiter=',')
        np.savetxt(home_basepath + file_ + '/njets_binning.csv', np.asarray(binning), delimiter=',')

        
        ## Make Histogram
        bins_center = []
        for left, right in zip(bins[1:], bins[:-1]):
            bins_center.append(left + (right - left) / 2)

        plt.figure(figsize=(7, 6))
        plt.hist(bins_center, weights=heights_nom, bins=bins, histtype="step", lw=1.5, color='C0')
        plt.hist(bins_center, weights=heights_up, bins=bins, histtype="step", lw=1.5, ls=':', color='C1')
        plt.hist(bins_center, weights=heights_down, bins=bins, histtype="step", lw=1.5, ls='--', color='C1')
        plt.plot([0], [0], lw=2, color='C0', label="nominal")
        plt.plot([0], [0], lw=2, ls=':', color='C1', label="up shift")
        plt.plot([0], [0], lw=2, ls='--', color='C1', label="down shift")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
        plt.xlabel("njets")
        plt.ylabel("Counts")
        plt.savefig(home_basepath + file_ + '/{}_njets_shapeshift.png'.format(file_), bbox_inches = "tight")

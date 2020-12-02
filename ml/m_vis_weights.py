import ROOT
import numpy as np
from utils import config as cfg

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc("font", size=16, family="serif")

import os


home_basepath = '/home/gristo/workspace_m_vis/htautau/deeptau_02-20/2018/ntuples/'

for filename in cfg.files:
    print(filename)
    #if filename in 'qqh':
    for file_ in cfg.files[filename]:
        print(file_)
        #if file_ in ['VBFHToTauTauHTXSFilterSTXS1p1Bin203to205M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1']:
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
        ## Read branch in dictionary
        nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["m_vis"])
        
        ## Prepatre for Hist
        bins = 20
        minrange = 0
        maxrange = 220
        binning = np.linspace(minrange, maxrange, bins + 1)
        heights_nom, bins = np.histogram(nominal["m_vis"], bins=bins, range=(minrange, maxrange))
        
        ## Calculate shifts
        ## Upshift: scale every event by 1.1
        ## Downshift: scale every event by 0.9
        heights_up, _ = np.histogram(nominal["m_vis"] * 1.2, bins=bins, range=(minrange, maxrange))
        heights_down, _ = np.histogram(nominal["m_vis"] * 0.8, bins=bins, range=(minrange, maxrange))


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
        print("Saving weights as .csv")
        np.savetxt(home_basepath + file_ + '/{}_m_vis_weights_up.csv'.format(file_), np.asarray(weights_up), delimiter=',')
        np.savetxt(home_basepath + file_ + '/{}_m_vis_weights_down.csv'.format(file_), np.asarray(weights_down), delimiter=',')
        np.savetxt(home_basepath + file_ + '/m_vis_binning.csv', np.asarray(binning), delimiter=',')

        
        ## Make Histogram
        print("Plotting histogram")
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
        plt.xlabel("m_vis")
        plt.ylabel("Counts")
        plt.savefig(home_basepath + file_ + '/{}_m_vis_shapeshift.png'.format(file_), bbox_inches = "tight")
        print("Done \n\n")

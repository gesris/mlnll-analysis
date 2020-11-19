import ROOT
import numpy as np
from utils import config as cfg
import array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc("font", size=16, family="serif")

import os
            


for filename in cfg.files:
    print(filename)
    if filename in 'wjets':
        for file_ in cfg.files[filename]:
            print(file_)
            if file_ in ['W1JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2']:
                
                ## Loading root file
                path = cfg.basepath + 'ntuples/' + file_ + '/' + file_ + '.root'
                file = ROOT.TFile(path)

                ## Prepare for hist
                bins = 10
                #minrange = -10
                #maxrange = 800
                minrange = 0
                maxrange = 200
                binning = np.linspace(minrange, maxrange, bins + 1)
                nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["jpt_1"])
                heights_nom, _ = np.histogram(nominal["jpt_1"], bins=bins, range=(minrange, maxrange))
                hist_upshifts = {}
                hist_downshifts = {}

                for key in file.GetListOfKeys():
                    name = key.GetName()
                    if 'mt_jecUnc' in name:
                        if 'Up' in name:
                            upshift = ROOT.RDataFrame(name + '/ntuple', path).AsNumpy(["jpt_1"])
                            hist_upshifts[name], _ = np.histogram(upshift["jpt_1"], bins=bins, range=(minrange, maxrange))
                            
                        elif 'Down' in name: 
                            downshift = ROOT.RDataFrame(name + '/ntuple', path).AsNumpy(["jpt_1"])
                            hist_downshifts[name], _ = np.histogram(downshift["jpt_1"], bins=bins, range=(minrange, maxrange))           
                            
                ## Make histograms
                bins_center = []
                for left, right in zip(binning[1:], binning[:-1]):
                    bins_center.append(left + (right - left) / 2)
                
                for uncertainty_up, uncertainty_down in zip(hist_upshifts, hist_downshifts):
                    plt.figure(figsize=(7, 6))
                    plt.hist(bins_center, weights=heights_nom, bins=bins, histtype="step", lw=1.5, color='C0')
                    plt.hist(bins_center, weights=hist_upshifts[uncertainty_up], bins=bins, histtype="step", lw=1.5, ls=':', color='C1')
                    plt.hist(bins_center, weights=hist_downshifts[uncertainty_down], bins=bins, histtype="step", lw=1.5, ls='--', color='C1')
                    plt.plot([0], [0], lw=2, color='C0', label="nominal")
                    plt.plot([0], [0], lw=2, ls=':', color='C1', label="up shift")
                    plt.plot([0], [0], lw=2, ls='--', color='C1', label="down shift")
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
                    plt.xlabel("jpt_1")
                    plt.ylabel("Counts")
                    plt.savefig('/home/gristo/workspace/jpt_1_plots/{}.png'.format(uncertainty_up), bbox_inches = "tight")



       
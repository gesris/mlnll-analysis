import ROOT
import numpy as np
from utils import config as cfg

import matplotlib
matplotlib.use('Agg')
matplotlib.rc("font", size=16, family="serif")
matplotlib.rc("text", usetex=False)
matplotlib.rc('text.latex', preamble=r'\usepackage{cancel}')
from matplotlib import gridspec
import matplotlib.pyplot as plt

import os


home_basepath = '/home/gristo/workspace_met/htautau/deeptau_02-20/2018/ntuples/'

for filename in cfg.files:
    print(filename)
    if filename in 'ggh':
        for file_ in cfg.files[filename]:
            print(file_)
            if file_ in ['GluGluHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2']:
            
                ## Loading root files
                path = cfg.basepath + 'ntuples/' + file_ + '/' + file_ + '.root'
                ## Read branch in dictionary
                nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["met"])
                
                ## Prepatre for Hist
                bins = 20
                minrange = 0
                maxrange = 220
                binning = np.linspace(minrange, maxrange, bins + 1)
                heights_nom, bins = np.histogram(nominal["met"], bins=bins, range=(minrange, maxrange))
                
                ## Calculate shifts
                ## Upshift: scale every event by 1.1
                ## Downshift: scale every event by 0.9
                heights_up, _ = np.histogram(nominal["met"] * 1.1, bins=bins, range=(minrange, maxrange))
                heights_down, _ = np.histogram(nominal["met"] * 0.9, bins=bins, range=(minrange, maxrange))


                ## Calculate weights
                epsilon = 1
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


                print("Plotting histogram")
                bins_center = []
                for left, right in zip(bins[1:], bins[:-1]):
                    bins_center.append(left + (right - left) / 2)
                lw = 2
                label = "$\cancel{E}_{\mathrm{T}}$ in GeV"
                plt.figure(figsize=(6, 6))
                plt.subplots_adjust(hspace=0.05)
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
                h_up, _, _ = ax1.hist(bins_center, bins, weights=heights_up, histtype="step", lw=lw, color="C2", ls="-")
                h_down, _, _ = ax1.hist(bins_center, bins, weights=heights_down, histtype="step", lw=lw, color="C3", ls="-")
                h_nom, _, _ = ax1.hist(bins_center, bins, weights=heights_nom, histtype="step", lw=lw, color="C7", ls="-")
                p_nom, = ax1.plot([-999], [-999], lw=lw, color="C7", label="Nominal")
                p_up, = ax1.plot([-999], [-999], lw=lw, color="C2", label="Up-shift")
                p_down, = ax1.plot([-999], [-999], lw=lw, color="C3", label="Down-shift")
                ax2.plot(bins_center, h_up / h_nom, "o-", mew=3, ms=4, lw=1, color="C2")
                ax2.plot(bins_center, h_down / h_nom, "o-", mew=3, ms=4, lw=1, color="C3")
                ax1.set_xlim((bins[0], bins[-1]))
                ax2.set_xlim((bins[0], bins[-1]))
                ax2.set_xlabel(label)
                ax1.set_ylabel("Count")
                ax2.set_ylabel("Ratio to nominal\n")
                ax1.set_xticklabels([])
                #ax1.set_yticks([50, 100, 150, 200, 250])
                ax1.set_ylim([0, np.max([np.max(h_up), np.max(h_down)]) * 1.2])
                ax2.set_yticks([0.5, 1.0, 1.5])
                ax2.set_ylim((0.0, 2.0))
                ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,3))
                ax1.legend(handles=[p_nom, p_up, p_down])
                plt.savefig(home_basepath + file_ + '/{}_met_shapeshift_fancy.png'.format(file_), bbox_inches = "tight")



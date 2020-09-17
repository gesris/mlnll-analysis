import ROOT
import numpy as np
from utils import config as cfg
import array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#mpl.rc("font", size=16, family="serif")

# cfg.basepath + cfg.files = root file location
"""
path = '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root'
df_nominal = ROOT.RDataFrame('mt_nominal/ntuple', path)
df_up = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path)
df_down = ROOT.RDataFrame('mt_jecUncRelativeBalDown/ntuple', path)

#dir_up = df_up.AsNumpy(columns=["jpt_1"])    # hist is now a dictionary with entries for jpt_1
#print(dir_up["jpt_1"])

nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["jpt_1"])
upshift = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path).AsNumpy(["jpt_1"])
downshift = ROOT.RDataFrame('mt_jecUncRelativeBalDown/ntuple', path).AsNumpy(["jpt_1"])
#diff = nominal["jpt_1"] - upshift["jpt_1"]

heights_nom, bins = np.histogram(nominal["jpt_1"], bins=10, range=(-10, 800))
heights_up, _ = np.histogram(upshift["jpt_1"], bins=10, range=(-10, 800))
heights_down, _ = np.histogram(downshift["jpt_1"], bins=10, range=(-10, 800))

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
plt.xlabel("jpt_1")
plt.ylabel("Counts")
plt.savefig('/home/gristo/workspace/plots/test_hist.png', bbox_inches = "tight")
"""




for filename in cfg.files:
    if 'ggh' in filename:
        print(filename)
        tot_nom = np.zeros(10)
        tot_upshift = np.zeros(10)
        tot_downshift = np.zeros(10)
        n = 0

        for file in cfg.files[filename]:
            path = cfg.basepath + 'ntuples/' + file + '/' + file + '.root'
            df_nominal = ROOT.RDataFrame('mt_nominal/ntuple', path)
            nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["jpt_1"])
            heights_nom, bins = np.histogram(nominal["jpt_1"], bins=10, range=(-10, 800))

            ## SUM OF SQUARES
            tot_nom += np.square(heights_nom)
            n += 1
            
            f = ROOT.TFile(path)
            for key in f.GetListOfKeys():
                name = key.GetName()

                if 'mt_jecUnc' in name:
                    if 'Up' in name:
                        df_up = ROOT.RDataFrame(name + '/ntuple', path)
                        upshift = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path).AsNumpy(["jpt_1"])
                        heights_up, _ = np.histogram(upshift["jpt_1"], bins=10, range=(-10, 800))
                        
                        ## SUM Of SQUARE DIFF
                        tot_upshift += np.square(heights_up - heights_nom)

                    elif 'Down' in name:
                        df_down = ROOT.RDataFrame(name + '/ntuple', path)   
                        downshift = ROOT.RDataFrame('mt_jecUncRelativeBalDown/ntuple', path).AsNumpy(["jpt_1"])
                        heights_down, _ = np.histogram(downshift["jpt_1"], bins=10, range=(-10, 800))           
                        
                        ## SUM Of SQUARE DIFF
                        tot_downshift += np.square(heights_nom - heights_down)

        tot_nom = np.sqrt(tot_nom / n)
        tot_upshift = np.sqrt(tot_upshift)
        tot_downshift = np.sqrt(tot_downshift)

        bins_center = []
        for left, right in zip(bins[1:], bins[:-1]):
            bins_center.append(left + (right - left) / 2)

        plt.figure(figsize=(7, 6))
        plt.hist(bins_center, weights=tot_nom, bins=bins, histtype="step", lw=1.5, color='C0')
        plt.hist(bins_center, weights=tot_nom + tot_upshift, bins=bins, histtype="step", lw=1.5, ls=':', color='C1')
        plt.hist(bins_center, weights=tot_nom - tot_downshift, bins=bins, histtype="step", lw=1.5, ls='--', color='C1')
        plt.plot([0], [0], lw=2, color='C0', label="nominal")
        plt.plot([0], [0], lw=2, ls=':', color='C1', label="up shift")
        plt.plot([0], [0], lw=2, ls='--', color='C1', label="down shift")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
        plt.xlabel("jpt_1")
        plt.ylabel("Counts")
        plt.savefig('/home/gristo/workspace/plots/jes_totshift_{}.png'.format(filename), bbox_inches = "tight")


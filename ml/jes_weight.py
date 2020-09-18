import ROOT
import numpy as np
from utils import config as cfg
import array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#mpl.rc("font", size=16, family="serif")

import os

# homedir: /home/gristo/workspace/htautau/deeptau_02-20/2018
home_basepath = '/home/gristo/workspace/htautau/deeptau_02-20/2018/'


for filename in cfg.files:
    print(filename)
    if filename in ['qqh']:
        for file_ in cfg.files[filename]:
            if file_ in ['WplusHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2']:
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
                nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["jpt_1"])

                ## Prepatre for Hist
                bins = 50
                file_upshift = np.zeros(bins)
                file_downshift = np.zeros(bins)
                heights_nom, bins = np.histogram(nominal["jpt_1"], bins=bins, range=(-10, 800))

                ## Calculate shifts
                f = ROOT.TFile(path)
                for key in f.GetListOfKeys():
                    name = key.GetName()

                    if 'mt_jecUnc' in name:
                        if 'Up' in name:
                            upshift = ROOT.RDataFrame(name + '/ntuple', path).AsNumpy(["jpt_1"])
                            heights_up, _ = np.histogram(upshift["jpt_1"], bins=bins, range=(-10, 800))
                            
                            ## SUM Of SQUARE DIFF
                            file_upshift += np.square(heights_up - heights_nom)

                        elif 'Down' in name: 
                            downshift = ROOT.RDataFrame(name + '/ntuple', path).AsNumpy(["jpt_1"])
                            heights_down, _ = np.histogram(downshift["jpt_1"], bins=bins, range=(-10, 800))           
                            
                            ## SUM Of SQUARE DIFF
                            file_downshift += np.square(heights_nom - heights_down)

                file_upshift = np.sqrt(file_upshift)
                file_downshift = np.sqrt(file_downshift)

                ## Calculate weights
                epsilon = 1e-6
                heights_up[heights_up == 0] = epsilon
                heights_down[heights_down == 0] = epsilon
                heights_nom[heights_nom == 0] = epsilon

                weights_up = (heights_nom + heights_up) / heights_nom
                weights_down = (heights_nom - heights_down) / heights_nom
                print("WEIGHTS DOWN:\n{}".format(weights_down))

                """
                ## Make Histogram
                bins_center = []
                for left, right in zip(bins[1:], bins[:-1]):
                    bins_center.append(left + (right - left) / 2)

                plt.figure(figsize=(7, 6))
                plt.hist(bins_center, weights=heights_nom, bins=bins, histtype="step", lw=1.5, color='C0')
                plt.hist(bins_center, weights=heights_nom + file_upshift, bins=bins, histtype="step", lw=1.5, ls=':', color='C1')
                plt.hist(bins_center, weights=heights_nom - file_downshift, bins=bins, histtype="step", lw=1.5, ls='--', color='C1')
                plt.plot([0], [0], lw=2, color='C0', label="nominal")
                plt.plot([0], [0], lw=2, ls=':', color='C1', label="up shift")
                plt.plot([0], [0], lw=2, ls='--', color='C1', label="down shift")
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
                plt.xlabel("jpt_1")
                plt.ylabel("Counts")
                plt.savefig(home_basepath + file_ + '/{}_jpt1_totshift.png'.format(file_), bbox_inches = "tight")
                """

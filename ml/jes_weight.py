import ROOT
import numpy as np
from utils import config as cfg
import matplotlib.pyplot as plt

# cfg.basepath + cfg.files = root file location

df = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root')
hist = df.AsNumpy(columns=["jpt_1"])    # hist is now a dictionary with entries for jpt_1
print(hist["jpt_1"])


"""
for name in cfg.files:
    for path in cfg.files[name]:
        f = ROOT.TFile(cfg.basepath + 'ntuples/' + path + '/' + path + '.root')
        #d = f.Get(path)
        for key in f.GetListOfKeys():
            name = key.GetName()
            if 'mt_jecUnc' in name:
                if 'Up' in name:
                    d = f.Get(name)
                    tree = d.Get('ntuple')
                    
                elif 'Down' in name:
                    pass
"""

plt.hist(hist["jpt_1"])
plt.savefig("/home/gristo/workspace/plots/test_histogram.png")

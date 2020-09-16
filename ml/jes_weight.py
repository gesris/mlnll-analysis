import ROOT
import numpy as np
from utils import config as cfg
import array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc("font", size=16, family="serif")

# cfg.basepath + cfg.files = root file location
path = '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root'



df_nominal = ROOT.RDataFrame('mt_nominal/ntuple', path)
df_up = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path)

dir_up = df_up.AsNumpy(columns=["jpt_1"])    # hist is now a dictionary with entries for jpt_1
print(dir_up["jpt_1"])

hist_nominal = df_nominal.Histo1D("jpt_1")
hist_up = df_up.Histo1D("jpt_1")

hist_nominal.Add(hist_up.GetPtr(), -1)
for i in range(20):
    print(hist_nominal.GetBinContent(i))



#nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["jpt_1"])
#upshift = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path).AsNumpy(["jpt_1"])
#print(nominal["jpt_1"] - upshift["jpt_1"])



"""
for name in cfg.files:
    for path in cfg.files[name]:
        f = ROOT.TFile(cfg.basepath + 'ntuples/' + path + '/' + path + '.root')
        #d = f.Get(path)
        for key in f.GetListOfKeys():
            name = key.GetName()
            if 'mt_jecUnc' in name:
                if 'Up' in name:
                    pass                    
                elif 'Down' in name:
                    pass
"""


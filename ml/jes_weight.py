import ROOT
import numpy as np
from utils import config as cfg
import matplotlib.pyplot as plt

# cfg.basepath + cfg.files = root file location
path = '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root'
df = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path)
hist = df.AsNumpy(columns=["jpt_1"])    # hist is now a dictionary with entries for jpt_1
print(hist["jpt_1"])

nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["jpt_1"])
upshift = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path).AsNumpy(["jpt_1"])

print(nominal["jpt_1"] - upshift["jpt_1"])

bins = np.linspace(0, 800, 9)
bins_center = []
for left, right in zip(bins[1:], bins[:-1]):
    bins_center.append(left + (right - left) / 2)

plt.figure(figsize=(7, 6))
plt.hist(bins_center, weights=nominal["jpt_1"], bins=bins, histtype="step", lw=2, color='C0')
plt.savefig('/home/gristo/workspace/plots/test_histogram.png')



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


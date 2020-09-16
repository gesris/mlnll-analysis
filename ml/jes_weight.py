import ROOT
import numpy as np
from utils import config as cfg
import array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#mpl.rc("font", size=16, family="serif")

# cfg.basepath + cfg.files = root file location

path = '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root'
df_nominal = ROOT.RDataFrame('mt_nominal/ntuple', path)
df_up = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path)

dir_up = df_up.AsNumpy(columns=["jpt_1"])    # hist is now a dictionary with entries for jpt_1
print(dir_up["jpt_1"])

#hist_nominal = df_nominal.Histo1D("jpt_1")
#hist_nominal2 = df_nominal.Histo1D("jpt_1")
#hist_up = df_up.Histo1D("jpt_1")

#hist_nominal.Add(hist_up.GetPtr(), -1)
#for i in range(1, 70):
#    print(hist_nominal2.GetBinContent(i), hist_up.GetBinContent(i), hist_nominal.GetBinContent(i))


nominal = ROOT.RDataFrame('mt_nominal/ntuple', path).AsNumpy(["jpt_1"])
upshift = ROOT.RDataFrame('mt_jecUncRelativeBalUp/ntuple', path).AsNumpy(["jpt_1"])
diff = nominal["jpt_1"] - upshift["jpt_1"]
print(diff)

heights_nom, bins = np.histogram(nominal["jpt_1"], bins=20, range=(-10, 800))
heights_up, _ = np.histogram(upshift["jpt_1"], bins=20, range=(-10, 800))
#print(heights_nom)
#print(bins)

bins_center = []
for left, right in zip(bins[1:], bins[:-1]):
    bins_center.append(left + (right - left) / 2)

plt.figure(figsize=(7, 6))
plt.hist(bins_center, weights=heights_nom, bins=bins, histtype="step", lw=1, color='C0')
plt.hist(bins_center, weights=heights_up, bins=bins, histtype="step", lw=1, color='C1')
plt.plot([0], [0], lw=2, color='C0', label="nominal")
plt.plot([0], [0], lw=2, color='C1', label="up shift")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
plt.xlabel("jpt_1")
plt.ylabel("Counts")
plt.yscale('log')
plt.savefig('/home/gristo/workspace/plots/test_hist.png', bbox_inches = "tight")




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


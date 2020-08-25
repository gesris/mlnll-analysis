import ROOT
import numpy as np

path = '/work/gristo/second_mlnll-analysis/output/8_bins_nosysimpl_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')

tot_jes_hist = ROOT.TH1F("tot_jes_hist", "", 8, 0, 1)
entries = []
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Up' in name:
        h = d.Get(name)
        tot_jes_hist.Add(h)

for i in range(0, 13):
    entries.append(tot_jes_hist.GetBinContent(i))

print(entries)

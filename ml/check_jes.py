import ROOT
import numpy as np

path = '/work/gristo/second_mlnll-analysis/output/8_bins_nosysimpl_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')

for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Up' in name:
        print(name)
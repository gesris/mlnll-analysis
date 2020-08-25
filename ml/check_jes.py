import ROOT
import numpy as np

path = '/work/gristo/second_mlnll-analysis/output/8_bins_nosysimpl_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')

tot_jes_upshift = ROOT.TH1F("tot_jes_upshift", "", 8, 0, 1)
jes_upshift = []
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Up' in name:
        h = d.Get(name)
        tot_jes_upshift.Add(h)

tot_jes_downshift = ROOT.TH1F("tot_jes_downshift", "", 8, 0, 1)
jes_downshift = []
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Down' in name:
        h = d.Get(name)
        tot_jes_downshift.Add(h)

tot_sig_bkg = ROOT.TH1F("tot_sig_bkg", "", 8, 0, 1)
sig_bkg = []
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Up' in name:
        pass
    elif 'Down' in name:
        pass
    elif 'data_obs' in name:
        pass
    else:
        h = d.Get(name)
        tot_sig_bkg.Add(h)

for i in range(1, 9):
    jes_upshift.append(tot_jes_upshift.GetBinContent(i))
    jes_downshift.append(tot_jes_downshift.GetBinContent(i))
    sig_bkg.append(tot_sig_bkg.GetBinContent(i))

print("UPSHIFT: {} \nDOWNSHIFT: {}\nSIG+BKG: {}".format(jes_upshift, jes_downshift, sig_bkg))

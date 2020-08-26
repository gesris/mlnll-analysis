import ROOT
import numpy as np
from utils import config as cfg

path = '/work/gristo/second_mlnll-analysis/output/8_bins_nosysimpl_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')
classes = ['W', 'ZTT', 'ZL', 'ZJ', 'TTT', 'TTL', 'TTJ', 'VVJ', 'VVT', 'VVL', 'ggH125', 'qqH125']

diff_hist = ROOT.TH1F("DIFF", "", 8, 0, 1)
tot_jes_upshift = ROOT.TH1F("tot_jes_upshift", "", 8, 0, 1)
jes_upshift = []
upshifts = {}
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Up' in name:
        for class_name in classes:
            if class_name in name:
                h_nom = d.Get(class_name)
                h_shift = d.Get(name)
                # subtract hists to get shift only
                diff_hist.Add(h_nom, h_shift, -1, 1)
                tot_jes_upshift.Add(diff_hist)

                shift_array = []
                nom_array = []
                for i in range(1, 9):
                    shift_array.append(h_shift.GetBinContent(i)**2)
                    nom_array.append(h_shift.GetBinContent(i)**2)
                upshifts[name] = np.array(shift_array) - np.array(nom_array)
tot_upshifts = [0, 0, 0, 0, 0, 0, 0, 0]
for h in upshifts:
    tot_upshifts += upshifts[h]
    print(upshifts[h])

#print(tot_upshifts)


downshifts = {}
tot_jes_downshift = ROOT.TH1F("tot_jes_downshift", "", 8, 0, 1)
jes_downshift = []
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Down' in name:
        for class_name in classes:
            if class_name in name:
                h_nom = d.Get(class_name)
                h_shift = d.Get(name)
                # subtract hists to get shift only
                diff_hist.Add(h_nom, h_shift, 1, -1)
                tot_jes_downshift.Add(diff_hist)

                placeholder_array = []
                for i in range(1, 9):
                    placeholder_array.append(h_shift.GetBinContent(i))
                downshifts[name] = placeholder_array


tot_sig_bkg = ROOT.TH1F("tot_sig_bkg", "", 8, 0, 1)
sig_bkg = []
procs = {}
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

        placeholder_array = []
        for i in range(1, 9):
            placeholder_array.append(h.GetBinContent(i))
        procs[name] = placeholder_array


for i in range(1, 9):
    jes_upshift.append(tot_jes_upshift.GetBinContent(i))
    jes_downshift.append(tot_jes_downshift.GetBinContent(i))
    sig_bkg.append(tot_sig_bkg.GetBinContent(i))

#print("UPSHIFT: {} \nSUM: {}".format(jes_upshift,np.sum(jes_upshift)))
#print("DOWNSHIFT: {} \nSUM: {}".format(jes_downshift,np.sum(jes_downshift)))
#print("SIG + BKG: {} \nSUM: {}".format(sig_bkg,np.sum(sig_bkg)))

"""
Signal = ['ggH125', 'qqH125']
Background = ['W', 'ZTT', 'ZL', 'ZJ', 'TTT', 'TTL', 'TTJ', 'VVJ', 'VVT', 'VVL', 'QCD']

## Calculate NLL
bins = np.array(cfg.analysis_binning)
for i, (up, down) in enumerate(zip(bins[1:], bins[:-1])):
    sig = 0
    for p in Signal:
        sig += procs[p]
    
    bkg = 0
    for p in Background:
        bkg += procs[p]
    
"""
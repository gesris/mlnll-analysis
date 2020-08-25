import ROOT
import numpy as np

path = '/work/gristo/second_mlnll-analysis/output/8_bins_nosysimpl_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path)
d = f.Get('htt_mt_0_2018')

classes = ['W', 'ZTT', 'ZL', 'ZJ', 'TTT', 'TTL', 'TTJ', 'VVJ', 'VVT', 'VVL', 'ggH125', 'qqH125']

diff_hist = ROOT.TH1F("tot_jes_upshift", "", 8, 0, 1)

tot_jes_upshift = ROOT.TH1F("tot_jes_upshift", "", 8, 0, 1)
jes_upshift = []
for key in d.GetListOfKeys():
    name = key.GetName()
    if 'Up' in name:
        for class_name in classes:
            if class_name in name:
                h_nom = d.Get(class_name)
                h_shift = d.Get(name)
                # subtract hists to get shift only
                tot_jes_upshift.Add(h_nom, h_shift, -1, 1)

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
                tot_jes_downshift.Add(h_nom, h_shift, -1, 1)

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

print("UPSHIFT: {} \nSUM: {}".format(jes_upshift,np.sum(jes_upshift)))
print("DOWNSHIFT: {} \nSUM: {}".format(jes_downshift,np.sum(jes_downshift)))
print("SIG + BKG: {} \nSUM: {}".format(sig_bkg,np.sum(sig_bkg)))



diff_hist = ROOT.TH1F("DIFF", "", 8, 0, 1)
ztt_hist = d.Get('ZTT')
ztt_shift_hist = d.Get('ZTT_CMS_scale_j_FlavorQCDDown')
diff_hist.Add(ztt_hist, ztt_shift_hist, -1, 1)

ztt=[]
ztt_shift=[]
diff=[]
for i in range(1, 9):
    ztt.append(ztt_hist.GetBinContent(i))
    ztt_shift.append(ztt_shift_hist.GetBinContent(i))
    diff.append(diff_hist.GetBinContent(i))
print("ZTT: {}".format(ztt))
print("ZTT Shift: {}".format(ztt_shift))
print("DIFF: {}".format(diff))



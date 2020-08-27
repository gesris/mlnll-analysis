import ROOT
import numpy as np
from utils import config as cfg

path = '/work/gristo/second_mlnll-analysis/output/8_bins_nosysimpl_shapes/cmb/common/htt_input_2018.root'
f = ROOT.TFile(path, 'update')
d = f.Get('htt_mt_0_2018')
classes = ['W', 'ZTT', 'ZL', 'ZJ', 'TTT', 'TTL', 'TTJ', 'VVJ', 'VVT', 'VVL', 'ggH125', 'qqH125']


## Calculation total up- and downshift in root hist
diff_hist = ROOT.TH1F("DIFF", "", 8, 0, 1)
tot_jes_upshift = ROOT.TH1F("tot_jes_upshift", "", 8, 0, 1)
tot_jes_downshift = ROOT.TH1F("tot_jes_downshift", "", 8, 0, 1)
upshifts = {}
downshifts = {}
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
                    shift_array.append(h_shift.GetBinContent(i))
                    nom_array.append(h_nom.GetBinContent(i))
                upshifts[name] = np.square(np.array(shift_array) - np.array(nom_array))
    elif 'Down' in name:
        for class_name in classes:
            if class_name in name:
                h_nom = d.Get(class_name)
                h_shift = d.Get(name)
                # subtract hists to get shift only
                diff_hist.Add(h_nom, h_shift, 1, -1)
                tot_jes_downshift.Add(diff_hist)

                shift_array = []
                nom_array = []
                for i in range(1, 9):
                    shift_array.append(h_shift.GetBinContent(i))
                    nom_array.append(h_nom.GetBinContent(i))
                downshifts[name] = np.square(np.array(shift_array) - np.array(nom_array))

## Transferring root hists to np arrays
tot_upshifts = [0, 0, 0, 0, 0, 0, 0, 0]
for h in upshifts:
    tot_upshifts += upshifts[h]
tot_upshifts = np.sqrt(np.abs(np.array(tot_upshifts)))
print("UPSHIFT:   {}".format(tot_upshifts))

tot_downshifts = [0, 0, 0, 0, 0, 0, 0, 0]
for h in downshifts:
    tot_downshifts += downshifts[h]
tot_downshifts = np.sqrt(np.abs(np.array(tot_downshifts)))
print("DOWNSHIFT: {}".format(tot_downshifts))


## Writing new histograms with total up- and downshift
for key in d.GetListOfKeys():
    name = key.GetName()
    if name in classes:
        h = d.Get(name)

        ## Upshift
        newhup = h.Clone()
        newhup.SetTitle(name + "_scale_j_totUp")
        newhup.SetName(name + "_scale_j_totUp")

        ## Downshift
        newhdown = h.Clone()
        newhdown.SetTitle(name + "_scale_j_totDown")
        newhdown.SetName(name + "_scale_j_totDown")

        ## Fill Bincontent
        for i in range(1, 9):
            newhup.SetBinContent(i, tot_upshifts[i - 1])
            newhdown.SetBinContent(i, tot_downshifts[i - 1])

        ## Write content
        d.Write(newhup)
        d.Write(newhdown)


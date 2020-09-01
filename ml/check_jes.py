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


class_tot_upshifts = {}
class_tot_downshifts = {}

for class_name in classes:
    tot_upshift = [0, 0, 0, 0, 0, 0, 0, 0]
    for shift_name in upshifts:
        if class_name in shift_name:
            tot_upshift += upshifts[shift_name]
    tot_upshift[tot_upshift < 0] = 0
    tot_upshift = np.sqrt(np.array(tot_upshift))
    #class_tot_upshifts[class_name + '_scale_j_totUp'] = tot_upshift
    class_tot_upshifts[class_name] = tot_upshift
    
    tot_downshift = [0, 0, 0, 0, 0, 0, 0, 0]
    for shift_name in downshifts:
        if class_name in shift_name:
            tot_downshift += downshifts[shift_name]
    tot_downshift[tot_downshift < 0] = 0
    tot_downshift = np.sqrt(np.array(tot_downshift))
    #class_tot_downshifts[class_name + '_scale_j_totDown'] = tot_downshift
    class_tot_downshifts[class_name] = tot_downshift


tot_class_events = {"W": 127537, \
"ZTT": 237267, \
"QCD": 34166.3, \
"ZL": 9376.46, \
"ZJ": 5844.12, \
"TTT": 4032.03, \
"TTL": 60032.6, \
"TTJ": 11925.7, \
"VVJ": 3261.57, \
"VVT": 1322.81, \
"VVL": 12812.7, \
"ggH125": 2118.61, \
"qqH125": 269.456}

for name in class_tot_upshifts:
    print("{} SQRSMSQ: {}, TOT: {}, WEIGHT: {}".format(name, np.sqrt(np.sum(np.square(class_tot_upshifts[name]))), tot_class_events[name], (np.sqrt(np.sum(np.square(class_tot_upshifts[name]))) + tot_class_events[name]) / tot_class_events[name]))
print("---\n")
for name in class_tot_downshifts:
    print("{} SQSM: {}, TOT: {}".format(name, class_tot_downshifts[name], tot_class_events[name]))


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
            newhup.SetBinContent(i, h.GetBinContent(i) + class_tot_upshifts[name + "_scale_j_totUp"][i - 1])
            newhdown.SetBinContent(i, h.GetBinContent(i) - class_tot_downshifts[name + "_scale_j_totDown"][i - 1])
        
        ## Write content
        #d.cd()
        #newhup.Write()
        #newhdown.Write()

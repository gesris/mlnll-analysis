import ROOT
import numpy as np


def write_hists_names_yields():
    path = '/work/gristo/mlnll-analysis/output/32_bins_shapes/cmb/common/htt_input_2018.root'
    f = ROOT.TFile(path)
    d = f.Get('htt_mt_0_2018')
    nbins = 32

    yields = {}
    bincounts = {}

    hnames = []     # contains all names of root histograms
    hyields = []    # contains yields of root histograms
    hists = []      # contains python histograms of root histograms

    for key in d.GetListOfKeys():
        name = key.GetName()
        if 'ggH125' in name:
            h = d.Get(name)

            yields[name] = [h.Integral()]

            hnames.append(name)
            hyields.append(yields[name][0])
            bincounts[name] = [h.GetBinContent(i + 1) for i in range(nbins)]
            hists.append(bincounts[name])
    return hists, hnames, hyields


def diff_hists():
    nbins = 32
    hists, hnames, hyields = write_hists_names_yields()

    # first histogram is nominal ggH125, rest systematics
    nominal_ggH125 = np.array(hists[0])
    systematics_ggH125 = np.array(hists[1:])

    diff_sums = [0.000]
    for hist in systematics_ggH125:
        diff_sums.append(np.sum(np.abs(nominal_ggH125 - hist)))
    
    for i in range(len(hnames)):
        print("Name: {},        Yield: {:.3f}       DiffSum: {:.3f}".format(hnames[i], hyields[i], diff_sums[i]))

diff_hists()



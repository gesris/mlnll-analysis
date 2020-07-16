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

    diff_sums = [0.000]     # first entry = nominal
    for i, hist in enumerate(systematics_ggH125):
        # normalization of sys hists
        hist *= hyields[0] / hyields[i + 1]
        diff_sums.append(np.sum((nominal_ggH125 - hist)**2))
    
    for i in range(len(hnames)):
        print("Yield: {:.3f}       SquaredDiffSum: {:.3f}      Name: {}".format(hyields[i], diff_sums[i], hnames[i]))
    

#diff_hists()
array = []
path_ = '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root'
file_ = ROOT.TFile(path_)
tree_ = file_.Get("mt_nominal/ntuple")
for name in tree_:
   print(name)



tree_ = "ntuple"
column_ = 'THU_ggH_Mig01'



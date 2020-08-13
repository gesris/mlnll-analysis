import ROOT
import numpy as np


def write_hists_names_yields():
    path = '/home/gristo/mlnll-analysis/output/8_bins_bbb_shapes/cmb/common/htt_input_2018.root'
    f = ROOT.TFile(path)
    d = f.Get('htt_mt_0_2018')

    for key in d.GetListOfKeys():
        name = key.GetName()
        print("{}: {}".format(name, key.GetBinError(1)))
    #return hists, hnames,

write_hists_names_yields()



'''
# READING TREE ENTRIES OF SINGLE BRANCH FROM ROOT FILE

mig01 = [] # Mig01 systemattics
path_ = '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root'
file_ = ROOT.TFile(path_)
tree_ = file_.Get("mt_nominal/ntuple")
for i, event in enumerate(tree_):
    mig01.append(event.THU_ggH_Mig01)
'''





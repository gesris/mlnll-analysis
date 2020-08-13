import ROOT
import numpy as np


def write_hists_names_yields():
    path = '/home/gristo/mlnll-analysis/output/8_bins_bbb_shapes/cmb/common/htt_input_2018.root'
    f = ROOT.TFile(path)
    d = f.Get('htt_mt_0_2018')

    for key in d.GetListOfKeys():
        name = key.GetName()
        if name in ['TTL', 'TTJ']:
            ttt = d.Get('TTT')
            ttt.Add(d.Get(name))
        elif name in ['VVL', 'VVJ']:
            vv = d.Get('VVT')
            vv.Add(d.Get(name))
        elif name in ['ZJ']:
            zl = d.Get('ZL')
            zl.Add(d.Get(name))
    for key in d.GetListOfKeys():
        if name in ['W', 'ZTT', 'ZL', 'TTT', 'VVT', 'ggH125', 'qqH125']:
            h = d.Get(name)
            errors = []
            for i in range(1, 9):
                errors.append(h.GetBinError(i))
            errors = np.array(errors)
            np.set_printoptions(precision=3)
            print("{}: {}".format(name, errors))

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





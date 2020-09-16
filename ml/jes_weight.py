import ROOT
import numpy as np
from utils import config as cfg

# cfg.basepath + cfg.files = root file location

pfad = '/ceph/htautau/deeptau_02-20/2018/ntuples/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2/GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2.root'

for name in cfg.files:
    for path in cfg.files[name]:
        #f = ROOT.TFile(cfg.basepath + path + '/' + path + '.root', 'update')
        f = ROOT.TFile(pfad, 'update')
        #d = f.Get(path)
        #for key in d.GetListOfKeys():
        #    name = key.GetName()
        #    print(name)
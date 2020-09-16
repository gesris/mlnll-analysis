import ROOT
import numpy as np
from utils import config as cfg

# cfg.basepath + cfg.files = root file location

upshifts = {}

for name in cfg.files:
    for path in cfg.files[name]:
        f = ROOT.TFile(cfg.basepath + 'ntuples/' + path + '/' + path + '.root')
        #d = f.Get(path)
        for key in f.GetListOfKeys():
            name = key.GetName()
            if 'mt_jecUnc' in name:
                if 'Up' in name:
                    h_up = f.Get(name + '/jpt_1')
                    upshifts[name] = h_up
                elif 'Down' in name:
                    pass

print(upshifts)


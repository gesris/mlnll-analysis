import ROOT
import numpy as np
from utils import config as cfg

# cfg.basepath + cfg.files = root file location

for name in cfg.files:
    for path in cfg.files[name]:
        f = ROOT.TFile(cfg.basepath + path, 'update')
        d = f.Get()
        for key in d.GetListOfKeys():
            name = key.GetName()
            print(name)
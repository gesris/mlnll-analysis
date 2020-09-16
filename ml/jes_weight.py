import ROOT
import numpy as np
from utils import config as cfg

# cfg.basepath + cfg.files = root file location

for name in cfg.files:
    for path in cfg.files[name]:
        print(cfg.basepath + path)
import ROOT
import numpy as np
from utils import config as cfg

for name in cfg.files:
    print(name, cfg.files[name])
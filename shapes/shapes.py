from itertools import product
import numpy as np

from ntuple_processor import Histogram
from ntuple_processor import dataset_from_artusoutput
from ntuple_processor import Unit
from ntuple_processor import UnitManager
from ntuple_processor import GraphManager
from ntuple_processor import RunManager

import config as cfg

import logging
logger = logging.getLogger("")


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main():
    # Define histograms
    binning = np.linspace(0, 300, 30)
    hists = [Histogram('m_vis', 'm_vis', binning)]

    # Define nominal units
    units = {}
    units['data'] = Unit(
            dataset_from_artusoutput('data', cfg.files['singlemuon'], 'mt_nominal', cfg.ntuples_base, cfg.friends_base),
            [cfg.channel], hists)
    units['w'] = Unit(
            dataset_from_artusoutput('w', cfg.files['wjets'], 'mt_nominal', cfg.ntuples_base, cfg.friends_base),
            [cfg.channel, cfg.w], hists)
    dy_dataset = dataset_from_artusoutput('dy', cfg.files['dy'], 'mt_nominal', cfg.ntuples_base, cfg.friends_base)
    units['ztt'] = Unit(dy_dataset,[cfg.channel, cfg.dy, cfg.ztt], hists)

    # Book units with variations
    um = UnitManager()
    um.book([units[name] for name in ['data', 'w', 'ztt']])

    # Optimize graphs
    g_manager = GraphManager(um.booked_units)
    g_manager.optimize(2)
    graphs = g_manager.graphs

    # Run computations
    r_manager = RunManager(graphs)
    r_manager.run_locally('shapes.root', nworkers=12, nthreads=1)


if __name__ == "__main__":
    setup_logging('shapes.log', logging.INFO)
    main()

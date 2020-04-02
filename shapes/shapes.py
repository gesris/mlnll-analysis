from itertools import product
import argparse
import numpy as np
import os

from ntuple_processor import dataset_from_artusoutput
from ntuple_processor import Unit
from ntuple_processor import UnitManager
from ntuple_processor import GraphManager
from ntuple_processor import RunManager
from ntuple_processor import Histogram

import utils.config as cfg

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


def main(args):
    # Define histograms
    hists = [Histogram(var, var, cfg.binning[var]) for var in cfg.variables]

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
    units['zl'] = Unit(dy_dataset,[cfg.channel, cfg.dy, cfg.zl], hists)
    units['zj'] = Unit(dy_dataset,[cfg.channel, cfg.dy, cfg.zj], hists)

    tt_dataset = dataset_from_artusoutput('tt', cfg.files['tt'], 'mt_nominal', cfg.ntuples_base, cfg.friends_base)
    units['ttt'] = Unit(tt_dataset,[cfg.channel, cfg.tt, cfg.ttt], hists)
    units['ttl'] = Unit(tt_dataset,[cfg.channel, cfg.tt, cfg.ttl], hists)
    units['ttj'] = Unit(tt_dataset,[cfg.channel, cfg.tt, cfg.ttj], hists)


    vv_dataset = dataset_from_artusoutput('vv', cfg.files['vv'], 'mt_nominal', cfg.ntuples_base, cfg.friends_base)
    units['vvt'] = Unit(vv_dataset,[cfg.channel, cfg.vv, cfg.vvt], hists)
    units['vvl'] = Unit(vv_dataset,[cfg.channel, cfg.vv, cfg.vvl], hists)
    units['vvj'] = Unit(vv_dataset,[cfg.channel, cfg.vv, cfg.vvj], hists)

    units['ggh'] = Unit(
            dataset_from_artusoutput('ggh', cfg.files['ggh'], 'mt_nominal', cfg.ntuples_base, cfg.friends_base),
            [cfg.channel, cfg.htt, cfg.ggh], hists)

    units['qqh'] = Unit(
            dataset_from_artusoutput('qqh', cfg.files['qqh'], 'mt_nominal', cfg.ntuples_base, cfg.friends_base),
            [cfg.channel, cfg.htt, cfg.qqh], hists)

    # Book units with variations
    um = UnitManager()
    um.book([units[name] for name in ['data', 'w', 'ztt', 'zl', 'zj', 'ttt', 'ttl', 'ttj', 'vvt', 'vvl', 'vvj', 'ggh', 'qqh']])
    um.book([units[name] for name in ['data', 'zl', 'zj', 'w', 'ttt', 'ttj', 'ttl', 'vvt', 'vvj', 'vvl']], [cfg.same_sign])

    # Optimize graphs
    g_manager = GraphManager(um.booked_units)
    g_manager.optimize(2)
    graphs = g_manager.graphs

    # Run computations
    r_manager = RunManager(graphs)
    r_manager.run_locally(os.path.join(args.workdir, 'shapes_main.root'), nworkers=8, nthreads=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="Working directory for outputs")
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'shapes.log'), logging.INFO)
    main(args)

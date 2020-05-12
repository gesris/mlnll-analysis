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
logger = logging.getLogger('')


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main(args):
    # Define nominal units
    def define_units(name, category_selections, hists):
        units = {}
        friends = cfg.friends_base + cfg.ml_score_base

        units['data'] = Unit(
                dataset_from_artusoutput('data', cfg.files['singlemuon'], 'mt_nominal', cfg.ntuples_base, friends),
                [cfg.channel] + category_selections, hists)

        units['w'] = Unit(
                dataset_from_artusoutput('w', cfg.files['wjets'], 'mt_nominal', cfg.ntuples_base, friends),
                [cfg.channel, cfg.mc, cfg.w] + category_selections, hists)

        dy_dataset = dataset_from_artusoutput('dy', cfg.files['dy'], 'mt_nominal', cfg.ntuples_base, friends)
        units['ztt'] = Unit(dy_dataset, [cfg.channel, cfg.mc, cfg.dy, cfg.ztt] + category_selections, hists)
        units['zl'] = Unit(dy_dataset, [cfg.channel, cfg.mc, cfg.dy, cfg.zl] + category_selections, hists)
        units['zj'] = Unit(dy_dataset, [cfg.channel, cfg.mc, cfg.dy, cfg.zj] + category_selections, hists)

        tt_dataset = dataset_from_artusoutput('tt', cfg.files['tt'], 'mt_nominal', cfg.ntuples_base, friends)
        units['ttt'] = Unit(tt_dataset, [cfg.channel, cfg.mc, cfg.tt, cfg.ttt] + category_selections, hists)
        units['ttl'] = Unit(tt_dataset, [cfg.channel, cfg.mc, cfg.tt, cfg.ttl] + category_selections, hists)
        units['ttj'] = Unit(tt_dataset, [cfg.channel, cfg.mc, cfg.tt, cfg.ttj] + category_selections, hists)


        vv_dataset = dataset_from_artusoutput('vv', cfg.files['vv'], 'mt_nominal', cfg.ntuples_base, friends)
        units['vvt'] = Unit(vv_dataset, [cfg.channel, cfg.mc, cfg.vv, cfg.vvt] + category_selections, hists)
        units['vvl'] = Unit(vv_dataset, [cfg.channel, cfg.mc, cfg.vv, cfg.vvl] + category_selections, hists)
        units['vvj'] = Unit(vv_dataset, [cfg.channel, cfg.mc, cfg.vv, cfg.vvj] + category_selections, hists)

        units['ggh'] = Unit(
                dataset_from_artusoutput('ggh', cfg.files['ggh'], 'mt_nominal', cfg.ntuples_base, friends),
                [cfg.channel, cfg.mc, cfg.htt, cfg.ggh] + category_selections, hists)

        units['qqh'] = Unit(
                dataset_from_artusoutput('qqh', cfg.files['qqh'], 'mt_nominal', cfg.ntuples_base, friends),
                [cfg.channel, cfg.mc, cfg.htt, cfg.qqh] + category_selections, hists)

        return units

    # Add control units with inclusive selection
    units = {}
    units['inclusive'] = define_units('inclusive', [],
            [Histogram(var + '_inclusive', var, cfg.binning[var]) for var in cfg.control_variables])

    # Add units of the analysis categories
    for name in cfg.analysis_categories:
        units[name] = define_units(name, [cfg.analysis_categories[name]],
                [Histogram(cfg.analysis_variable + '_' + name, cfg.analysis_variable, cfg.analysis_binning)])

    # Book nominal units
    um = UnitManager()
    categories = list(units.keys())
    um.book([units[category][name] for name, category in product(
        ['data', 'w', 'ztt', 'zl', 'zj', 'ttt', 'ttl', 'ttj', 'vvt', 'vvl', 'vvj', 'ggh', 'qqh'], categories)])

    # Same-sign region for the QCD estimation (inclusive and analysis categories)
    um.book([units[category][name] for name, category in product(
        ['data', 'w', 'ztt', 'zl', 'zj', 'ttt', 'ttl', 'ttj', 'vvt', 'vvl', 'vvj'], categories)],
        [cfg.same_sign])

    # ggH uncertainties
    um.book([units[category][name] for name, category in product(['ggh'], cfg.analysis_categories)], [*cfg.ggh_wg1])

    # Optimize graphs
    g_manager = GraphManager(um.booked_units)
    g_manager.optimize(2)
    graphs = g_manager.graphs

    # Run computations
    r_manager = RunManager(graphs)
    r_manager.run_locally(os.path.join(args.workdir, 'shapes_main.root'), nworkers=8, nthreads=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'shapes.log'), logging.INFO)
    main(args)

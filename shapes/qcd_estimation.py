import os
import argparse

import ROOT
from utils import Reader
from utils.config import control_variables, analysis_variable, analysis_categories

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
    r = Reader([os.path.join(args.workdir, 'shapes_main.root')])
    f = ROOT.TFile(os.path.join(args.workdir, 'shapes_qcd.root'), 'RECREATE')
    for variable in [v + '_inclusive' for v in control_variables] + \
                    [analysis_variable + '_' + c for c in analysis_categories]:
        logger.debug('Produce qcd shape for variable %s', variable)
        data = r.get('data', 'same_sign', variable)
        for process in ['w', 'ztt', 'zl', 'zj', 'ttt', 'ttl', 'ttj', 'vvt', 'vvl', 'vvj']:
            h = r.get(process, 'same_sign', variable)
            data.Add(h, -1)

        name = str(data.GetName()).replace('data', 'qcd').replace('same_sign', 'Nominal')
        data.SetNameTitle(name, name)
        data.Write()
    f.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'qcd_estimation.log'), logging.INFO)
    main(args)

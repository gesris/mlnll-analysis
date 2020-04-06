import os
import argparse

import ROOT
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


process_map = {
        'data': 'data_obs',
        'ztt': 'ZTT',
        'zl': 'ZL',
        'zj': 'ZJ',
        'ttt': 'TTT',
        'ttl': 'TTL',
        'ttj': 'TTJ',
        'vvt': 'VVT',
        'vvl': 'VVL',
        'vvj': 'VVJ',
        'w': 'W',
        'ggh': 'ggH125',
        'qqh': 'qqH125',
        }


def main(args):
    outfile = ROOT.TFile(os.path.join(args.workdir, 'shapes_ch.root'), 'RECREATE')

    # Make folder structure
    outfile.mkdir('2018')
    for category in analysis_categories:
        outfile.mkdir('2018/mt_' + category)

    # Convert shapes
    for filename in ['shapes_main.root']:
        f = ROOT.TFile(os.path.join(args.workdir, filename), 'READ')
        for category in analysis_categories:
            # Get folder of this category
            outdirname = '2018/mt_' + category
            outdir = outfile.Get(outdirname)
            # Find shapes related to this category, convert and write to output file
            for key in f.GetListOfKeys():
                # Get properties
                name = key.GetName()
                props = name.split('#')
                dataset = props[0]
                selections = props[1]
                variation = props[2]
                # Select category shapes
                if not category in selections:
                    logger.debug('Dropped {} because shape does not belong to category {}'.format(name, category))
                    continue
                # Remove same sign shapes
                if 'same_sign' == variation:
                    logger.debug('Dropped {} because variation is same_sign'.format(name))
                    continue
                # Find process
                process = None
                if dataset == 'data':
                    process = 'data'
                else:
                    for p in process_map:
                        if '-' + p + '-' in selections:
                            process = p
                            break
                if not process:
                    logger.debug('Dropped {} because no process could be associated'.format(name))
                    break
                # Write histogram
                h = f.Get(name)
                tag = ''
                if variation != "Nominal":
                    tag = '_' + variation
                outname = process_map[process] + tag
                logger.debug('Write histogram {} as {} to {}'.format(name, outname, outdirname))
                outdir.WriteTObject(h, outname)
        f.Close()
    outfile.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'convert2ch.log'), logging.INFO)
    main(args)

import os
import argparse

import ROOT
ROOT.DisableImplicitMT()
from utils import config as cfg

import logging
logger = logging.getLogger('')


# Global lsit of chains to keep everything alive
chains = []


def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def collect_cuts_weights(selections):
    weights = []
    cuts = []
    for s in selections:
        if hasattr(s, 'weights'):
            t = []
            for w in s.weights:
                t += [w.expression]
            weights += t
        if hasattr(s, 'cuts'):
            t = []
            for c in s.cuts:
                t += [c.expression]
            cuts += t
    cutstr = '&&'.join(['({})'.format(c) for c in cuts])
    weightstr = '*'.join(['({})'.format(w) for w in weights])
    if cutstr == '':
        cutstr = 'true'
    if weightstr == '':
        weightstr = '1'
    return cutstr, weightstr


def make_chain(files, basepath, foldername):
    c = ROOT.TChain('{}/ntuple'.format(foldername))
    for f in files:
        path = os.path.join(basepath, f, f + '.root')
        if not os.path.exists(path):
            logger.fatal('File %s does not exist', path)
            raise Exception
        c.AddFile(path)
    chains.append(c)
    return c


def make_dataset(files, ntuples_base, friends_base, foldername):
    n = make_chain(files, ntuples_base, foldername)
    for f in friends_base:
        c = make_chain(files, f, foldername)
        n.AddFriend(c)
    return n


def write_dataset(d, workdir, name, group, fold, weightstr, cutstr):
    df = ROOT.RDataFrame(d)
    variables = ROOT.std.vector(ROOT.std.string)()
    for v in cfg.ml_variables:
        variables.push_back(v)
    variables.push_back(cfg.ml_weight)
    df.Filter('event % 2 == {}'.format(fold))\
      .Filter(cutstr)\
      .Define(cfg.ml_weight, weightstr)\
      .Snapshot(group, os.path.join(workdir, '{}_fold{}.root'.format(name, fold)), variables)


def write_dataset_jpt_1(d, workdir, name, group, fold, weightstr, cutstr):
    df = ROOT.RDataFrame(d)
    variables = ROOT.std.vector(ROOT.std.string)()
    #for v in cfg.ml_variables:
    #    variables.push_back(v)
    #variables.push_back(cfg.ml_weight)
    variables.push_back("jpt_1_weights_up")
    variables.push_back("jpt_1_weights_down")
    df.Filter('event % 2 == {}'.format(fold))\
      .Filter(cutstr)\
      .Define(cfg.ml_weight, weightstr)\
      .Snapshot(group, os.path.join(workdir, '{}_fold{}.root'.format(name, fold)), variables)


def ggh():
    return cfg.files['ggh'], [cfg.channel, cfg.mc, cfg.htt, cfg.ggh], 'ggh', 'ggh'

def qqh():
    return cfg.files['qqh'], [cfg.channel, cfg.mc, cfg.htt, cfg.qqh], 'qqh', 'qqh'

def ztt():
    return cfg.files['dy'], [cfg.channel, cfg.mc, cfg.dy, cfg.ztt], 'ztt', 'ztt'

def zj():
    return cfg.files['dy'], [cfg.channel, cfg.mc, cfg.dy, cfg.zj], 'zj', 'zl'

def zl():
    return cfg.files['dy'], [cfg.channel, cfg.mc, cfg.dy, cfg.zl], 'zl', 'zl'

def w():
    return cfg.files['wjets'], [cfg.channel, cfg.mc, cfg.w], 'w', 'w'

def ttt():
    return cfg.files['tt'], [cfg.channel, cfg.mc, cfg.tt, cfg.ttt], 'ttt', 'tt'

def ttl():
    return cfg.files['tt'], [cfg.channel, cfg.mc, cfg.tt, cfg.ttl], 'ttl', 'tt'

def ttj():
    return cfg.files['tt'], [cfg.channel, cfg.mc, cfg.tt, cfg.ttj], 'ttj', 'tt'

def vvt():
    return cfg.files['vv'], [cfg.channel, cfg.mc, cfg.vv, cfg.vvt], 'vvt', 'vv'

def vvl():
    return cfg.files['vv'], [cfg.channel, cfg.mc, cfg.vv, cfg.vvl], 'vvl', 'vv'

def vvj():
    return cfg.files['vv'], [cfg.channel, cfg.mc, cfg.vv, cfg.vvj], 'vvj', 'vv'

def data():
    return cfg.files['singlemuon'], [cfg.channel], 'data', 'data'


def main(args):
    ROOT.EnableImplicitMT(args.nthreads)
    
    # Collect nominal events
    for process in [ggh, qqh, ztt, zl, zj, w, ttt, ttl, ttj, vvt, vvl, vvj]:
        files, selections, name, group = process()
        cutstr, weightstr = collect_cuts_weights(selections)
        d = make_dataset(files, cfg.ntuples_base, cfg.friends_base, 'mt_nominal')
        logger.info('Create dataset for %s with label %s, group %s and %u events', process, name, group, d.GetEntries())
        logger.debug('Weight string: %s', weightstr)
        logger.debug('Cut string: %s', cutstr)
        for fold in [0, 1]:
            write_dataset(d, args.workdir, name, group, fold, weightstr, cutstr)

    # Collect events for QCD estimation
    for process in [data, ztt, zl, zj, w, ttt, ttl, ttj, vvt, vvl, vvj]:
        files, selections, name, group = process()
        cutstr, weightstr = collect_cuts_weights(selections)
        ssos = ('q_1*q_2<0')
        if not ssos in cutstr:
            logger.fatal('Cannot find SS/OS substring in cutstring')
            raise Exception
        cutstr_ss = cutstr.replace(ssos, 'q_1*q_2>0')
        d = make_dataset(files, cfg.ntuples_base, cfg.friends_base, 'mt_nominal')
        for fold in [0, 1]:
            write_dataset(d, args.workdir, name + '_ss', group + '_ss', fold, weightstr, cutstr_ss)
    
    ## Collect JES systematic shift
    for process in [ggh, qqh, ztt, zl, zj, w, ttt, ttl, ttj, vvt, vvl, vvj]:
        files, selections, name, group = process()
        cutstr, weightstr = collect_cuts_weights(selections)
        d = make_dataset(files, cfg.ntuples_jpt_1_base, cfg.friends_base, 'mt_nominal')
        logger.info('Create dataset for %s with label %s, group %s and %u events', process, name, group, d.GetEntries())
        logger.debug('Weight string: %s', weightstr)
        logger.debug('Cut string: %s', cutstr)
        for fold in [0, 1]:
            write_dataset_jpt_1(d, args.workdir, name, group, fold, weightstr, cutstr)


    # Collect systematic shifts
    '''
    for sys in ['jecUncRelativeSampleYearUp', 'jecUncRelativeSampleYearDown']:
        for process in [ggh, qqh]:
            files, selections, name, group = process()
            cutstr, weightstr = collect_cuts_weights(selections)
            d = make_dataset(files, cfg.ntuples_base, cfg.friends_base, 'mt_' + sys)
            for fold in [0, 1]:
                write_dataset(d, args.workdir, name + '_' + sys, group + '_' + sys, fold, weightstr, cutstr)
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('--nthreads', default=12, help='Number of threads')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_dataset.log'), logging.INFO)
    main(args)

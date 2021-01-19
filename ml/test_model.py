import os
import argparse
import pickle

#from tqdm import tqdm
import numpy as np
np.random.seed(1234)
from scipy import interpolate
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1234)
import tensorflow_probability as tfp

from utils import config as cfg
from train import build_dataset, model

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc("font", size=16, family="serif")

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

@tf.custom_gradient
def count_masking(x, up, down):
    mask = tf.cast(
            tf.cast(x > down, tf.float64) * tf.cast(x <= up, tf.float64),
            tf.float64)
    mask = tf.squeeze(mask)

    def grad(dy):
        width = up - down
        mid = down + 0.5 * width
        sigma = 0.5 * width
        gauss = tf.exp(-1.0 * (x - mid)**2 / 2.0 / sigma**2)
        g = -1.0 * gauss * (x - mid) / sigma**2
        g = tf.squeeze(g) * tf.squeeze(dy)
        return (g, None, None)

    return mask, grad


def main(args):
    ## Load dataset
    classes = cfg.ml_classes + [n + '_ss' for n in cfg.ml_classes if n not in ['ggh', 'qqh']] + ['data_ss']
    x, y, w, m_vis_upshift, m_vis_downshift, met_upshift, met_downshift = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)), classes, args.fold,
                            use_class_weights=False, make_categorical=False)
    fold_factor = 2.
    w = w * fold_factor
    preproc = pickle.load(open(os.path.join(args.workdir, 'preproc_fold{}.pickle'.format(args.fold)), 'rb'))
    x_preproc = preproc.transform(x)


    ## Load model
    x_ph = tf.placeholder(tf.float64, shape=(None,len(cfg.ml_variables)))
    _, f, _ = model(x_ph, len(cfg.ml_variables), 1, args.fold)
    path = tf.train.latest_checkpoint(os.path.join(args.workdir, 'model_fold{}'.format(args.fold)))
    logger.debug('Load model {}'.format(path))
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    
    y_ph = tf.placeholder(tf.float64, shape=(None,))
    w_ph = tf.placeholder(tf.float64, shape=(None,))
    # njets_upshift_ph = tf.placeholder(tf.float64)
    # njets_downshift_ph = tf.placeholder(tf.float64)
    scale_ph = tf.placeholder(tf.float64)

    bins = np.array(cfg.analysis_binning)
    bins_center = []
    for left, right in zip(bins[1:], bins[:-1]):
        bins_center.append(left + (right - left) / 2)
    

    ## Get bincontent   
    bincontent_nom = {}
    bincontent_up = {}
    bincontent_down = {}


    for i, (up, down) in enumerate(zip(bins[1:], bins[:-1])):
        counts_nom = {}
        counts_up = {}
        counts_down = {}
        logger.debug('Add NLL for bin {} with boundaries [{}, {}]'.format(i, down, up))
        up = tf.constant(up, tf.float64)
        down = tf.constant(down, tf.float64)

        # Processes
        mask = count_masking(f, up, down)
        procs = {}
        procs_up = {}
        procs_down = {}

        for j, name in enumerate(classes):
            proc_w = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph
            # proc_w_up = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph * njets_upshift_ph
            # proc_w_down = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph * njets_downshift_ph
            procs[name] = tf.reduce_sum(proc_w)
            procs_up[name] = tf.reduce_sum(proc_w) * 1.2
            procs_down[name] = tf.reduce_sum(proc_w) * 0.8

        # QCD estimation
        procs['qcd'] = procs['data_ss']
        for p in [n for n in cfg.ml_classes if not n in ['ggh', 'qqh']]:
            procs['qcd'] -= procs[p + '_ss']
        procs['qcd'] = tf.maximum(procs['qcd'], 0)
        # procs_up['qcd'] = tf.maximum(procs['qcd'] * njets_upshift_ph, 0)
        # procs_down['qcd'] = tf.maximum(procs['qcd'] * njets_downshift_ph, 0)

        # Nominal signal and background
        for p in ['ggh', 'qqh', 'ztt', 'zl', 'w', 'tt', 'vv', 'qcd']:
            counts_nom[p] = procs[p]
        
        # Shifted signal and background
        for p in ['ztt']:
            counts_up[p] = procs_up[p]
            counts_down[p] = procs_down[p]

        # Add total content to nested dictionary
        bincontent_nom[i] = counts_nom
        bincontent_up[i] = counts_up
        bincontent_down[i] = counts_down

    ## Calculating bbb-unc. and bincontent for histogram
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(session, path)

    bincontent_nom_, bincontent_up_, bincontent_down_ = session.run([bincontent_nom, bincontent_up, bincontent_down], \
                        feed_dict={x_ph: x_preproc, y_ph: y, w_ph: w, scale_ph: fold_factor})


    ## Plotting histogram
    def plot(bincontent_nom, bincontent_up, bincontent_down, bins, bins_center):
        plt.figure(figsize=(7, 6))
        color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for i, element in enumerate(['ggh', 'qqh', 'ztt']):
            content = []
            for id, classes in bincontent_nom.items():
                content.append(classes[element])
            plt.hist(bins_center, weights=content, bins=bins, histtype="step", lw=2, color=color[i])
            plt.plot([0], [0], lw=2, color=color[i], label=element)
            if element in ['ztt']:
                content_up = []
                for id, classes_up in bincontent_up.items():
                    content_up.append(classes_up[element])
                plt.hist(bins_center, weights=content_up, bins=bins, histtype="step", lw=2, ls=':', color=color[i])
                plt.plot([0], [0], lw=2, ls=':', color=color[i], label=element + ' up')
                content_down = []
                for id, classes_down in bincontent_down.items():
                    content_down.append(classes_down[element])
                plt.hist(bins_center, weights=content_down, bins=bins, histtype="step", lw=2, ls='--', color=color[i])
                plt.plot([0], [0], lw=2, ls='--', color=color[i], label=element + ' down')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., prop={'size': 14})
        plt.xlabel("$f$")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.savefig(os.path.join(args.workdir, 'model_fold{}/histogram{}.png'.format(args.fold, args.fold)), bbox_inches = "tight")
        logger.info("Saving histogram in {}/model_fold{}".format(args.workdir, args.fold))
    plot(bincontent_nom_, bincontent_up_, bincontent_down_, bins, bins_center)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_test_fold{}.log'.format(args.fold)), logging.INFO)
    main(args)
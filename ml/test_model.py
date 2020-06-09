import os
import argparse
import pickle

import ROOT
import numpy as np
np.random.seed(1234)
from sklearn.metrics import confusion_matrix
from utils import config as cfg

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1234)

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


def plot(signal, background, category, bins, bins_center):
    plt.figure(figsize=(7, 6))
    plt.hist(bins_center, weights= signal, bins= bins, histtype="step", lw=2, color="C0")
    plt.hist(bins_center, weights= background[0], bins= bins, histtype="step", lw=2, color="C1")
    plt.hist(bins_center, weights= background[1], bins= bins, histtype="step", lw=2, color="C2")
    plt.hist(bins_center, weights= background[2], bins= bins, histtype="step", lw=2, color="C3")
    plt.plot([0], [0], lw=2, color="C0", label="Htt")
    plt.plot([0], [0], lw=2, color="C1", label="Ztt")
    plt.plot([0], [0], lw=2, color="C2", label="W")
    plt.plot([0], [0], lw=2, color="C3", label="ttbar")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.xlabel("$f$")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.savefig("./histogram.png", bbox_inches = "tight")


@tf.custom_gradient
def count_masking(x, up, down):
    mask = tf.cast(
            tf.cast(x > down, tf.float32) * tf.cast(x <= up, tf.float32),
            tf.float32)
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
    inv_fold = [1, 0][args.fold]
    x, y, w = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(inv_fold)), cfg.ml_classes, inv_fold,
                            make_categorical=False, use_class_weights=False)

    preproc = pickle.load(open(os.path.join(args.workdir, 'preproc_fold{}.pickle'.format(args.fold)), 'rb'))
    x_preproc = preproc.transform(x)

    ####
    #### Prepare masking
    ####

    y_array = np.array(y)

    Htt_mask_feed = np.where(y_array == 0, 1, 0)
    Ztt_mask_feed = np.where(y_array == 1, 1, 0)
    W_mask_feed = np.where(y_array == 2, 1, 0)
    ttbar_mask_feed = np.where(y_array == 3, 1, 0)

    x_ph = tf.placeholder(tf.float32)
    w_ph = tf.placeholder(tf.float32)
    Htt_mask = tf.placeholder(tf.float32)
    Ztt_mask = tf.placeholder(tf.float32)
    W_mask = tf.placeholder(tf.float32)
    ttbar_mask = tf.placeholder(tf.float32)

    
    ####
    #### Load model
    ####

    _, f = model(x_ph, len(cfg.ml_variables), args.fold)
    path = tf.train.latest_checkpoint(os.path.join(args.workdir, 'model_fold{}'.format(args.fold)))
    logger.debug('Load model {}'.format(path))
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    
    bins = cfg.analysis_binning
    upper_edges, lower_edges = bins[1:], bins[:-1]
    bins_center = []
    for i in range(0, len(bins) - 1):
        bins_center.append(bins[i] + (bins[i + 1] - bins[i]) / 2)
    background_category = ['Ztt', 'W', 'ttbar']
    Htt = []
    Ztt = []
    W = []
    ttbar = []

    for i, up, down in zip(range(len(upper_edges)), upper_edges, lower_edges):
        # Bin edges
        up_ = tf.constant(up, tf.float32)
        down_ = tf.constant(down, tf.float32)
        
        Htt.append(tf.reduce_sum(count_masking(f, up_, down_) * Htt_mask * w_ph))
        Ztt.append(tf.reduce_sum(count_masking(f, up_, down_) * Ztt_mask * w_ph))
        W.append(tf.reduce_sum(count_masking(f, up_, down_) * W_mask * w_ph))  
        ttbar.append(tf.reduce_sum(count_masking(f, up_, down_) * ttbar_mask * w_ph))
    
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(session, path)

    Htt_counts, Ztt_counts, W_counts, ttbar_counts = session.run([Htt, Ztt, W, ttbar], \
                        feed_dict={x_ph: x_preproc, w_ph: w, \
                                    Htt_mask: Htt_mask_feed, \
                                    Ztt_mask: Ztt_mask_feed, \
                                    W_mask: W_mask_feed, \
                                    ttbar_mask: ttbar_mask_feed})

    logger.info("\n\nHtt Counts: {}".format(Htt_counts))
    logger.info("Ztt Counts: {}".format(Ztt_counts))
    logger.info("W Counts: {}".format(W_counts))
    logger.info("ttbar Counts: {}\n\n".format(ttbar_counts))

    plot(Htt_counts, [Ztt_counts, W_counts, ttbar_counts], background_category, bins, bins_center)

    # histogramme mit matplotlib machen!
    # abbrechen lassen, wenn counts mit der aussage von TF nicht übereinstimmen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_test_fold{}.log'.format(args.fold)), logging.INFO)
    main(args)

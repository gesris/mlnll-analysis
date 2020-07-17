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
    plt.hist(bins_center, weights= background[3], bins= bins, histtype="step", lw=2, ls=':', color="C0")
    plt.hist(bins_center, weights= background[4], bins= bins, histtype="step", lw=2, ls='--', color="C0")
    plt.plot([0], [0], lw=2, color="C0", label="Htt")
    plt.plot([0], [0], lw=2, color="C1", label="Ztt")
    plt.plot([0], [0], lw=2, color="C2", label="W")
    plt.plot([0], [0], lw=2, color="C3", label="ttbar")
    plt.plot([0], [0], lw=2, ls=':', color="C0", label="Htt Up")
    plt.plot([0], [0], lw=2, ls='--', color="C0", label="Htt Down")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.xlabel("$f$")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.savefig(os.path.join(args.workdir, 'model_fold{}/histogram.png'.format(args.fold)), bbox_inches = "tight")


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
    x, y, w, mig01 = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(inv_fold)), cfg.ml_classes, inv_fold,
                            make_categorical=True, use_class_weights=False)
    
    # Magnify Mig01 to have a bigger impact on training
    mean_value = np.mean(mig01)
    mig01 = (mig01 - mean_value) * 10 + mean_value

    # Process Mig01 to have same number of entries as other variables
    mig01 = np.append(mig01, np.ones(len(w) - len(mig01)))
    
    preproc = pickle.load(open(os.path.join(args.workdir, 'preproc_fold{}.pickle'.format(args.fold)), 'rb'))
    x_preproc = preproc.transform(x)

    ####
    #### Prepare masking
    ####

    y_array = np.array(y)

    fold_factor = 2.
    logger.info("\n\nHTT SUMWEIGHTS: {}".format(fold_factor * np.sum(w[y_array[:, 0] == 1])))
    logger.info("\n\nZTT SUMWEIGHTS: {}".format(fold_factor * np.sum(w[y_array[:, 1] == 1])))
    logger.info("\n\nW SUMWEIGHTS: {}".format(fold_factor * np.sum(w[y_array[:, 2] == 1])))
    logger.info("\n\nTTBAR SUMWEIGHTS: {}".format(fold_factor * np.sum(w[y_array[:, 3] == 1])))

    
    # only possible, wher make_categorical=False
    #Htt_mask_feed = np.where(y_array == 0, 1, 0)
    #Ztt_mask_feed = np.where(y_array == 1, 1, 0)
    #W_mask_feed = np.where(y_array == 2, 1, 0)
    #ttbar_mask_feed = np.where(y_array == 3, 1, 0)

    # oly possible, when make_categorical=True
    Htt_mask_feed = y_array[:, 0]
    Ztt_mask_feed = y_array[:, 1]
    W_mask_feed = y_array[:, 2]
    ttbar_mask_feed = y_array[:, 3]

    x_ph = tf.placeholder(tf.float32)
    w_ph = tf.placeholder(tf.float32)
    mig01_ph = tf.placeholder(tf.float32)
    fold_scale = tf.placeholder(tf.float32)
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
    background_category = ['Ztt', 'W', 'ttbar', 'Htt Up', 'Htt Down']
    Htt = []
    Htt_up = []
    Htt_down = []
    Ztt = []
    W = []
    ttbar = []
    
    ttbar_labels = []
    ttbar_weights = []
    ttbar_events_noweights = []

    for i, up, down in zip(range(len(upper_edges)), upper_edges, lower_edges):
        # Bin edges
        up_ = tf.constant(up, tf.float32)
        down_ = tf.constant(down, tf.float32)
        
        Htt.append(tf.reduce_sum(count_masking(f, up_, down_) * Htt_mask * w_ph * fold_scale))
        Htt_up.append(tf.reduce_sum(count_masking(f, up_, down_) * Htt_mask * w_ph * fold_scale * mig01_ph))
        Htt_down.append(tf.reduce_sum(count_masking(f, up_, down_) * Htt_mask * w_ph * fold_scale / mig01_ph))
        Ztt.append(tf.reduce_sum(count_masking(f, up_, down_) * Ztt_mask * w_ph * fold_scale))
        W.append(tf.reduce_sum(count_masking(f, up_, down_) * W_mask * w_ph * fold_scale))  
        ttbar.append(tf.reduce_sum(count_masking(f, up_, down_) * ttbar_mask * w_ph * fold_scale))

        ttbar_events_noweights.append(tf.reduce_sum(count_masking(f, up_, down_)))
        
    ttbar_labels.append(ttbar_mask)
    ttbar_weights.append(ttbar_mask * w_ph)
    
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(session, path)
    
    Htt_counts, Ztt_counts, W_counts, ttbar_counts, Htt_up_counts, Htt_down_counts = session.run([Htt, Ztt, W, ttbar, Htt_up, Htt_down], \
                        feed_dict={x_ph: x_preproc, w_ph: w, mig01_ph: mig01, \
                                    Htt_mask: Htt_mask_feed, \
                                    Ztt_mask: Ztt_mask_feed, \
                                    W_mask: W_mask_feed, \
                                    ttbar_mask: ttbar_mask_feed,\
                                    fold_scale: 2.})
    # Printing out Histograms
    logger.info('\nHtt Total:      {:.3f}\
                \nHtt Up Total:   {:.3f}\
                \nHtt Down Total: {:.3f}\
                \nZtt Total:      {:.3f}\
                \nW Total:        {:.3f}\
                \nttbar Total:    {:.3f}\n'.format(np.sum(Htt_counts), np.sum(Htt_up_counts), np.sum(Htt_down_counts), np.sum(Ztt_counts), np.sum(W_counts), np.sum(ttbar_counts)))

    ### save counts into csv file for scan_cross_check.py
    # first empty existing file
    open(os.path.join(args.workdir, 'model_fold{}/hists.csv'.format(args.fold)), "w").close()
    with open(os.path.join(args.workdir, 'model_fold{}/hists.csv'.format(args.fold)), "ab") as file:
        np.savetxt(file, [Htt_counts])
        np.savetxt(file, [Ztt_counts])
        np.savetxt(file, [W_counts])
        np.savetxt(file, [ttbar_counts])

    plot(Htt_counts, [Ztt_counts, W_counts, ttbar_counts, Htt_up_counts, Htt_down_counts], background_category, bins, bins_center)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_test_fold{}.log'.format(args.fold)), logging.INFO)
    main(args)

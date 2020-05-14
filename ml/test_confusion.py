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


def plot(c, tag):
    plt.figure(figsize=(7,6))
    axis = plt.gca()
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            axis.text(
                i + 0.5,
                j + 0.5,
                '{:.2f}'.format(c[i, -1 - j]),
                ha='center',
                va='center')
    q = plt.pcolormesh(np.transpose(c)[::-1], cmap='Wistia')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    num_classes = len(cfg.ml_classes)
    plt.xticks(np.array(range(num_classes)) + 0.5,
            cfg.ml_classes, rotation='vertical')
    plt.yticks(np.array(range(num_classes)) + 0.5,
            cfg.ml_classes[::-1], rotation='horizontal')
    plt.colorbar()
    plt.savefig(os.path.join(args.workdir, 'confusion_{}_fold{}.png'.format(tag, args.fold)), bbox_inches='tight')


def main(args):
    inv_fold = [1, 0][args.fold]
    x, y, w = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(inv_fold)), cfg.ml_classes, inv_fold,
                            make_categorical=False, use_class_weights=True)

    preproc = pickle.load(open(os.path.join(args.workdir, 'preproc_fold{}.pickle'.format(args.fold)), 'rb'))
    x_preproc = preproc.transform(x)

    x_ph = tf.placeholder(tf.float32)
    _, f = model(x_ph, len(cfg.ml_variables), len(cfg.ml_classes))
    path = tf.train.latest_checkpoint(os.path.join(args.workdir, 'model_fold{}'.format(args.fold)))
    logger.debug('Load model {}'.format(path))
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, path)
    p = session.run(f, feed_dict={x_ph: x_preproc})
    p = np.argmax(p, axis=1)

    c = confusion_matrix(y, p, sample_weight=w)
    logger.debug('Confusion matrix (plain): {}'.format(c))
    plot(c, 'plain')

    c_norm_rows = c.copy()
    c_norm_cols = c.copy()
    for i in range(c.shape[0]):
        c_norm_rows[:, i] = c_norm_rows[:, i] / np.sum(c_norm_rows[:, i])
        c_norm_cols[i, :] = c_norm_cols[i, :] / np.sum(c_norm_cols[i, :])

    logger.debug('Confusion matrix (norm rows): {}'.format(c_norm_rows))
    plot(c_norm_rows, 'norm_rows')

    logger.debug('Confusion matrix (norm cols): {}'.format(c_norm_cols))
    plot(c_norm_cols, 'norm_cols')

    c_norm_all = c / np.sum(w)
    logger.debug('Confusion matrix (norm all): {}'.format(c_norm_all))
    plot(c_norm_all, 'norm_all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_test_fold{}.log'.format(args.fold)), logging.INFO)
    main(args)

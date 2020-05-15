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
from matplotlib import cm


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


def plot1d(grad_matrix, tag):
    plt.figure(figsize=(len(cfg.ml_variables), len(cfg.ml_classes)))
    for i in range(grad_matrix.shape[0]):
        for j in range(grad_matrix.shape[1]):
            plt.gca().text(j + 0.5, i + 0.5, '{:.2f}'.format(grad_matrix[i, j]), ha='center', va='center')
    q = plt.pcolormesh(grad_matrix, cmap='Wistia')
    plt.xticks(np.array(range(len(cfg.ml_variables))) + 0.5, cfg.ml_variables, rotation='vertical')
    plt.yticks(np.array(range(len(cfg.ml_classes))) + 0.5, cfg.ml_classes, rotation='horizontal')
    plt.xlim(0, len(cfg.ml_variables))
    plt.ylim(0, len(cfg.ml_classes))
    plt.savefig(os.path.join(args.workdir, 'taylor1d_{}_fold{}.png'.format(tag, args.fold)), bbox_inches='tight')


def plot2d(grad_matrix, name, tag):
    plt.figure(figsize=(len(cfg.ml_variables), len(cfg.ml_variables)))
    for i in range(grad_matrix.shape[0]):
        for j in range(grad_matrix.shape[1]):
            plt.gca().text(j + 0.5, i + 0.5, '{:.2f}'.format(grad_matrix[i, j]), ha='center', va='center')
    q = plt.pcolormesh(grad_matrix, cmap='Wistia')
    plt.xticks(np.array(range(len(cfg.ml_variables))) + 0.5, cfg.ml_variables, rotation='vertical')
    plt.yticks(np.array(range(len(cfg.ml_variables))) + 0.5, cfg.ml_variables, rotation='horizontal')
    plt.xlim(0, len(cfg.ml_variables))
    plt.ylim(0, len(cfg.ml_variables))
    plt.savefig(os.path.join(args.workdir, 'taylor2d_{}_{}_fold{}.png'.format(name, tag, args.fold)), bbox_inches='tight')


def main(args):
    inv_fold = [1, 0][args.fold]
    x, y, w = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(inv_fold)), cfg.ml_classes, inv_fold,
                            make_categorical=False, use_class_weights=True)

    preproc = pickle.load(open(os.path.join(args.workdir, 'preproc_fold{}.pickle'.format(args.fold)), 'rb'))
    x_preproc = preproc.transform(x)

    x_ph = tf.placeholder(tf.float32)
    _, f = model(x_ph, len(cfg.ml_variables), len(cfg.ml_classes), args.fold)
    path = tf.train.latest_checkpoint(os.path.join(args.workdir, 'model_fold{}'.format(args.fold)))
    logger.debug('Load model {}'.format(path))
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(session, path)

    # 1D coeffs
    grad1d_ops = []
    for i in range(len(cfg.ml_classes)):
        grad1d_ops.append(tf.gradients(f[:, i], x_ph)[0])
    grads1d = session.run(grad1d_ops, feed_dict={x_ph: x_preproc})
    grad_matrix = np.zeros((len(cfg.ml_classes), len(cfg.ml_variables)), dtype=np.float32)
    for i, g in enumerate(grads1d):
        grad_matrix[i, :] = np.mean(np.abs(g), axis=0)

    plot1d(grad_matrix, 'plain')
    grad_matrix_norm = grad_matrix.copy()
    for i in range(grad_matrix.shape[0]):
        grad_matrix_norm[i, :] = grad_matrix_norm[i, :] / np.sum(grad_matrix_norm[i, :])
    plot1d(grad_matrix_norm, 'normrows')

    # 2D coeffs
    grad2d_ops = []
    for i in range(len(cfg.ml_classes)):
        tmp = []
        for j in range(len(cfg.ml_variables)):
            tmp.append(tf.reduce_mean(tf.abs(tf.gradients(grad1d_ops[i][:, j], x_ph)[0]), axis=0))
        grad2d_ops.append(tmp)
    grads2d = session.run(grad2d_ops, feed_dict={x_ph: x_preproc})

    for i, name in enumerate(cfg.ml_classes):
        # NOTE: not the coefficients, only the gradients!
        grad2d_matrix = np.vstack(grads2d[i])
        plot2d(grad2d_matrix, name, 'plain')

        grad2d_matrix = grad2d_matrix / np.sum(grad2d_matrix)
        plot2d(grad2d_matrix, name, 'normed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_test_taylor_fold{}.log'.format(args.fold)), logging.INFO)
    main(args)

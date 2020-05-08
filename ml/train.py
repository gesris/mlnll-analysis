import os
import argparse

import ROOT
import numpy as np
np.random.seed(1234)
from sklearn.model_selection import train_test_split
from utils import config as cfg
import tensorflow as tf
tf.set_random_seed(1234)


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


def tree2numpy(path, tree, columns):
    df = ROOT.RDataFrame(tree, path)
    return df.AsNumpy(columns)


def build_dataset(path, classes, fold):
    columns = cfg.ml_variables + [cfg.ml_weight]
    xs = [] # Inputs
    ys = [] # Targets
    ws = [] # Event weights
    for i, c in enumerate(classes):
        d = tree2numpy(path, c, columns)
        xs.append(np.vstack([np.array(d[k], dtype=np.float32) for k in cfg.ml_variables]).T)
        w = np.array(d[cfg.ml_weight], dtype=np.float32)
        ws.append(w)
        ys.append(np.ones(d[cfg.ml_weight].shape) * i)

    # Stack inputs
    xs = np.vstack(xs)
    logger.debug('Input dataset (shape): {}'.format(xs.shape))

    # Stack targets
    ys = np.hstack(ys)
    logger.debug('Targets, not categorical (shape): {}'.format(ys.shape))

    # Stack weights
    ws = np.hstack(ws)
    logger.debug('Weights, without class weights (shape, sum): {}, {}'.format(ws.shape, np.sum(ws)))

    # Multiply class weights to event weights
    sum_all = np.sum(ws)
    for i in range(len(classes)):
        mask = ys == i
        ws[mask] = ws[mask] / np.sum(ws[mask]) * sum_all
    logger.debug('Weights, with class weights (shape, sum): {}, {}'.format(ws.shape, np.sum(ws)))

    # Convert targets to categorical
    ys = tf.keras.utils.to_categorical(ys)
    logger.debug('Targets, categorical (shape): {}'.format(ys.shape))

    return xs, ys, ws


def model_simple():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, activation='relu', input_shape=(len(cfg.ml_variables),)))
    model.add(tf.keras.layers.Dense(len(cfg.ml_classes), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def main(args):
    x, y, w = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)), cfg.ml_classes, args.fold)
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(x, y, w, test_size=0.5, random_state=1234)

    model = model_simple()
    model.fit(
            x=x_train,
            y=y_train,
            sample_weight=w_train,
            batch_size=10000,
            epochs=1000,
            validation_data=(x_val, y_val, w_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, verbose=2),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(args.workdir, 'model_fold{}.h5'.format(args.fold)), verbose=2, save_best_only=True)
                ]
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_train.log'), logging.DEBUG)
    main(args)

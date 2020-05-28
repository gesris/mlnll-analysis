import os
import argparse
import pickle

import ROOT
import numpy as np
np.random.seed(1234)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import config as cfg
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
tf.set_random_seed(1234)


import logging
logger = logging.getLogger('')


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


def build_dataset(path, classes, fold, make_categorical=True, use_class_weights=True):
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
    if use_class_weights:
        sum_all = np.sum(ws)
        for i in range(len(classes)):
            mask = ys == i
            ws[mask] = ws[mask] / np.sum(ws[mask]) * sum_all
        logger.debug('Weights, with class weights (shape, sum): {}, {}'.format(ws.shape, np.sum(ws)))

    # Convert targets to categorical
    if make_categorical:
        ys = tf.keras.utils.to_categorical(ys)
        logger.debug('Targets, categorical (shape): {}'.format(ys.shape))

    return xs, ys, ws


def model(x, num_variables, num_classes, fold, reuse=False):
    hidden_nodes = 100
    with tf.variable_scope('model_fold{}'.format(fold), reuse=reuse):
        w1 = tf.get_variable('w1', shape=(num_variables, hidden_nodes), initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', shape=(hidden_nodes), initializer=tf.constant_initializer())
        w2 = tf.get_variable('w2', shape=(hidden_nodes, hidden_nodes), initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('b2', shape=(hidden_nodes), initializer=tf.constant_initializer())
        w3 = tf.get_variable('w3', shape=(hidden_nodes, num_classes), initializer=tf.random_normal_initializer())
        b3 = tf.get_variable('b3', shape=(num_classes), initializer=tf.constant_initializer())

    l1 = tf.tanh(tf.add(b1, tf.matmul(x, w1)))
    l2 = tf.tanh(tf.add(b2, tf.matmul(l1, w2)))
    logits = tf.add(b3, tf.matmul(l2, w3))
    f = tf.nn.softmax(logits)

    return logits, f


def main(args):
    # Build nominal dataset
    x, y, w = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)), cfg.ml_classes, args.fold)
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(x, y, w, test_size=0.25, random_state=1234)
    logger.info('Number of train/val events in nominal dataset: {} / {}'.format(x_train.shape[0], x_val.shape[0]))
    print("\n\nX-TRAIN: {}\nLength: {}\nWidth: {}\n".format(x_train, len(x_train[0]), len(x_train[:])))
    print("\n\nY-TRAIN: {}\nLength: {}\nWidth: {}\n".format(y_train, len(y_train[0]), len(y_train[:])))
    print("\n\nW TRAIN: {}\nLength: {}\n".format(w_train, len(w_train)))
    # Build dataset for systematic shifts
    """
    x_sys, y_sys, w_sys = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)),
            ['htt', 'htt_jecUncRelativeSampleYearUp', 'htt_jecUncRelativeSampleYearDown'], args.fold,
            make_categorical=False, use_class_weights=True)
    x_sys_train, x_sys_val, w_sys_train, w_sys_val = train_test_split(x_sys, w_sys, test_size=0.25, random_state=1234)
    logger.info('Number of train/val events in varied datasets: {} / {}'.format(x_sys_train.shape[0], x_sys_val.shape[0]))
    logger.debug('Sum of weights for nominal/up/down: {} / {} / {}'.format(
        np.sum(w_sys[y_sys == 0]), np.sum(w_sys[y_sys == 1]), np.sum(w_sys[y_sys == 2])))
    """

    # Preprocessing
    preproc = StandardScaler()
    preproc.fit(x_train)
    pickle.dump(preproc, open(os.path.join(args.workdir, 'preproc_fold{}.pickle'.format(args.fold)), 'wb'))
    x_train_preproc = preproc.transform(x_train)
    x_val_preproc = preproc.transform(x_val)
    for i, (var, mean, std) in enumerate(zip(cfg.ml_variables, preproc.mean_, preproc.scale_)):
        logger.info('Variable: %s', var)
        logger.info('Preprocessing parameter (mean, std): %s, %s', mean, std)
        logger.info('Preprocessed data (mean, std): %s, %s', np.mean(x_train_preproc[:, i]), np.std(x_train_preproc[:, i]))

    # Create model
    x_ph = tf.placeholder(tf.float32)
    logits, f = model(x_ph, len(cfg.ml_variables), len(cfg.ml_classes), args.fold)

    # Add CE loss
    y_ph = tf.placeholder(tf.float32)
    w_ph = tf.placeholder(tf.float32)
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits) * w_ph)
    
    # Add loss treating systematics

    ####                ####
    ####    NLL LOSS    ####
    ####                ####

    batch_scale = tf.placeholder(tf.float32, shape=[])
    bins = np.linspace(0, 1, 3)
    upper_edges = bins[1:]
    lower_edges = bins[:-1]
    mask_algo = count_masking

    theta = tf.constant(0.0, tf.float32)
    mu = tf.constant(1.0, tf.float32)

    one = tf.constant(1, tf.float32)
    zero = tf.constant(0, tf.float32)
    epsilon = tf.constant(1e-9, tf.float32)

    nll = zero
    for i, up, down in zip(range(len(upper_edges)), upper_edges, lower_edges):
        # Bin edges
        print("\nBin (up, down, mid): {:g} / {:g} / {:g}\n".format(
            up, down, down + 0.5 * (up - down)))
        up_ = tf.constant(up, tf.float32)
        down_ = tf.constant(down, tf.float32)

        # Signals
        mask = mask_algo(f, up_, down_)
        Htt = tf.reduce_sum(mask * y_ph * w_ph * batch_scale)
        Ztt = tf.reduce_sum(mask * y_ph * w_ph * batch_scale)
        W = tf.reduce_sum(mask * y_ph * w_ph * batch_scale)
        ttbar = tf.reduce_sum(mask * y_ph * w_ph * batch_scale)

        print("\nY_PH: {}\n".format(y_ph))

        # Likelihood
        exp = mu * Htt + Ztt + W + ttbar
        sys = zero  # systematic has to be added later
        obs = Htt + Ztt + W + ttbar
        nll -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
        print("\nNLL Value: {}\n".format(nll))
    # Nuisance constraint 
    #nll -= tfp.distributions.Normal(loc=0, scale=1).log_prob(theta)


    ####                ####
    ####    SD LOSS     ####
    ####                ####

    # POI constraint (full NLL)
    def get_constraint(nll, params):
        hessian = [tf.gradients(g, params) for g in tf.unstack(tf.gradients(nll, params))]
        inverse = tf.matrix_inverse(hessian)
        covariance_poi = inverse[0][0]
        constraint = tf.sqrt(covariance_poi)
        return constraint

    sd_loss_statsonly = get_constraint(nll, [mu])
    #print("\nSD LOSS: {}\n".format(sd_loss))


    # Combine losses
    #loss = ce_loss
    loss = sd_loss_statsonly

    # Add minimization ops
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(loss)

    # Train
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer()])
    saver = tf.train.Saver(max_to_keep=1)

    patience = 30
    patience_count = patience
    min_loss = 1e9
    tolerance = 0.001
    step = 0
    batch_size = 1000
    validation_steps = int(x_train.shape[0] / batch_size)
    while True:
        idx = np.random.choice(x_train_preproc.shape[0], batch_size)
        loss_train, _ = session.run([loss, minimize],
                feed_dict={x_ph: x_train_preproc[idx], y_ph: y_train[idx], w_ph: w_train[idx], batch_scale: 2.0})
        if step % validation_steps == 0:
            logger.info('Step / patience: {} / {}'.format(step, patience_count))
            logger.info('Train loss: {:.5f}'.format(loss_train))
            loss_val = session.run(loss, feed_dict={x_ph: x_val_preproc, y_ph: y_val, w_ph: w_val, batch_scale: 2.0})
            logger.info('Validation loss: {:.5f}'.format(loss_val))

            if min_loss > loss_val and np.abs(min_loss - loss_val) / min_loss > tolerance:
                min_loss = loss_val
                patience_count = patience
                path = saver.save(session, os.path.join(args.workdir, 'model_fold{}/model.ckpt'.format(args.fold)), global_step=step)
                logger.info('Save model to {}'.format(path))
            else:
                patience_count -= 1

            if patience_count == 0:
                logger.info('Stop training')
                break

        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_train_fold{}.log'.format(args.fold)), logging.DEBUG)
    main(args)

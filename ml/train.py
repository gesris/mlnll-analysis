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
import tensorflow_probability as tfp
tf.disable_v2_behavior()
tf.set_random_seed(1234)

import matplotlib.pyplot as plt

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


def build_dataset(path, classes, fold, make_categorical=True, use_class_weights=False): #use_class_weight=True is default
    columns = cfg.ml_variables + [cfg.ml_weight, "THU_ggH_Mig01"]
    xs = [] # Inputs
    ys = [] # Targets
    ws = [] # Event weights
    mig01s = []
    for i, c in enumerate(classes):
        d = tree2numpy(path, c, columns)
        xs.append(np.vstack([np.array(d[k], dtype=np.float32) for k in cfg.ml_variables]).T)
        w = np.array(d[cfg.ml_weight], dtype=np.float32)
        mig01 = np.array(d["THU_ggH_Mig01"], dtype=np.float32)
        ws.append(w)
        ys.append(np.ones(d[cfg.ml_weight].shape) * i)
        mig01s.append(mig01)
        
    # Stack inputs
    xs = np.vstack(xs)
    logger.debug('Input dataset (shape): {}'.format(xs.shape))

    # Stack targets
    ys = np.hstack(ys)
    logger.debug('Targets, not categorical (shape): {}'.format(ys.shape))

    # Stack weights
    ws = np.hstack(ws)
    logger.debug('Weights, without class weights (shape, sum): {}, {}'.format(ws.shape, np.sum(ws)))

    # Stack Mig01 systematic weights
    mig01s = np.hstack(mig01s)
    logger.debug('Mig01 systematic weights without class weights (shape, sum): {}, {}'.format(mig01s.shape, np.sum(mig01s)))

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

    return xs, ys, ws, mig01s


def model(x, num_variables, fold, reuse=False):
    hidden_nodes = 100
    with tf.variable_scope('model_fold{}'.format(fold), reuse=reuse):
        w1 = tf.get_variable('w1', shape=(num_variables, hidden_nodes), initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', shape=(hidden_nodes), initializer=tf.constant_initializer())
        w2 = tf.get_variable('w2', shape=(hidden_nodes, 1), initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('b2', shape=(1), initializer=tf.constant_initializer())

    l1 = tf.tanh(tf.add(b1, tf.matmul(x, w1)))
    logits = tf.add(b2, tf.matmul(l1, w2))
    f = tf.nn.sigmoid(logits)

    return (w1, b1, w2, b2), f



def main(args):
    
    ####
    ####  Build nominal + sys dataset
    ####

    x, y, w, mig01 = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)), cfg.ml_classes, args.fold)
    test_size = 0.25    # has to be used later for correct batch scale
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(x, y, w, test_size=test_size, random_state=1234)
    logger.info('Number of train/val events in nominal dataset: {} / {}'.format(x_train.shape[0], x_val.shape[0]))
    logger.info("Mig01 systematic weights: {}".format(mig01))

    # Build masks for each class Htt, Ztt, W and ttbar
    y_train_array = np.array(y_train)
    y_val_array = np.array(y_val)

    Htt_mask_train = y_train_array[:, 0]
    Ztt_mask_train = y_train_array[:, 1]
    W_mask_train = y_train_array[:, 2]
    ttbar_mask_train = y_train_array[:, 3]
    #Htt_up_mask_train = y_train[:, 4]
    #Htt_down_mask_train = y_train[:, 5]

    Htt_mask_val = y_val_array[:, 0]
    Ztt_mask_val = y_val_array[:, 1]
    W_mask_val = y_val_array[:, 2]
    ttbar_mask_val = y_val_array[:, 3]    
    #Htt_up_mask_val = y_val[:, 4]
    #Htt_down_mask_val = y_val[:, 5]


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
    train_vars, f = model(x_ph, len(cfg.ml_variables), args.fold)
    w_ph = tf.placeholder(tf.float32)
    
    

    ####                ####
    ####    NLL LOSS    ####
    ####                ####

    # model parameters
    batch_scale = tf.placeholder(tf.float32)
    fold_scale = tf.placeholder(tf.float32)
    bins = cfg.analysis_binning
    logger.info("\nBins: {}".format(bins))
    upper_edges, lower_edges = bins[1:], bins[:-1]

    # method parameters
    theta = tf.constant(0.0, tf.float64)
    mu = tf.constant(1.0, tf.float64)
    zero = tf.constant(0, tf.float64)
    epsilon = tf.constant(1e-9, tf.float64)

    # nominal masks
    Htt_mask = tf.placeholder(tf.float32)
    Ztt_mask = tf.placeholder(tf.float32)
    W_mask = tf.placeholder(tf.float32)
    ttbar_mask = tf.placeholder(tf.float32)

    # sys shift
    mig01 = 1.02

    # count function
    def hist(f, bins, masking, w_ph, batch_scale, fold_scale, custom_scale):
        counts = []
        for right_edge, left_edge in zip(bins[1:], bins[:-1]):
            Events = tf.reduce_sum(count_masking(f, right_edge, left_edge) * masking * w_ph * batch_scale * fold_scale * custom_scale)
            counts.append(Events)
        return tf.squeeze(tf.stack(counts))

    # nominal counts
    Htt = tf.cast(hist(f, bins, Htt_mask, w_ph, batch_scale, fold_scale, 1), tf.float64)
    Ztt = tf.cast(hist(f, bins, Ztt_mask, w_ph, batch_scale, fold_scale, 1), tf.float64)
    W = tf.cast(hist(f, bins, W_mask, w_ph, batch_scale, fold_scale, 1), tf.float64)
    ttbar = tf.cast(hist(f, bins, ttbar_mask, w_ph, batch_scale, fold_scale, 1), tf.float64)

    # sys counts
    Htt_up = tf.cast(hist(f, bins, Htt_mask, w_ph, batch_scale, fold_scale, mig01), tf.float64)
    Htt_down = tf.cast(hist(f, bins, Htt_mask, w_ph, batch_scale, fold_scale, 1 / mig01), tf.float64)

    # Calculation of NLL (with and without sys)
    nll = zero
    nll_statsonly = zero
    for i in range(0, len(bins) - 1):
        # Likelihood
        exp = mu * Htt[i] + Ztt[i] + W[i] + ttbar[i]
        sys = tf.maximum(theta, zero) * Htt_up[i] + tf.maximum(theta, zero) * Htt_down[i]
        obs = Htt[i] + Ztt[i] + W[i] + ttbar[i]
        
        nll -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
        nll_statsonly -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))
    # Nuisance constraint 
    nll -= tf.cast(tfp.distributions.Normal(loc=0, scale=1).log_prob(tf.cast(theta, tf.float32)), tf.float64)



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

    sd_loss = get_constraint(nll, [mu, theta])
    sd_loss_statsonly = get_constraint(nll_statsonly, [mu])

    # Combine losses
    loss = sd_loss

    # Add minimization ops
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(loss)
    

    
    ####                ####
    ####    TRAINING    ####
    ####                ####

    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer()])
    saver = tf.train.Saver(max_to_keep=1)

    # training parameters
    patience = 30
    patience_count = patience
    min_loss = 1e9
    tolerance = 0.001
    step = 0
    batch_size = 1000
    validation_steps = int(x_train.shape[0] / batch_size)

    # Parameters for plotting loss convergence
    loss_train_list = []
    loss_val_list = []
    steps_list = []

    for epoch in range(0, 20000):
        loss_train, _ = session.run([loss, minimize],
                feed_dict={x_ph: x_train_preproc, w_ph: w_train,\
                            Htt_mask: Htt_mask_train, \
                            Ztt_mask: Ztt_mask_train, \
                            W_mask: W_mask_train, \
                            ttbar_mask: ttbar_mask_train, \
                            batch_scale: (1 / (1 - test_size)), \
                            fold_scale: 2})
        # Validation
        if step % 10 == 0:
            logger.info('Step / patience: {} / {}'.format(step, patience_count))
            logger.info('Train loss: {:.5f}'.format(loss_train))
            loss_val, Htt_, Htt_up_, Htt_down_, Ztt_, W_, ttbar_  = session.run([loss, Htt, Htt_up, Htt_down, Ztt, W, ttbar], feed_dict={x_ph: x_val_preproc, w_ph: w_val,\
                            Htt_mask: Htt_mask_val, \
                            Ztt_mask: Ztt_mask_val, \
                            W_mask: W_mask_val, \
                            ttbar_mask: ttbar_mask_val, \
                            batch_scale: (1 / test_size), \
                            fold_scale: 2})
            logger.info('Validation loss: {:.5f}'.format(loss_val))
            logger.info('\nHtt Total:      {:.3f}\
                        \nHtt Up Total:   {:.3f}\
                        \nHtt Down Total: {:.3f}\
                        \nZtt Total:      {:.3f}\
                        \nW Total:        {:.3f}\
                        \nttbar Total:    {:.3f}\n'.format(np.sum(Htt_), np.sum(Htt_up_), np.sum(Htt_down_), np.sum(Ztt_), np.sum(W_), np.sum(ttbar_)))
            np.set_printoptions(precision=3)
            logger.info('\nHtt Hist:       {}\
                        \nHtt Up Hist:    {}\
                        \nHtt Down Hist:  {}\
                        \nZtt Hist:       {}\
                        \nW Hist:         {}\
                        \nttbar Hist:     {}\n'.format(Htt_, Htt_up_, Htt_down_, Ztt_, W_, ttbar_))

            ### feed loss values in lists for plot 
            loss_train_list.append(loss_train)
            loss_val_list.append(loss_val)
            steps_list.append(step)

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
    
    ## Plot minimization of loss
    plt.figure()
    plt.plot(steps_list, loss_train_list)
    plt.plot(steps_list, loss_val_list)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.workdir, 'model_fold{}/minimization.png'.format(args.fold)), bbox_inches = "tight")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_train_fold{}.log'.format(args.fold)), logging.DEBUG)
    main(args)

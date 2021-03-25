import os
import argparse
import pickle

import numpy as np
np.random.seed(1234)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", size=16, family="serif")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import config as cfg
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1234)
import tensorflow_probability as tfp

import ROOT

import logging
logger = logging.getLogger('')


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
    columns = cfg.ml_variables + [cfg.ml_weight] + ['jpt_1_weights_up'] + ['jpt_1_weights_down']
    xs = [] # Inputs
    ys = [] # Targets
    ws = [] # Event weights
    jpt_1_upshifts = [] # JES upshift weights
    jpt_1_downshifts = [] # # JES downshift weights
    for i, c in enumerate(classes):
        d = tree2numpy(path, c, columns)
        xs.append(np.vstack([np.array(d[k], dtype=np.float64) for k in cfg.ml_variables]).T)
        w = np.array(d[cfg.ml_weight], dtype=np.float64)
        ws.append(w)
        ys.append(np.ones(d[cfg.ml_weight].shape, dtype=np.float64) * i)

        # JES Weights
        jpt_1_upshift = np.array(d['jpt_1_weights_up'], dtype=np.float64)
        jpt_1_upshifts.append(jpt_1_upshift)
        jpt_1_downshift = np.array(d['jpt_1_weights_down'], dtype=np.float64)
        jpt_1_downshifts.append(jpt_1_downshift)

    # Stack inputs
    xs = np.vstack(xs)
    logger.debug('Input dataset (shape): {}'.format(xs.shape))

    # Stack targets
    ys = np.hstack(ys)
    logger.debug('Targets, not categorical (shape): {}'.format(ys.shape))

    # Stack weights
    ws = np.hstack(ws)
    logger.debug('Weights, without class weights (shape, sum): {}, {}'.format(ws.shape, np.sum(ws)))

    # Stack JES weights
    jpt_1_upshifts = np.hstack(jpt_1_upshifts)
    jpt_1_downshifts = np.hstack(jpt_1_downshifts)
    logger.debug('JES upshift weights (shape): {}'.format(jpt_1_upshifts.shape))
    logger.debug('JES downshift weights (shape): {}'.format(jpt_1_downshifts.shape))

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

    return xs, ys, ws, jpt_1_upshifts, jpt_1_downshifts


def model(x, num_variables, num_classes, fold, reuse=False):
    hidden_nodes = 100
    with tf.variable_scope('model_fold{}'.format(fold), reuse=reuse):
        w1 = tf.get_variable('w1', shape=(num_variables, hidden_nodes), initializer=tf.random_normal_initializer(), dtype=tf.float64)
        b1 = tf.get_variable('b1', shape=(hidden_nodes), initializer=tf.constant_initializer(), dtype=tf.float64)
        w2 = tf.get_variable('w2', shape=(hidden_nodes, num_classes), initializer=tf.random_normal_initializer(), dtype=tf.float64)
        b2 = tf.get_variable('b2', shape=(num_classes), initializer=tf.constant_initializer(), dtype=tf.float64)

    l1 = tf.tanh(tf.add(b1, tf.matmul(x, w1)))
    logits = tf.add(b2, tf.matmul(l1, w2))
    f = tf.nn.sigmoid(logits)
    f = tf.squeeze(f)

    return logits, f, [w1, b1, w2, b2]


def main(args):
    # Build nominal dataset
    classes = cfg.ml_classes + [n + '_ss' for n in cfg.ml_classes if n not in ['ggh', 'qqh']] + ['data_ss']
    x, y, w, jpt_1_upshift, jpt_1_downshift = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)), classes, args.fold,
                            use_class_weights=False, make_categorical=False)
    x_train, x_val, y_train, y_val, w_train, w_val, jpt_1_upshift_train, jpt_1_upshift_val, jpt_1_downshift_train, jpt_1_downshift_val = train_test_split(x, y, w, jpt_1_upshift, jpt_1_downshift, test_size=0.25, random_state=1234)
    logger.info('Number of train/val events in nominal dataset: {} / {}'.format(x_train.shape[0], x_val.shape[0]))

    # Scale to expectation in the full dataset
    scale_train = 4.0 / 3.0 * 2.0 # train/test split + two fold
    scale_val = 4.0 * 2.0
    w_train = w_train * scale_train
    w_val = w_val * scale_val
    
    for i, name in enumerate(classes):
        s_train = np.sum(w_train[y_train == i])
        s_val = np.sum(w_val[y_val == i])
        logger.debug('Class / train / val: {} / {} / {}'.format(name, s_train, s_val))

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
    x_ph = tf.placeholder(tf.float64, shape=(None,len(cfg.ml_variables)))
    logits, f, w_vars = model(x_ph, len(cfg.ml_variables), 1, args.fold)

    # Build NLL loss
    y_ph = tf.placeholder(tf.float64, shape=(None,))
    w_ph = tf.placeholder(tf.float64, shape=(None,))
    jpt_1_upshift_ph = tf.placeholder(tf.float64)
    jpt_1_downshift_ph = tf.placeholder(tf.float64)

    nll = 0.0
    nll_statsonly = 0.0
    bins = np.array(cfg.analysis_binning)
    mu = tf.constant(1.0, tf.float64)
    n_jes = tf.constant(0.0, tf.float64)
    nuisances = []
    zero = tf.constant(0, tf.float64)
    epsilon = tf.constant(1e-9, tf.float64)
    for i, (up, down) in enumerate(zip(bins[1:], bins[:-1])):
        logger.debug('Add NLL for bin {} with boundaries [{}, {}]'.format(i, down, up))
        up = tf.constant(up, tf.float64)
        down = tf.constant(down, tf.float64)

        # Processes
        mask = count_masking(f, up, down)
        procs = {}
        procs_up = {}
        procs_down = {}
        procs_noweight = {}
        scale_bbb = 0.0
        sum_weighted = 0.0
        sum_nonweighted = 0.0

        for j, name in enumerate(classes):
            proc_w = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph
            proc_w_up = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph * jpt_1_upshift_ph
            proc_w_down = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph * jpt_1_downshift_ph
            procs[name] = tf.reduce_sum(proc_w)
            procs_up[name] = tf.reduce_sum(proc_w_up)
            procs_down[name] = tf.reduce_sum(proc_w_down)

            # BBB
            proc_noweight = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64)
            procs_noweight[name] = tf.reduce_sum(proc_noweight) 
            sum_weighted += procs[name]
            sum_nonweighted += procs_noweight[name]
        scale_bbb += sum_weighted / sum_nonweighted


        # QCD estimation
        procs['qcd'] = procs['data_ss']
        for p in [n for n in cfg.ml_classes if not n in ['ggh', 'qqh']]:
            procs['qcd'] -= procs[p + '_ss']
        procs['qcd'] = tf.maximum(procs['qcd'], 0)

        # Nominal signal and background
        sig = 0
        for p in ['ggh', 'qqh']:
            sig += procs[p]

        bkg = 0
        for p in ['ztt', 'zl', 'w', 'tt', 'vv', 'qcd']:
            bkg += procs[p]

        # JES Uncertainty
        sys = 0.0
        for p in ['ggh', 'qqh', 'ztt', 'zl', 'w', 'tt', 'vv']:
            Delta_up = tf.maximum(n_jes, zero) * (procs_up[p] - procs[p])
            Delta_down = tf.minimum(n_jes, zero) * (procs[p] - procs_down[p])
            sys += Delta_up + Delta_down
        
        # BBB Uncertainty
        bbb = tf.constant(0.0, tf.float64)
        n_bbb = tf.constant(0.0, tf.float64)
        for p in ['ztt', 'zl', 'w', 'tt', 'vv', 'qcd']:
            bbb += procs_noweight[p]
        sys += n_bbb * (tf.sqrt(bbb) * scale_bbb) 
        nuisances.append(n_bbb)

        # Expectations
        obs = sig + bkg
        exp = mu * sig + bkg + sys 

        # Likelihood
        nll -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))
        nll_statsonly -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))
    
    # Nuisance constraints
    nuisances.append(n_jes)
    for n in nuisances:
        nll -= tfp.distributions.Normal(
                loc=tf.constant(0.0, dtype=tf.float64), scale=tf.constant(1.0, dtype=tf.float64)
                ).log_prob(n)

    # Compute constraint of mu
    def get_constraint(nll, params):
        hessian = [tf.gradients(g, params) for g in tf.unstack(tf.gradients(nll, params))]
        inverse = tf.matrix_inverse(hessian)
        covariance_poi = inverse[0][0]
        constraint = tf.sqrt(covariance_poi)
        return constraint
    loss_statsonly = get_constraint(nll_statsonly, [mu])
    logger.info("Mark 1")
    loss_fullnll = get_constraint(nll, [mu] + nuisances)
    logger.info("Mark 2")

    
    # Add minimization ops
    def get_minimize_op(loss):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(loss, var_list=w_vars)

    minimize_fullnll = get_minimize_op(loss_fullnll)
    logger.info("Mark 3")
    minimize_statsonly = get_minimize_op(loss_statsonly)
    logger.info("Mark 4")

    # Train
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer()])
    saver = tf.train.Saver(max_to_keep=1)

    patience = 5000
    patience_count = patience
    min_loss = 1e9
    tolerance_init = 0.01
    tolerance_min = 0.001
    step = 0
    validation_steps = 20
    warmup_steps = 100

    steps_list = []
    loss_train_list = []
    loss_val_list = []

    while True:
        if step < warmup_steps:
            loss = loss_statsonly
            minimize = minimize_statsonly
            is_warmup = True
        else:
            loss = loss_fullnll
            minimize = minimize_fullnll
            is_warmup = False

        loss_train, _ = session.run([loss, minimize],
                feed_dict={x_ph: x_train_preproc, y_ph: y_train, w_ph: w_train, jpt_1_upshift_ph: jpt_1_upshift_train, jpt_1_downshift_ph: jpt_1_downshift_train})
        if is_warmup:
            loss_val = session.run(loss, feed_dict={x_ph: x_val_preproc, y_ph: y_val, w_ph: w_val, jpt_1_upshift_ph: jpt_1_upshift_val, jpt_1_downshift_ph: jpt_1_downshift_val})
        else:
            loss_val = session.run(loss, feed_dict={x_ph: x_val_preproc, y_ph: y_val, w_ph: w_val, jpt_1_upshift_ph: jpt_1_upshift_val, jpt_1_downshift_ph: jpt_1_downshift_val})
            tolerance = np.maximum(tolerance_init  * (100 / (100 + step - warmup_steps)), tolerance_min)
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

        ## Loss Display / Plot
        if step % validation_steps == 0:
            logger.info('Step / patience: {} / {}'.format(step, patience_count))
            logger.info('Train loss: {:.5f}'.format(loss_train))
            loss_val = session.run(loss, feed_dict={x_ph: x_val_preproc, y_ph: y_val, w_ph: w_val, jpt_1_upshift_ph: jpt_1_upshift_val, jpt_1_downshift_ph: jpt_1_downshift_val})
            logger.info('Validation loss: {:.5f}'.format(loss_val))
            # path = saver.save(session, os.path.join(args.workdir, 'model_fold{}/model.ckpt'.format(args.fold)), global_step=step)
            # logger.info('Save model to {}'.format(path))

        if is_warmup:
            logger.info('Warmup: {} / {}'.format(step, warmup_steps))
        else:
            steps_list.append(step)
            loss_train_list.append(loss_train)
            loss_val_list.append(loss_val)
        step += 1


    ## Plot minimization of loss
    plt.figure()
    plt.plot(steps_list, loss_train_list)
    plt.plot(steps_list, loss_val_list)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.workdir, 'model_fold{}/minimization_fold{}.png'.format(args.fold, args.fold)), bbox_inches = "tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_train_fold{}.log'.format(args.fold)), logging.DEBUG)
    main(args)

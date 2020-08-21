import os
import argparse
import pickle

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
    x, y, w = build_dataset(os.path.join(args.workdir, 'fold{}.root'.format(args.fold)), classes, args.fold,
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
    scale_ph = tf.placeholder(tf.float64)

    bins = np.array(cfg.analysis_binning)
    bins_center = []
    for left, right in zip(bins[1:], bins[:-1]):
        bins_center.append(left + (right - left) / 2)
    
    
    ## Calculate NLL
    mu0 = tf.constant(1.0, tf.float64)
    
    def nll_value(mu):
        nll = 0.0
        theta = tf.constant(0.0, tf.float64)
        epsilon = tf.constant(1e-9, tf.float64)
        nuisance_param = {}
        bincontent = {}
        tot_procssumw2 = {}

        for i, (up, down) in enumerate(zip(bins[1:], bins[:-1])):
            counts = {}
            logger.debug('Add NLL for bin {} with boundaries [{}, {}]'.format(i, down, up))
            up = tf.constant(up, tf.float64)
            down = tf.constant(down, tf.float64)

            # Processes
            mask = count_masking(f, up, down)
            procs = {}
            procs_sumw2 = {}
            for j, name in enumerate(classes):
                proc_w = mask * tf.cast(tf.equal(y_ph, tf.constant(j, tf.float64)), tf.float64) * w_ph
                procs[name] = tf.reduce_sum(proc_w)
                procs_sumw2[name] = tf.reduce_sum(tf.square(proc_w / scale_ph) * scale_ph)


            # QCD estimation
            procs['qcd'] = procs['data_ss']
            for p in [n for n in cfg.ml_classes if not n in ['ggh', 'qqh']]:
                procs['qcd'] -= procs[p + '_ss']
            procs['qcd'] = tf.maximum(procs['qcd'], 0)


            # Nominal signal and background
            sig = 0
            for p in ['ggh', 'qqh']:
                sig += procs[p]
                counts[p] = procs[p]

            bkg = 0
            for p in ['ztt', 'zl', 'w', 'tt', 'vv', 'qcd']:
                bkg += procs[p]
                counts[p] = procs[p]


            # Add total content to nested dictionary
            bincontent[i] = counts
            tot_procssumw2[i] = procs_sumw2


            # Bin by bin uncertainties
            shift = 0.0
            #for p in ['ggh', 'qqh', 'ztt', 'zl', 'w', 'tt', 'vv']:
            for p in ['w']:
                shift += procs_sumw2[p]
            shift = tf.sqrt(shift)
            nuisance_param["bbb"] = theta
            sys = theta * shift

            # Expectations
            obs = sig + bkg
            exp = mu * sig + bkg + sys 
            
            # Likelihood
            nll -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))
        # Nuisance constraints
        #for n in nuisance_param:
        #    nll -= tfp.distributions.Normal(
        #            loc=tf.constant(0.0, dtype=tf.float64), scale=tf.constant(1.0, dtype=tf.float64)
        #            ).log_prob(nuisance_param[n])
        nll -= tfp.distributions.Normal(
                    loc=tf.constant(0.0, dtype=tf.float64), scale=tf.constant(1.0, dtype=tf.float64)
                    ).log_prob(theta)
        return nll, bincontent, tot_procssumw2


    ## Calculating bbb-unc. and bincontent for histogram
    _, bincontent, tot_procssumw2 = nll_value(mu0)
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(session, path)
    bincontent_, tot_procssumw2_ = session.run([bincontent, tot_procssumw2], \
                        feed_dict={x_ph: x_preproc, y_ph: y, w_ph: w, scale_ph: fold_factor})


    ## Printing bbb uncertainty for every class 
    summe = 0   
    for i, element in enumerate(['w']):
        content = []
        for id, classes in tot_procssumw2_.items():
            content.append(np.sqrt(classes[element]))
        content = np.array(content)
        np.set_printoptions(precision=3)
        summe += np.sum(content)
        print("{}: {}".format(element, content))
    print("TOTAL SUM OF ALL UNC: {}".format(summe))


    ## Plotting histogram
    def plot(bincontent, bins, bins_center):
        plt.figure(figsize=(7, 6))
        color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for i, element in enumerate(['ggh', 'qqh', 'ztt', 'zl', 'w', 'tt', 'vv', 'qcd']):
            content = []
            for id, classes in bincontent.items():
                content.append(classes[element])
            #if element in ['ggh', 'qqh']:
            #    plt.hist(bins_center, weights= content, bins= bins, histtype="step", lw=2, color="C0")
            #    plt.plot([0], [0], lw=2, color="C0", label=element)
            #else:
            #    plt.hist(bins_center, weights= content, bins= bins, ls="--", histtype="step", lw=2)
            #    plt.plot([0], [0], lw=2, ls="--", label=element)
            plt.hist(bins_center, weights=content, bins=bins, histtype="step", lw=2, color=color[i])
            plt.plot([0], [0], lw=2, color=color[i], label=element)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
        plt.xlabel("$f$")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.savefig(os.path.join(args.workdir, 'model_fold{}/histogram{}.png'.format(args.fold, args.fold)), bbox_inches = "tight")
    plot(bincontent_, bins, bins_center)

    
    ## Calculate DNLL
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(session, path)
    
    x = np.linspace(0.0, 2.0, 30)
    dnll_array = []
    for i in x:
        mu1 = tf.constant(i, tf.float64)
        nll0_, nll1_ = session.run([nll_value(mu0), nll_value(mu1)], \
            feed_dict={x_ph: x_preproc, y_ph: y, w_ph: w, scale_ph: fold_factor})
        nll0, _, _, _ = nll0_
        nll1, _, _, _ = nll1_
        dnll_array.append(-2 * (nll0 - nll1))


    ## Plot scan
    def plot_scan(x, dnll_array):
        ## Interpolate DNLL data
        f_dnll_array = interpolate.UnivariateSpline(x, dnll_array, s=0)
        x_new = np.arange(0.0, 2.0, 0.02)

        y_target = 1
        y_reduced = np.array(dnll_array) - y_target
        freduced = interpolate.UnivariateSpline(x, y_reduced, s=0)
        constraints_xval = freduced.roots()
        constraints = [1 - constraints_xval[0], constraints_xval[1] - 1]

        y_limit = [0.0, 4.5]
        x_limit = [0.5, 1.5]
        linewidth_narrow = 1.
        linewidth_wide = 2.

        ## Plot interpolation
        plt.figure(figsize=(7,6))
        plt.xlabel("$\mu$")
        plt.xlim((x_limit[0], x_limit[1]))
        plt.ylabel("-2 $\cdot \/ \Delta$NLL")
        plt.ylim((y_limit[0], y_limit[1]))
        plt.plot(x_new, f_dnll_array(x_new), color='C0', lw=linewidth_wide)
        #plt.plot(x, y, color='C1', lw=linewidth_wide)
        plt.plot([x_limit[0], constraints_xval[0]], [1, 1], 'k', lw=linewidth_narrow)
        plt.plot([constraints_xval[1], x_limit[1]], [1, 1], 'k', lw=linewidth_narrow)
        vscale = 1 / y_limit[1]
        plt.axvline(x=constraints_xval[0], ymax=1. * vscale, color='C0', lw=linewidth_narrow)
        plt.axvline(x=constraints_xval[1], ymax=1. * vscale, color='C0', lw=linewidth_narrow)
        plt.plot([0], [0], color='C0', label="$\mu$ = 1.00 (-{:.3f} +{:.3f})".format(constraints[1], constraints[0]))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0., prop={'size': 14})
        plt.savefig(os.path.join(args.workdir, 'model_fold{}/scan_cross_check{}.png'.format(args.fold, args.fold)), bbox_inches="tight")
    plot_scan(x, dnll_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'ml_test_fold{}.log'.format(args.fold)), logging.INFO)
    main(args)
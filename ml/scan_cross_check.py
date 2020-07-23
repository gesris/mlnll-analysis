import os
import argparse
import csv
from csv import reader

import numpy as np
np.random.seed(1234)
from scipy import interpolate
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tf.disable_v2_behavior()
tf.set_random_seed(1234)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc("font", size=16, family="serif")

from tqdm import tqdm


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


def main():
    def load_hists():
        with open(os.path.join(args.workdir, 'model_fold{}/hists.csv'.format(args.fold)), 'rU') as file:
            counts = []
            for line in file:
                lines = []
                elements = line.split()
                for i in range(0, len(elements)):
                    lines.append(float(elements[i]))
                counts.append(lines)
        Htt = tf.constant(counts[0], tf.float64)
        Ztt = tf.constant(counts[1], tf.float64)
        W = tf.constant(counts[2], tf.float64)
        ttbar = tf.constant(counts[3], tf.float64)
        return(Htt, Ztt, W, ttbar)

    Htt, Ztt, W, ttbar = load_hists()
    mu = tf.constant(1.0, tf.float64)

    def nll_value(mu, Htt, Ztt, W, ttbar):
        zero = tf.constant(0, tf.float64)
        epsilon = tf.constant(1e-9, tf.float64)
        nll = zero
        nll_statsonly = zero
        length = tf.Session().run(tf.squeeze(tf.shape(Htt)))
        for i in range(0, length):
            # Likelihood
            exp = mu * Htt[i] + Ztt[i] + W[i] + ttbar[i]
            sys = zero  # systematic has to be added later
            obs = Htt[i] + Ztt[i] + W[i] + ttbar[i]
            
            nll -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
            nll_statsonly -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))
        return tf.Session().run(nll_statsonly)

    ## Calculate DNLL
    x = np.linspace(0.0, 2.0, 30)
    dnll_array = []
    for i in x:
        mu1 = tf.constant(i, tf.float64)
        dnll_array.append(- 2 * (nll_value(mu, Htt, Ztt, W, ttbar) - nll_value(mu1, Htt, Ztt, W, ttbar)))

    ## Interpolate DNLL data
    f_dnll_array = interpolate.UnivariateSpline(x, dnll_array, s=0)
    x_new = np.arange(0.0, 2.0, 0.02)

    y_target = 1
    y_reduced = np.array(dnll_array) - y_target
    freduced = interpolate.UnivariateSpline(x, y_reduced, s=0)
    constraints_xval = freduced.roots()
    constraints = [1 - constraints_xval[0], constraints_xval[1] - 1]

    ####
    #### Create data for parabola fit
    ####

    #def f(x, a, b):
    #    return a*(x-b)**2

    #def second_derivative(mu, Htt, Ztt, W, ttbar):
    #    return tf.Session().run(tf.gradients(tf.gradients(nll_value(mu, Htt, Ztt, W, ttbar), mu), mu))

    #a = second_derivative(mu, Htt, Ztt, W, ttbar)
    #y = f(x, a, 1)
    

    ####
    #### Plot data
    ####

    ## Plot settings
    y_limit = [0.0, 4.5]
    x_limit = [0.5, 1.5]
    linewidth_narrow = 1.
    linewidth_wide = 2.

    plt.figure(figsize=(7,6))
    plt.xlabel("$\mu$")
    plt.xlim((x_limit[0], x_limit[1]))
    plt.ylabel("-2 $\cdot \/ \Delta$NLL")
    plt.ylim((y_limit[0], y_limit[1]))
    plt.plot(x_new, f_dnll_array, color='C0', lw=linewidth_wide)
    #plt.plot(x, y, color='C1', lw=linewidth_wide)
    plt.plot([x_limit[0], constraints_xval[0]], [1, 1], 'k', lw=linewidth_narrow)
    plt.plot([constraints_xval[1], x_limit[1]], [1, 1], 'k', lw=linewidth_narrow)
    vscale = 1 / y_limit[1]
    plt.axvline(x=constraints_xval[0], ymax=1. * vscale, color='C0', lw=linewidth_narrow)
    plt.axvline(x=constraints_xval[1], ymax=1. * vscale, color='C0', lw=linewidth_narrow)
    plt.plot([0], [0], color='C0', label="$\mu_{}$         = 1.00 (-{:.3f} +{:.3f})".format('\mathrm{stat.}', constraints[1], constraints[0]))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.savefig(os.path.join(args.workdir, 'model_fold{}/scan_cross_check{}.png'.format(args.fold, args.fold)), bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'scan_fold{}.log'.format(args.fold)), logging.INFO)
    main()
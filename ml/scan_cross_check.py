import os
import argparse

import numpy as np
np.random.seed(1234)
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_probability as tfp
tf.disable_v2_behavior()
tf.set_random_seed(1234)

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from csv import reader

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
    mu = tf.constant(1.0, tf.float64)

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
        Htt_up = tf.constant(counts[4], tf.float64)
        Htt_down = tf.constant(counts[5], tf.float64)
        return(Htt, Ztt, W, ttbar, Htt_up, Htt_down)
    

    def nll_value(mu, Htt, Ztt, W, ttbar, Htt_up, Htt_down):
        magnification = 10.
        zero = tf.constant(0, tf.float64)
        epsilon = tf.constant(1e-9, tf.float64)
        nll = zero
        nll_statsonly = zero
        theta = tf.Variable(0.0, dtype=tf.float64, trainable=True)
        length = tf.Session().run(tf.squeeze(tf.shape(Htt)))
        for i in range(0, length):
            # Likelihood
            exp = mu * Htt[i] + Ztt[i] + W[i] + ttbar[i]
            sys = (tf.maximum(theta, zero) * (Htt_up[i] - Htt[i]) + tf.minimum(theta, zero) * (Htt[i] - Htt_down[i])) * magnification   # magnifying systematic shift by factor of 10
            obs = Htt[i] + Ztt[i] + W[i] + ttbar[i]
            
            nll -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
            nll_statsonly -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))
        nll -= tf.cast(tfp.distributions.Normal(loc=0, scale=1).log_prob(tf.cast(theta, tf.float32)), tf.float64)

        # Minimize Theta
        opt = tf.train.AdamOptimizer().minimize(nll, var_list=[theta])
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print("---")
            for i in range(20):
                session.run(opt)
                print(session.run([theta, nll]))
        
        return nll_statsonly, nll


    def create_dnll_file(mu0, x, Htt, Ztt, W, ttbar, Htt_up, Htt_down):
        # empty file
        open(os.path.join(args.workdir, 'model_fold{}/dnll_value_list.csv'.format(args.fold)), "w").close()

        # write new data into file
        mu1 = tf.constant(x, dtype=tf.float64)
        for i in tqdm(range(0, len(x))):
            # NOSYS
            nll_val_nosys, _ = nll_value(mu0, Htt, Ztt, W, ttbar, Htt_up, Htt_down)
            _, nll_val_sys = nll_value(mu0, Htt, Ztt, W, ttbar, Htt_up, Htt_down)
            nll_val_nosys_var, _  = nll_value(mu1[i], Htt, Ztt, W, ttbar, Htt_up, Htt_down)
            _, nll_val_sys_var  = nll_value(mu1[i], Htt, Ztt, W, ttbar, Htt_up, Htt_down)

            dnll = 2 * (nll_val_nosys_var - nll_val_nosys)
            dnll_sys = 2 * (nll_val_sys_var - nll_val_sys)

            session = tf.Session()
            session.run([tf.global_variables_initializer()])
            d_value_nosys, d_value_sys = session.run([dnll, dnll_sys])

            with open(os.path.join(args.workdir, 'model_fold{}/dnll_value_list_nosys.csv'.format(args.fold)), "ab") as file:
                np.savetxt(file, [d_value_nosys])
            with open(os.path.join(args.workdir, 'model_fold{}/dnll_value_list_sys.csv'.format(args.fold)), "ab") as file:
                np.savetxt(file, [d_value_sys])


    def scan_from_file(x, method):
        with open(os.path.join(args.workdir, 'model_fold{}/dnll_value_list_{}.csv'.format(args.fold, method)), 'r') as file:
            diff = []
            sigma_left_list = []
            for i, d_value_ in enumerate(reader(file)):
                d_value = float(d_value_[0])
                if d_value <= 1.05 and d_value >= 0.95 and i > len(x) / 2:
                    sigma_right = x[i] - 1
                elif d_value <= 1.05 and d_value >= 0.95 and i < len(x) / 2:
                    sigma_left_list.append(1 - x[i])  #choose value furthest away from 1
                    #sigma_left = 1 - i * scaling
                diff.append(d_value)
            sigma_left = sigma_left_list[0]
        return diff, sigma_left, sigma_right 
                

    #def second_derivative(mu, Htt, Ztt, W, ttbar, Htt_up, Htt_down):
    #    return tf.Session().run(tf.gradients(tf.gradients(nll_value(mu, Htt, Ztt, W, ttbar, Htt_up, Htt_down), mu), mu))


    Htt, Ztt, W, ttbar, Htt_up, Htt_down = load_hists()

    ####
    #### Create data for parabola fit
    ####

    #def f(x, a, b):
    #    return a*(x-b)**2

    x = np.linspace(0.0, 2.0, 100)
    #a = second_derivative(mu, Htt, Ztt, W, ttbar, Htt_up, Htt_down)
    #y = f(x, a, 1)
    

    ####
    #### only call this function, if there is no .csv file containing dnll-values
    ####

    create_dnll_file(1.0, x, Htt, Ztt, W, ttbar, Htt_up, Htt_down)


    ####
    #### assign values from .csv file
    ####

    diff_nll, sigma_left, sigma_right = scan_from_file(x, 'nosys')
    diff_nll_sys, sigma_left_sys, sigma_right_sys = scan_from_file(x, 'sys')


    ####
    #### Plot data
    ####

    plt.figure()
    plt.plot(x, diff_nll)
    plt.plot(x, diff_nll_sys, color='k')
    #plt.plot(x, y, color='k')
    plt.xlabel("r = 1.0 +{:.4f} -{:.4f}".format(sigma_right, sigma_left))
    plt.xlim((0, 2))
    plt.ylabel("-2 Delta NLL")
    plt.ylim((0, 9))
    plt.axvline(x= 1. - sigma_left, ymax=1. / 9., color='r')
    plt.axvline(x= 1. + sigma_right, ymax=1. / 9., color='r')
    plt.axhline(y=1., xmin=0., xmax=(1.-sigma_left) / 2., color='r')
    plt.axhline(y=1., xmin=(1.+sigma_right) / 2., xmax=2. / 2., color='r')
    #plt.axhline(y=1., color='r')
    #plt.savefig("./scan_cross_check.png", bbox_inches="tight")
    plt.savefig(os.path.join(args.workdir, 'model_fold{}/scan_cross_check.png'.format(args.fold)), bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('fold', type=int, help='Training fold')
    args = parser.parse_args()
    setup_logging(os.path.join(args.workdir, 'scan_fold{}.log'.format(args.fold)), logging.INFO)
    main()


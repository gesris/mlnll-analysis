import numpy as np
np.random.seed(1234)
import tensorflow.compat.v1 as tf
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
    mu = tf.constant(1.0, tf.float32)

    def load_hists():
        with open('./hists.csv', 'rU') as file:
            counts = []
            for line in file:
                lines = []
                for element in line:
                    lines.append(float(element))
                counts.append(lines)
        Htt = counts[0]
        Ztt = counts[1]
        W = counts[2]
        ttbar = counts[3]
        return(Htt, Ztt, W, ttbar)
    

    def nll_value(mu, Htt, Ztt, W, ttbar):
        zero = tf.constant(0, tf.float32)
        epsilon = tf.constant(1e-9, tf.float32)
        nll = zero
        nll_statsonly = zero
        for i in range(0, len(Htt)):
            # Likelihood
            exp = mu * Htt[i] + Ztt[i] + W[i] + ttbar[i]
            sys = zero  # systematic has to be added later
            obs = Htt[i] + Ztt[i] + W[i] + ttbar[i]
            
            nll -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
            nll_statsonly -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))
        return nll_statsonly


    def create_dnll_file(mu0, x, Htt, Ztt, W, ttbar):
        # empty file
        open("dnll_value_list.csv", "w").close()

        # write new data into file
        mu1 = tf.constant(x, dtype=tf.float32)
        for i in tqdm(range(0, len(x))):
            d_value = [tf.Session().run(2 * (nll_value(mu1[i], Htt, Ztt, W, ttbar) - nll_value(mu0, Htt, Ztt, W, ttbar)))]
            with open("./dnll_value_list.csv", "ab") as file:
                np.savetxt(file, d_value)


    def scan_from_file(x):
        with open('./dnll_value_list.csv', 'r') as file:
            scaling = 2. / len(x)
            diff = []
            sigma_left_list = []
            for i, d_value_ in enumerate(reader(file)):
                d_value = float(d_value_[0])
                if d_value <= 1.04 and d_value >= 0.96 and i * scaling > 1.:
                    sigma_right = i * scaling - 1
                elif d_value <= 1.03 and d_value >= 0.97 and i * scaling < 1.:
                    sigma_left_list.append(1 - i * scaling)  #choose value furthest away from 1
                    #sigma_left = 1 - i * scaling
                diff.append(d_value)
            sigma_left = sigma_left_list[0]
        return diff, sigma_left, sigma_right 
                

    def second_derivative(mu, Htt, Ztt, W, ttbar):
        return tf.Session().run(tf.gradients(tf.gradients(nll_value(mu, Htt, Ztt, W, ttbar), mu), mu))


    Htt, Ztt, W, ttbar = load_hists()

    ####
    #### Create data for parabola fit
    ####

    def f(x, a, b):
        return a*(x-b)**2

    x = np.linspace(0, 2, 51)
    a = second_derivative(mu, Htt, Ztt, W, ttbar)
    y = f(x, a, 1)
    
    ####
    #### only call this function, if there is no .csv file containing dnll-values
    ####

    create_dnll_file(mu, x, Htt, Ztt, W, ttbar)


    ####
    #### assign values from .csv file
    ####

    diff_nll, sigma_left, sigma_right = scan_from_file(x)
    #print('DIFF NLL: {}'.format(diff_nll))


    ####
    #### Plot data
    ####

    plt.figure()
    plt.plot(x, diff_nll)
    plt.plot(x, y, color='k')
    plt.xlabel("r = 1.0 +{:.4f} -{:.4f}".format(sigma_right, sigma_left))
    plt.xlim((0, 2))
    plt.ylabel("-2 Delta NLL")
    plt.ylim((0, 9))
    plt.axvline(x= 1. - sigma_left, ymax=1. / 9., color='r')
    plt.axvline(x= 1. + sigma_right, ymax=1. / 9., color='r')
    plt.axhline(y=1., xmin=0., xmax=(1.-sigma_left) / 2., color='r')
    plt.axhline(y=1., xmin=(1.+sigma_right) / 2., xmax=2. / 2., color='r')
    #plt.axhline(y=1., color='r')
    plt.savefig("./scan_cross_check.png", bbox_inches="tight")


if __name__ == '__main__':
    main()


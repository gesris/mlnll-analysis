import numpy as np
np.random.seed(1234)
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tf.disable_v2_behavior()
tf.set_random_seed(1234)

import matplotlib.pyplot as plt
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
    mu = tf.constant(1.0, tf.float32)

    Htt = [544.02484, 291.63315, 235.41945, 89.54457]
    Ztt = [107477.97, 7436.9565, 3119.643, 4390.905]
    W = [41067.562, 4640.386, 3397.768, 13653.995]
    ttbar = [12337.207, 441.09576, 337.55807, 677.2558]

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

    def scan(mu0, x, Htt, Ztt, W, ttbar):
        diff = []

        mu1 = tf.constant(x, dtype=tf.float32)
        one_plus = tf.constant(1.05, tf.float32)
        one_minus = tf.constant(0.95, tf.float32)
        sigma_left = []
        sigma_right = []
        for i in tqdm(range(0, len(x))):
            scaling = 2. / len(x)
            d_value = sess.run(2 * (nll_value(mu1[i], Htt, Ztt, W, ttbar) - nll_value(mu0, Htt, Ztt, W, ttbar)))
            diff.append(d_value)
            if d_value <= 1.1 and d_value >= 0.9 and i * scaling < 1:
                sigma_left = 1 - i * scaling   # ACHTUNG: nimmt den wert, der am nähsten zu 1 steht
            elif d_value <= 1.05 and d_value >= 0.95 and i * scaling > 1:
                sigma_right = i * scaling - 1
        print('SIGMA L: {}'.format(sigma_left))
        print('SIGMA R: {}'.format(sigma_right))
        return diff, sigma_left, sigma_right

    def second_derivative(mu, Htt, Ztt, W, ttbar):
        return tf.gradients(tf.gradients(nll_value(mu, Htt, Ztt, W, ttbar), mu), mu)

    sess = tf.Session()

    print(sess.run(second_derivative(mu, Htt, Ztt, W, ttbar)))

    '''
    x = np.linspace(0, 2, 201)
    diff_nll, sigma_left, sigma_right = scan(mu, x, Htt, Ztt, W, ttbar)
    print('DIFF NLL: {}'.format(diff_nll))
    
    plt.figure()
    plt.plot(x, diff_nll)
    plt.xlabel("r = 1.0 +{:.4f} -{:.4f}".format(sigma_right, sigma_left))
    plt.xlim((0, 2))
    plt.ylabel("-2 Delta NLL")
    plt.ylim((0, 9))
    plt.axvline(x= 1. - sigma_left, ymax=1. / 9., color='r')
    plt.axvline(x= 1. + sigma_right, ymax=1. / 9., color='r')
    plt.axhline(y=1., xmin=0., xmax=(1.-sigma_left) / 2., color='r')
    plt.axhline(y=1., xmin=(1.+sigma_right) / 2., xmax=2. / 2., color='r')
    plt.savefig("./scan_cross_check.png", bbox_inches="tight")
    '''

if __name__ == '__main__':
    main()


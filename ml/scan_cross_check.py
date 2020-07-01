import numpy as np
np.random.seed(1234)
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tf.disable_v2_behavior()
tf.set_random_seed(1234)

import matplotlib.pyplot as plt

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
        mu1 = tf.stack(x)
        for i in range(0, len(x)):
            diff.append(nll_value(mu1[i], Htt, Ztt, W, ttbar) - nll_value(mu0, Htt, Ztt, W, ttbar))
        return diff

    x = np.linspace(0, 2, 31)
    diff_nll = scan(mu, x, Htt, Ztt, W, ttbar)

    plt.figure()
    plt.plot(x, diff_nll)
    plt.xlabel("r")
    plt.ylabel("Delta NLL")
    #plt.show()



if __name__ == '__main__':
    main()


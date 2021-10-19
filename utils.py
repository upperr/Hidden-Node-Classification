from __future__ import division
import tensorflow as tf

def sample_normal(mean, log_std):
    # mu + standard_samples * stand_deviation
    x = mean + tf.random.normal(tf.shape(mean)) * tf.exp(log_std)
    return x

def kl_normal(mean_posterior, log_std, mean_prior = 0.):
    #mean, log_std: d × N × K
    kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + 2 * log_std - tf.square(mean_posterior - mean_prior) - tf.square(tf.exp(log_std)), axis = 1))
    return kl

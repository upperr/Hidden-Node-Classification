import tensorflow as tf
from utils import kl_normal

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

class Optimizer(object):
    def __init__(self, model, placeholders, epoch, batch_size):

        epoch = tf.cast(epoch, tf.float32)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)  # Adam Optimizer
        
        self.loss = tf.constant(0.0)
        # S MC samples:
        for s in range(model.S):
            # classification loss
            # use sigmoid loss if nodes can belong to multiple classes
            if FLAGS.sigmoid:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits = model.pred_logit_list[s],
                                            labels = placeholders['labels']))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits = model.pred_logit_list[s],
                                            labels = tf.stop_gradient(placeholders['labels'])))
            self.loss += loss
        self.loss = self.loss / model.S 

        # Regularization Loss
        self.regularization = model.get_regualizer_cost(tf.nn.l2_loss)
       
        # KL-divergence loss                
        mean_posterior = model.posterior_theta_param[0]
        log_std_posterior = model.posterior_theta_param[1]
        self.kl = kl_normal(mean_posterior, log_std_posterior) / tf.cast(batch_size, tf.float32)

        self.wu_beta = epoch / FLAGS.epochs
        if FLAGS.use_kl_warmup == 0:
            self.wu_beta = 1

        self.ae_loss = self.loss + self.regularization * FLAGS.weight_decay
        self.cost = self.loss + 1. * self.wu_beta * self.kl + FLAGS.weight_decay * self.regularization

        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # gradient clipping
        self.clipped_grads_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else 0, var)
                for grad, var in self.grads_vars]

        self.opt_op = self.optimizer.apply_gradients(self.clipped_grads_vars)

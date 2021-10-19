#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

from optimizer import Optimizer
from input_data import load_data
from model import DGVAE
from minibatch import NodeMinibatchIterator

# Settings
tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(1234)
np.random.seed(1234)
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'reddit', 'Dataset string: reddit, elliptic, flickr, Deezer')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Max number of epochs to train. Training may stop early if validation-error is not decreasing')
flags.DEFINE_string('encoder', '128_128', 'Number of units in encoder layers. Connect numbers using _.')
flags.DEFINE_string('num_neighbors', '25_10', 'Number of nodes sampled at each layer. Connect numbers using _.')
flags.DEFINE_integer('decoder', 50, 'Number of units in decoder layers')
flags.DEFINE_integer('degree_gate', 20, 'Number of units for the MLP degree gate in GNN')

# options
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 50, 'how many epochs to train after last best validation')
flags.DEFINE_integer('split_idx', 0, 'Dataset split (Total:10) 0-9')
flags.DEFINE_integer('use_kl_warmup', 1, 'Use a linearly increasing KL [0-1] coefficient -- see wu_beta in optimization.py')
flags.DEFINE_string('gpu','0','Which GPU to use. Leave blank to use None')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss.') #5e-4
flags.DEFINE_integer('mc_samples', 1, 'No. of MC samples for calculating gradients')
flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 100, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")

# Train on CPU (hide GPU) due to memory constraints
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

save_path_disk = "data/DLSM/data_models"

#Let's start time here
start_time = time.time()

def calculate_f1(y_true, y_pred):
    if FLAGS.sigmoid:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_true = np.argmax(y_true, axis = 1)
        y_pred = np.argmax(y_pred, axis = 1)
    
    f1_micro = f1_score(y_true, y_pred, average = "micro")
    f1_macro = f1_score(y_true, y_pred, average = "macro")
        
    return f1_micro, f1_macro

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, S = 2, size = None):

    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    
    val_pred = np.zeros(labels.shape)
    for s in range(S):
        outs = sess.run([model.node_pred], feed_dict = feed_dict_val)
        val_pred += outs[0]
    val_pred = val_pred / S
    f1_mic, f1_mac = calculate_f1(labels, val_pred)

    return f1_mic, f1_mac

def incremental_evaluate(sess, model, minibatch_iter, size, S = 5, test = False):

    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test = test)
        val_pred = np.zeros(batch_labels.shape)
        for s in range(S):
            outs = sess.run([model.node_pred], feed_dict = feed_dict_val)
            val_pred += outs[0]
        val_pred = val_pred / S
        val_preds.append(val_pred)
        labels.append(batch_labels)
        iter_num += 1
        
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_mic, f1_mac = calculate_f1(labels, val_preds)
    
    return f1_mic, f1_mac

# create_model 
def create_model(placeholders, adj_info, deg_info, features, num_classes):

    # Create model
    model = DGVAE(placeholders = placeholders, 
                     adj_info = adj_info, 
                     degrees = deg_info,
                     features = features,
                     num_classes = num_classes,
                     mc_samples = FLAGS.mc_samples,
                     identity_dim = FLAGS.identity_dim)

    # Optimizer
    with tf.compat.v1.name_scope('optimizer'):
        opt = Optimizer(model = model,
                        placeholders = placeholders, 
                        batch_size = placeholders['batch_size'],
                        epoch = placeholders['epoch'])

    return model, opt

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix
    log_dir += "/{model:s}_{lr:0.6f}/".format(model = FLAGS.model, lr = FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def train(placeholders, model, opt, features, sess, minibatch, adj_info_ph, deg_info_ph, name="single_fold"):
    
    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict = {adj_info_ph: minibatch.adj, deg_info_ph: minibatch.deg})
    saver = tf.compat.v1.train.Saver()

    total_steps = 0
    best_validation = 0.0

    train_adj_info = tf.compat.v1.assign(model.adj_info, minibatch.adj)
    val_adj_info = tf.compat.v1.assign(model.adj_info, minibatch.test_adj)
    train_deg_info = tf.compat.v1.assign(model.degrees, minibatch.deg)
    val_deg_info = tf.compat.v1.assign(model.degrees, minibatch.test_deg)
    
    # Train model
    time_start = time.time()
    for epoch in range(FLAGS.epochs):

        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        while not minibatch.end():
            
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict() # batch_size, batch1, batch2
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            feed_dict.update({placeholders['epoch']: epoch})
            
            # Training step
            outs = sess.run([opt.opt_op, opt.cost, opt.kl, model.z, model.node_pred], feed_dict = feed_dict)
            
            train_cost = outs[1]
            train_kl = outs[2]
            train_pred = outs[4]
            
            train_mic, train_mac = calculate_f1(labels, train_pred)
            
            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run([val_adj_info.op, val_deg_info.op])
                val_mic, val_mac = evaluate(sess, model, minibatch, size = FLAGS.validate_batch_size)
                sess.run([train_adj_info.op, train_deg_info.op])
            
                # Print results                
                t = time.time() - time_start
                time_start = time.time()
                #summary_writer.add_summary(outs[0], total_steps)
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_kl=", "{:.5f}".format(train_kl),
                      "train_f1_mic=", "{:.5f}".format(train_mic), 
                      "train_f1_mac=", "{:.5f}".format(train_mac),
                      "val_f1_mic=", "{:.5f}".format(val_mic), 
                      "val_f1_mac=", "{:.5f}".format(val_mac),
                      "time=", "{:.5f}".format(t))
            
                if val_mic > best_validation:
                        # save model
                        print ('Saving model')
                        saver.save(sess = sess, save_path = log_dir())
                        best_validation = val_mic
                        last_best_epoch = 0
                
            iter += 1
            total_steps += 1
        
        if last_best_epoch > FLAGS.early_stopping:
            break
        else:
            last_best_epoch += 1
    
    print("Optimization Finished!")
    
    # Testing
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict = {adj_info_ph: minibatch.adj, deg_info_ph: minibatch.deg})
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess = sess, save_path = log_dir())
    print ('Model restored')
    sess.run([val_adj_info.op, val_deg_info.op])
    test_f1_mic, test_f1_mac = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test = True)
    print('Test mirco averaged F1 score: ' + str(test_f1_mic))
    print('Test marco averaged F1 score: ' + str(test_f1_mac))
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("f1_micro={:.5f} f1_macro={:.5f}".format(test_f1_mic, test_f1_mac))

def main():
    
    print("Loading training data..")
    train_data = load_data('data/' + FLAGS.dataset)
    print("Done loading training data!")
    
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    labels = train_data[3]
    del train_data
    
    if isinstance(list(labels.values())[0], list):
        num_classes = len(list(labels.values())[0])
    else:
        num_classes = len(set(labels.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    # Define placeholders
    placeholders = {
            'labels' : tf.compat.v1.placeholder(tf.float32, shape = (None, num_classes), name = 'labels'),
            'batch' : tf.compat.v1.placeholder(tf.int32, shape = (None), name = 'batch'),
            'batch_size' : tf.compat.v1.placeholder(tf.int32, name = 'batch_size'),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
            'epoch': tf.compat.v1.placeholder(tf.int32),
    }
    
    print('Constructing minibatch iterator..')
    minibatch = NodeMinibatchIterator(G, 
                                      id_map,
                                      placeholders,
                                      labels,
                                      num_classes,
                                      batch_size = FLAGS.batch_size,
                                      max_degree = FLAGS.max_degree)
    print('Done constructing Minibatch iterator!')
    
    adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape = minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable = False, name = "adj_info")
    deg_info_ph = tf.compat.v1.placeholder(tf.float32, shape = minibatch.deg.shape)
    deg_info = tf.Variable(deg_info_ph, trainable = False, name = "deg_info")

    model, opt = create_model(placeholders, adj_info, deg_info, features, num_classes)
    config = tf.compat.v1.ConfigProto(log_device_placement = FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.compat.v1.Session(config = config)
    
    train(placeholders, model, opt, features, sess, minibatch, adj_info_ph, deg_info_ph)

if __name__ == '__main__':
    main()

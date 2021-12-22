from layers import GraphConvolution, Dense
import tensorflow as tf
from utils import sample_normal

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass
   
class DGVAE(Model):
    def __init__(self, placeholders, adj_info, degrees, features, num_classes, mc_samples = 1, identity_dim = 0, **kwargs):
        super().__init__(**kwargs)

        self.num_neighbors = [int(x) for x in FLAGS.num_neighbors.split('_')]
        self.adj_info = adj_info
        self.degrees = degrees
        
        if identity_dim > 0:
           self.embeds = tf.compat.v1.get_variable("node_embeddings", [adj_info.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.inputs = self.embeds
            self.input_dim = identity_dim
        else:
            self.inputs = tf.Variable(tf.constant(features, dtype = tf.float32), trainable = False)
            self.input_dim = features.shape[1]
            if not self.embeds is None:
                self.inputs = tf.concat([self.embeds, self.inputs], axis = 1)
                self.input_dim += identity_dim
        self.nodes = placeholders['batch']
        self.batch_size = placeholders['batch_size']
        self.encoder_layers = [int(x) for x in FLAGS.encoder.split('_')]
        self.decoder_layer = FLAGS.decoder
        self.num_encoder_layers = len(self.encoder_layers)
        self.num_classes = num_classes
        self.dropout = placeholders['dropout']
        self.S = mc_samples #No. of MC samples
        
        self.build()

    def get_regualizer_cost(self, regularizer):

        regularization = 0
        #regularization += self.last_layer.apply_regularizer(regularizer)
        
        for layer in self.layers:
            regularization += regularizer(layer.vars['weights'])# * FLAGS.weight_decay

        return regularization
    
    def uniform_neighbor_sampler(self, inputs):
        """
        Uniformly samples neighbors.
        Assumes that adj lists are padded with random re-sampling
        """
        ids, num_samples = inputs
        neighbors = tf.nn.embedding_lookup(self.adj_info, ids) 
        neighbors = tf.transpose(tf.random.shuffle(tf.transpose(neighbors)))
        samples = tf.slice(neighbors, [0, 0], [-1, num_samples])
        
        return samples
    
    def sample(self, nodes, degrees, batch_size = None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.
        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        if batch_size is None:
            batch_size = self.batch_size
        samples = [nodes]
        support_deg = [tf.nn.embedding_lookup(degrees, nodes)]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(self.num_encoder_layers):
            t = self.num_encoder_layers - k - 1
            support_size *= self.num_neighbors[t]
            support_sizes.append(support_size)
            neighbors = self.uniform_neighbor_sampler((samples[k], self.num_neighbors[t]))
            neighbors = tf.reshape(neighbors, [support_size * batch_size,])
            samples.append(neighbors)
            
            degrees_nei = tf.nn.embedding_lookup(degrees, neighbors)
            support_deg.append(degrees_nei)

        return samples, support_sizes, support_deg
    
    def aggregate(self, inputs, degrees, support_sizes, layer, gc, batch_size = None):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            inputs: the input features for each sample of various hops away.
            sample_indices: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            degrees: a list of degrees for each sample of various hops away.
                Length is the number of layers + 1.
            support_sizes: the number of nodes to gather information from for each layer.
            layer: current layer
            gc: the graph convolution network to aggregate hidden representations
        Returns:
            a list of hidden representations for each sample of various hops away at the current layer
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        h_hop = []
        for hop in range(self.num_encoder_layers - layer):
            nei_dim = [batch_size * support_sizes[hop], # batch size
                       self.num_neighbors[self.num_encoder_layers - hop - 1], # number of neighbors
                       -1] # dimension of input hidden representations
            deg_dim = [batch_size * support_sizes[hop], # batch size
                       self.num_neighbors[self.num_encoder_layers - hop - 1], # number of neighbors
                       1]
            h = gc((inputs[hop], # self
                    tf.reshape(inputs[hop + 1], nei_dim), # neighbors
                    tf.reshape(degrees[hop + 1], deg_dim))) # deg_nei
            h_hop.append(h)
        
        return h_hop
    
    def _build(self):

        print('Build Network...')

        samples, support_sizes, support_deg = self.sample(self.nodes, self.degrees)
        inputs = [tf.nn.embedding_lookup(self.inputs, node_samples) for node_samples in samples]
        # This selection is questionable. May not be much of effect in reality
        act = tf.nn.tanh
        
        #######################################################################
        # construct GNN and generate variational parameters
        self.layers = []
        for idx, encoder_layer in enumerate(self.encoder_layers):
            
            if idx == 0:
                gc_input = GraphConvolution(input_dim = self.input_dim,
                                            output_dim = encoder_layer,
                                            act = act,
                                            dropout = self.dropout,
                                            name = 'conv_input_' + str(idx),
                                            logging = self.logging)
                self.layers.append(gc_input)
                
                h = self.aggregate(inputs = inputs,
                                   degrees = support_deg, 
                                   support_sizes = support_sizes, 
                                   layer = idx, 
                                   gc = gc_input)
                
            elif idx == self.num_encoder_layers - 1:
                
                gc_mean = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                           output_dim = encoder_layer,
                                           act = lambda x: x,
                                           dropout = self.dropout,
                                           name = 'conv_mean_' + str(idx),
                                           logging = self.logging)
                self.layers.append(gc_mean)
                
                h_mean = self.aggregate(inputs = h,
                                        degrees = support_deg, 
                                        support_sizes = support_sizes, 
                                        layer = idx, 
                                        gc = gc_mean)
                
                gc_std = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                          output_dim = encoder_layer,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          name = 'conv_std_' + str(idx),
                                          logging = self.logging)
                self.layers.append(gc_std)
                
                h_std = self.aggregate(inputs = h,
                                       degrees = support_deg, 
                                       support_sizes = support_sizes, 
                                       layer = idx, 
                                       gc = gc_std)
                    
                self.h = [h_mean[0], h_std[0]]
                
            else:
                gc_h = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                        output_dim = encoder_layer,
                                        act = act,
                                        dropout = self.dropout,
                                        name = 'conv_h_' + str(idx),
                                        logging = self.logging)
                self.layers.append(gc_h)
                
                h = self.aggregate(inputs = h,
                                   degrees = support_deg, 
                                   support_sizes = support_sizes, 
                                   layer = idx, 
                                   gc = gc_h)

        # get variational parameters
        z_mean = self.h[0]
        z_std = self.h[1]
        self.posterior_theta_param = [z_mean, z_std]

        #######################################################################
        # generate MC samples and predict node labels
        self.z_list = []
        self.pred_logit_list = []
        self.node_pred_list = []
        
        for k in range(self.S):
            
            # draw node representations from Normal variational distributions
            z = sample_normal(self.posterior_theta_param[0], self.posterior_theta_param[1]) # N * K
            
            node_pred_layer = Dense(self.encoder_layers[-1], self.decoder_layer, self.num_classes, name = 'node_predict')
            self.last_layer = node_pred_layer
            # predict node labels
            pred_logit = node_pred_layer(z)
            # use sigmoid if nodes can belong to multiple classes
            if FLAGS.sigmoid:
                node_pred = tf.nn.sigmoid(pred_logit)
            else:
                node_pred = tf.nn.softmax(pred_logit)
            
            self.z_list.append(z)
            self.pred_logit_list.append(pred_logit)
            self.node_pred_list.append(node_pred)
            
        self.z = tf.reduce_mean(self.z_list, axis = 0)
        self.node_pred = tf.reduce_mean(self.node_pred_list, axis = 0)
        #######################################################################

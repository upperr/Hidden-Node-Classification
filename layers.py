from initializations import weight_variable_glorot
import tensorflow as tf

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class GraphConvolution(Layer):
    """Degree-gated GNN
    """
    def __init__(self, input_dim, output_dim, act = tf.nn.relu, dropout = 0., **kwargs):
        super().__init__(**kwargs)
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name = "bias")
            
        with tf.compat.v1.variable_scope(self.name + '_attention_vars'):
            self.vars['phi_attention'] = weight_variable_glorot(2 * output_dim, 1, name = 'weights_attention')
            #self.vars['bias_attention'] = tf.Variable(tf.zeros((1)), name = "bias_attention")

        with tf.compat.v1.variable_scope(self.name + '_degree_vars'):
            self.vars['weight_degree1'] = weight_variable_glorot(1, FLAGS.degree_gate, name = "weight_degree1")
            self.vars['bias_degree1'] = tf.Variable(tf.zeros((1)), name = "bias_degree1")
            self.vars['weight_degree2'] = weight_variable_glorot(FLAGS.degree_gate, 1, name = "weight_degree2")
            self.vars['bias_degree2'] = tf.Variable(tf.zeros((1)), name = "bias_degree2")
            
        with tf.compat.v1.variable_scope(self.name + '_merge_vars'):
            self.vars['weight_self'] = weight_variable_glorot(1, 1, name = 'weight_self')
            self.vars['weight_neighbor'] = weight_variable_glorot(1, 1, name = 'weight_neighbor')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout = dropout

    def _call(self, inputs):
        x_self, x_neighbor, deg_neighbor = inputs
        num_nodes = tf.shape(x_neighbor)[0]
        num_neighbors = tf.shape(x_neighbor)[1]
        
        x_self = tf.nn.dropout(x_self, self.dropout) # dim: batch_size × K^l
        x_neighbor = tf.nn.dropout(x_neighbor, self.dropout) # dim: batch_size × num_neighbor × K^l
        x_self_transformed = tf.matmul(x_self, self.vars['weights']) + self.vars['bias'] # dim: batch_size × K^(l + 1)
        x_neighbor_transformed = tf.reshape(tf.matmul(tf.reshape(x_neighbor, [-1, self.input_dim]), self.vars['weights']) + self.vars['bias'], [num_nodes, num_neighbors, self.output_dim])
        
        # get the degree gate through a 3-layer MLP
        gate_degree = tf.nn.sigmoid((tf.matmul(tf.nn.sigmoid(tf.matmul(deg_neighbor, self.vars['weight_degree1']) + self.vars['bias_degree1']), 
                                               self.vars['weight_degree2']) + self.vars['bias_degree2']))
        x_neighbor_filtered = tf.multiply(x_neighbor_transformed, gate_degree) # dim: batch_size × num_neighbor × K^(l + 1)

        # get self and neighbor representations for attention mechanism
        x_self_expanded = tf.tile(tf.expand_dims(x_self_transformed, 1), [1, num_neighbors, 1]) # dim: batch_size × num_neighbor × K^(l + 1)
        x_concat = tf.concat([x_neighbor_transformed, x_self_expanded], axis = 2) # dim: batch_size × num_neighbor × 2K^(l + 1)

        # calculate attention weight
        attention = tf.reshape(tf.nn.leaky_relu(tf.matmul(tf.reshape(x_concat, [-1, 2 * self.output_dim]), 
                                                          self.vars['phi_attention']), alpha = 0.2), [-1, num_neighbors, 1]) # dim: batch_size × num_neighbor × 1
        attention = tf.nn.softmax(attention)
        
        # neighbor aggregation
        neighbor_info = tf.reduce_sum(tf.multiply(attention, x_neighbor_filtered), axis = 1) # dim: batch_size × K^(l + 1)
        output = self.act(self.vars['weight_self'] * x_self_transformed + self.vars['weight_neighbor'] * neighbor_info) #+ self.vars['bias_merge'])
        
        return output

    def apply_regularizer(self, regularizer):
        return 0
        #return regularizer(self.vars['weights'])

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0., act = tf.nn.relu, bias = True, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights1'] = tf.compat.v1.get_variable('weights1', shape = (input_dim, hidden_dim),
                                                              dtype = tf.float32,
                                                              initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale = 1.0, mode = "fan_avg", distribution = "uniform"),
                                                              regularizer = tf.keras.regularizers.l2(0.5 * (FLAGS.weight_decay)))
            self.vars['weights2'] = tf.compat.v1.get_variable('weights2', shape = (hidden_dim, output_dim),
                                                              dtype = tf.float32,
                                                              initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale = 1.0, mode = "fan_avg", distribution = "uniform"),
                                                              regularizer = tf.keras.regularizers.l2(0.5 * (FLAGS.weight_decay)))
            if self.bias:
                self.vars['bias1'] = tf.Variable(tf.zeros((hidden_dim)), name = "bias1")
                self.vars['bias2'] = tf.Variable(tf.zeros((output_dim)), name = "bias2")

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights1'])
        # bias
        if self.bias:
            output += self.vars['bias1']
        
        output = self.act(output)
        output = tf.matmul(output, self.vars['weights2'])
        if self.bias:
            output += self.vars['bias2']

        return output
    
    def apply_regularizer(self, regularizer):
        return 0
        #return regularizer(self.vars['weights'])

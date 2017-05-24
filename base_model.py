import os
import sys
import numpy as np
import tensorflow as tf
from utilities import *

try:
  import cPickle as pickle
except:
  import pickle

from tensorflow.python.ops import control_flow_ops

class BaseModel(object):
  def __init__(self, network_architecture, name=None, dir=None, load_path=None, debug_mode=0, seed=100):
    
    # Misc book-keeping parameters
    self._debug_mode = debug_mode
    self._dir = dir
    self._name = name
    self._load_path = load_path
    
    # Specify default network. Generic enough for any model.
    self.network_architecture = dict(model_type = 'DNN',   # Define Model Type
                                     n_in = 33,                   # data input dimension
                                     n_hid = 100,                 # hidden layer size
                                     n_layers = 1,                # number of hidden layers
                                     n_out = 1,                   # output dimensionality
                                     activation_fn = tf.nn.relu,  # Activation Fucntion
                                     output_fn = tf.nn.relu,      # Output function
                                     initializer = tf.contrib.layers.xavier_initializer, # Define Initializer
                                     L2=4e-7,                     # L2 weight decay.... Necessary??? Training Param?
                                     BN=False)                    # Batch Normalization
    
    # Add check that network architecture has all the necessary parameters???
    # If new model, need to update architecture
    if load_path == None:
      if network_architecture != None and len(network_architecture) > 0:
        for param in network_architecture:
          self.network_architecture[param[0]]=param[1]
    else:
      # Load data (deserialize) architecture from path
      arch_path = os.path.join(load_path, 'net_arch.pickle')
      with open(arch_path, 'rb') as handle:
        self.network_architecture = pickle.load(handle) 
    
    if (os.path.isfile(os.path.join(self._dir, 'LOG.txt')) or os.path.isfile(os.path.join(self._dir, 'weights.ckpt')) or os.path.isfile(os.path.join(self._dir, 'net_arch.pickle'))) and load_path == None:
      print 'Model exists in directory - exiting.'
      sys.exit()
    if load_path == None:
      with open(os.path.join(self._dir, 'LOG.txt'), 'w') as f:
        f.write('Creating Grader Model with configuration:\n')
        f.write('----------------------------------------------------------\n')
        for key in sorted(self.network_architecture.keys()):
          f.write(key+': '+str(self.network_architecture[key])+'\n')
        f.write('----------------------------------------------------------\n')

    # Parameters for training
    self._seed = seed
    self.initializer = self.network_architecture['initializer'] 
    
    # Tensorflow graph bookeeping
    self._graph = tf.Graph()
    # Construct Graph
    with self._graph.as_default():
      tf.set_random_seed(self._seed)
      self.sess = tf.Session()
  
  def save(self):
    """ Saves model and parameters to self._dir """
    with self._graph.as_default():
      path = os.path.join(self._dir, 'weights.ckpt')
      self._saver.save(self.sess, path)
    
    # Pickle network architecture into a file.
    path = os.path.join(self._dir, 'net_arch.pickle')
    with open(path, 'wb') as handle:
      pickle.dump(self.network_architecture, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
  def load_variables(self, load_scope, new_scope, load_path=None, trainable=False):
      # Restore parameters to DDN we are sampling from...
      if trainable:
        model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*"+new_scope+".*")
      else:
        model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, ".*"+new_scope+".*")
      dict={}
      for model_var in model_variables:
          print model_var.op.name, model_var.op.name.replace(new_scope, load_scope)
          dict[model_var.op.name.replace(new_scope, load_scope)]=model_var
      sampling_saver = tf.train.Saver(dict)
      if load_path == None:
        param_path = os.path.join(self._load_path, 'weights.ckpt')
      else:
        param_path = os.path.join(load_path, 'weights.ckpt')
      sampling_saver.restore(self.sess, param_path)
  
  def _read_config(self):
    pass

  def _split_test_eval(self, data_list, valid_size):
      """ Helper Function 
      Args:
        data_list: list of numpy arrays
        valid_size : int
      Returns:
        returns two lists of np arrays for eval and train data
      """
      evl_data_list = []
      trn_data_list = []
      for data in data_list:
        try: 
          evl_data_list.append(data[:valid_size, :])
          trn_data_list.append(data[valid_size:, :])
        except:
          evl_data_list.append(data[:valid_size])
          trn_data_list.append(data[valid_size:])
      return evl_data_list, trn_data_list
      
  def _add_loss_summaries(self, total_loss):
    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Expects existence of collection 'losses'
    Args:
      total_loss: Total loss tensor.
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.scalar_summary(l.op.name +' (raw)', l)
      tf.scalar_summary(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

  def _activation_summary(self, x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))
      
  def _variable_on_gpu(self, name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on GPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/gpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

  def _variable_with_weight_decay(self, name, shape, seed=100, wd=0.000, trainable=True):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      seed: int
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
     
    var = self._variable_on_gpu(name, shape, initializer=self.initializer(seed), trainable=trainable)
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var

  def _queue_data_initializer(self, data):
    
    with tf.device('/cpu:0'):
      if data.dtype == np.float32:
        dtype = tf.float32
      elif data.dtype == np.int32:
        dtype = tf.int32
      
      data_initializer =  tf.placeholder(dtype=data.dtype, shape=data.shape)
      data = tf.Variable(data_initializer, trainable=False, collections=[])
      return data, data_initializer
  
  def _construct_queue(self, data_list, batch_size, capacity=100000, num_threads=1):
    with tf.device('/cpu:0'):

      self.data_variables = []
      self.data_initializers = []
      for data in data_list:
        data_var, data_init = self._queue_data_initializer(data)
        self.data_variables.append(data_var)
        self.data_initializers.append(data_init)
        
      data_queue = tf.train.slice_input_producer(self.data_variables, shuffle=True, seed=self._seed, capacity = capacity, name='queue_1')
      self.data_queue_list = tf.train.batch(data_queue, capacity = capacity, num_threads=num_threads, batch_size=batch_size, name='queue_2')  

  def _train_queue_init(self, data_list):
    with tf.device('/cpu:0'):
      for data_var, init, data in zip(self.data_variables, self.data_initializers, data_list):
        self.sess.run(data_var.initializer, feed_dict={init: data}) 

  def _construct_mse_cost(self, targets, predictions, is_training=False):
    
    cost = tf.reduce_mean((targets -  predictions)**2, name='total_squared_error_per_batch')
    
    if self._debug_mode > 1:
      tf.scalar_summary('MSE', cost)
    
    if is_training:
      tf.add_to_collection('losses', cost)
      # The total loss is defined as the target loss plus all of the weight
      # decay terms (L2 loss).
      total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
      return cost, total_cost
    else:
      return cost
    
  def _construct_xent_cost(self, targets, logits, predictions=None, is_training=False):
    print 'Constructing XENT cost'
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name='total_xentropy_per_batch'))

    if self._debug_mode > 1:
      tf.scalar_summary('XENT', cost)
    
    if is_training:
      tf.add_to_collection('losses', cost)
      # The total loss is defined as the target loss plus all of the weight
      # decay terms (L2 loss).
      total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
      return cost, total_cost
    else:
      return cost
    
  def _construct_mbr_cost(self, targets, probabilities, is_training=False):
    print 'Constructing MBR cost'
    space= tf.expand_dims(tf.linspace(0.0, 6.0, 7), dim=1)
    targets=tf.cast(targets, dtype=tf.float32)
    weights = tf.transpose((space-targets)**2)
    mbr=tf.reduce_sum(weights*probabilities, reduction_indices=1)
    cost = tf.reduce_mean(mbr)

    if self._debug_mode > 1:
      tf.scalar_summary('MBR', cost)
    
    if is_training:
      tf.add_to_collection('losses', cost)
      # The total loss is defined as the target loss plus all of the weight
      # decay terms (L2 loss).
      total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
      return cost, total_cost, weights
    else:
      return cost

  def _construct_train_op(self, total_cost, optimizer, lr, lr_decay, batch_size,
          global_step, 
          num_examples, variable_scope_name=None):
    
    # Variables that affect learning rate.
    decay_steps = num_examples / batch_size
    
    # Decay the learning rate exponentially based on the number of steps.
    if lr != None:
      lr = tf.train.exponential_decay(learning_rate=lr,
                                      global_step=global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=lr_decay,
                                      staircase=False)
      
    # Compute gradients.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    if update_ops: 
      updates = tf.group(*update_ops) 
      total_cost = control_flow_ops.with_dependencies([updates], total_cost)
    #with tf.control_dependencies([updates]):
    if lr != None:
      opt = optimizer(learning_rate=lr, use_locking=True)
    else:
      opt = optimizer(learning_rate=self.lr, use_locking=True)
    if variable_scope_name is None:
      tvars = tf.trainable_variables()
    else:
      tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*"+variable_scope_name+'.*')
    grads_and_vars = opt.compute_gradients(total_cost, tvars, gate_gradients=opt.GATE_GRAPH)
    grads_and_vars = [(tf.clip_by_norm(gv[0], 10.0), gv[1]) for gv in grads_and_vars]
    
  

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
  
    # Add histograms for trainable variables.
    if self._debug_mode > 0:
      tf.scalar_summary('learning_rate', lr)
      
      for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    
      # Add histograms for gradients.
      for grad, var in grads_and_vars:
        if grad is not None:
          tf.histogram_summary(var.op.name + '/gradients', grad)
  
      # Track the moving averages of all trainable variables.
      #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
      #variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='train')
    return train_op
  

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.queues import QueueRunners
import numpy as np
import scipy as sp
import sys
import os
import time

from matplotlib import pyplot as plt
from math import ceil

from base_model import BaseModel

class RNNLM(BaseModel):
  def __init__(self, network_architecture=None, name=None, dir=None, load_path=None, debug_mode=0, seed=100):
      
    BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, dir=dir, load_path=load_path, debug_mode=debug_mode)
  
    with self._graph.as_default():
      with tf.variable_scope('input') as scope:
        self._input_scope = scope
        self.x = tf.placeholder(tf.int32, [None, None])
        self.seqlens = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.dropout =  tf.Variable(tf.ones(dtype=tf.float32, shape=[]), trainable=False, name='dropout_rate')
        self.batch_size = tf.placeholder(tf.int32, [])
        
      with tf.variable_scope('model') as scope:
        self._model_scope = scope
        self.predictions, self.logits = self._construct_network(input=self.x, 
                                                                seqlens=self.seqlens,
                                                                batch_size=self.batch_size, 
                                                                WD=self.network_architecture['L2'], 
                                                                keep_prob=self.dropout)
        
      # Not sure if this is even really necessary....
      init = tf.initialize_all_variables()
      self.sess.run(init)
      
      self._saver = tf.train.Saver(tf.all_variables())
      #If necessary, restore model from previous
      if load_path != None:
        arch_path = os.path.join(load_path, 'weights.ckpt')
        with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
          f.write('Restoring Model paratemters from: '+arch_path+'\n')
        self._saver.restore(self.sess, arch_path)

  def _construct_network(self,
                         input,
                         seqlens,
                         batch_size,
                         WD=1e-6,
                         keep_prob=1.0):
    """ Construct RNNLM network 
    Args:
      ?
    Returns:
      predictions, logits
    """
    initializer = self.network_architecture['initializer']
    
    with tf.variable_scope('Embeddings') as scope:
      embedding = self._variable_with_weight_decay("word_embedding", [self.network_architecture['n_in'], self.network_architecture['n_hid']], self._seed, WD)
      self.word_embeddings = embedding
      inputs = tf.nn.embedding_lookup(embedding, input, name='embedded_data')
      
    with tf.variable_scope('RNN', initializer=initializer(self._seed)) as scope:
      # Create RNNLM graph
#       cell =tf.contrib.rnn.GRUCell(num_units=self.network_architecture['n_hid'], 
#                                           activation=tf.nn.tanh)
      cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_hid'], 
                                          activation=tf.nn.tanh, 
                                          forget_bias=1.0, 
                                          state_is_tuple=True)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
      #cell = tf.contrib.rnn.MultiRNNCell([cell]*self.network_architecture['n_layers'])
      initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      outputs, state = tf.nn.dynamic_rnn(cell, 
                                         inputs, 
                                         sequence_length=seqlens, 
                                         dtype=tf.float32,
                                         initial_state=initial_state,
                                         parallel_iterations=32)
      
    with tf.variable_scope('softmax_output') as scope:
      weights = self._variable_with_weight_decay(name="weights", 
                                                 shape=[self.network_architecture['n_hid'], 
                                                        self.network_architecture['n_out']], 
                                                 seed=self._seed, 
                                                 wd=0.0)
      biases  = self._variable_on_gpu('biases', [self.network_architecture['n_out']], tf.constant_initializer(0.1))
      logits = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, self.network_architecture['n_hid']]), weights), [batch_size, -1, self.network_architecture['n_out']]) + biases
      predictions =tf.nn.softmax(logits, dim=-1, name='predictions')

    return predictions, logits
  
  def _construct_cost(self, targets, logits, seqlens, maxlen, is_training=False):
    #logits = tf.reshape(tf.concat(1, logits), [-1, self.network_architecture['n_out']])
    
    # Create a mast to zero out xent for inputs beyond sequence length
    mask = tf.sequence_mask(seqlens, maxlen, dtype=tf.float32) # Very cool!
    
#     N_OOS = self.network_architecture['n_in']-self.network_architecture['n_out']
#     condition = tf.equal(tf.convert_to_tensor(targets, dtype=tf.int32), 
#                          tf.convert_to_tensor(0, dtype=tf.int32))
#     norm_OOS = tf.log(tf.where(condition=condition, 
#                                x=tf.ones_like(targets, dtype=tf.float32, optimize=True)*N_OOS, 
#                                y=tf.ones_like(targets, dtype=tf.float32, optimize=True)))
    
    
      
    # Calculate sparse softmax xent for each example, apply mask, sum for all sequences and normalize by sequence length
    #if is_training:
    cost = tf.reduce_sum(mask*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name='xentropy_per_example'))
    #else:
    #  cost = tf.reduce_sum(mask*(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name='xentropy_per_example')-norm_OOS))
    # Compute Batch Mean xent
    
    if self._debug_mode > 1:
      tf.scalar_summary('XENT', cost)
    
    if is_training:
      cost = cost/tf.reduce_sum(tf.cast(seqlens, dtype=tf.float32))
      tf.add_to_collection('losses', cost)
      # The total loss is defined as the target loss plus all of the weight
      # decay terms (L2 loss).
      total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
      return cost, total_cost
    else:
      return cost
    
  def fit(self, valid_data_list, trn_data_list, learning_rate=1e-2, lr_decay=0.8, batch_size=50, dropout=1.0, optimizer=tf.train.AdamOptimizer, n_epochs=30):
    with self._graph.as_default():   
      #Compute number of training examples and batch size
      n_examples = trn_data_list[1].shape[0]
      n_batches = n_examples/batch_size
      
      #If some variables have been initialized - get them into a set
      temp = set(tf.all_variables())
      lr = learning_rate
      # Define Global step for training
      global_step = tf.Variable(0, trainable=False, name='global_step')
      
      self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False)
      
      # Set up inputs
      with tf.variable_scope(self._input_scope, reuse=True) as scope:
        # Construct training data queues
        batch_size_tensor = tf.constant(value=batch_size, dtype=tf.int32, shape=[])
        self._construct_queue(trn_data_list, batch_size=batch_size, capacity=n_examples)
        self._train_queue_init(trn_data_list)
      
      # Construct Training model
      with tf.variable_scope(self._model_scope, reuse=True) as scope:          
        self.trn_predictions, self.trn_logits = self._construct_network(input=self.data_queue_list[1], 
                                                                        batch_size=batch_size_tensor,
                                                                        seqlens=self.data_queue_list[2],
                                                                        WD=self.network_architecture['L2'],
                                                                        keep_prob=1.0)
      
      # Construct the Cost Graph and Train operation
      trn_cost, total_cost = self._construct_cost(targets=self.data_queue_list[0], 
                                                  logits=self.trn_logits, 
                                                  maxlen=20, 
                                                  seqlens=self.data_queue_list[2], 
                                                  is_training=True)
      print valid_data_list[2]-1
      evl_cost = self._construct_cost(targets=self.y, logits=self.logits, maxlen=np.max(valid_data_list[2]), seqlens=self.seqlens)
      train_op = self._construct_train_op(total_cost, optimizer, None, None, batch_size, global_step, n_examples)
      
      #Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
      init=tf.initialize_variables(set(tf.all_variables()) - temp)
      self.sess.run(init)
      
      #Create Summary ops and summary writer ### UNUSED!!! ###
      if self._debug_mode > 1:
        summary_op = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(self._dir, self._graph)
      
      # Start the Queue Runner
      tf.train.start_queue_runners(sess=self.sess)
      
      # Update Log with training details
      with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
        format_str = ('Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nOptimizer: %s')
        f.write(format_str % (learning_rate, lr_decay, batch_size,  str(optimizer))+'\n\n')
      
      format_str = ('Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, (%.1f examples/sec; %.3f ' 'sec/epoch')
      print n_batches
      start_time = time.time()
      old_eval_loss = 1000000.0
      decay = False
      for epoch in xrange(1, n_epochs+1):
        loss = 0.0
        batch_time = time.time()
        
        for batch in xrange(n_batches):
          #t = time.time()
          _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout : dropout})
          #print time.time() -t
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          loss+=loss_value
        
        duration = time.time() - batch_time
        loss/=n_batches
        
        examples_per_sec = batch_size / duration
        sec_per_epoch = float(duration)
        
        eval_loss=0.0
        for i in xrange(len(valid_data_list[0])/50):
          eval_loss += self.sess.run(evl_cost, feed_dict={self.y : valid_data_list[0][i*50:(i+1)*50], self.x : valid_data_list[1][i*50:(i+1)*50], self.seqlens : valid_data_list[2][i*50:(i+1)*50], self.batch_size : 50})
        eval_loss /= np.sum(valid_data_list[2])
        if (eval_loss >= old_eval_loss) or decay == True:
          lr /=2.0
          assign_op = self.lr.assign(lr)
          self.sess.run(assign_op)
          decay = True
        old_eval_loss = eval_loss
        print 'Eval PPL', np.exp(eval_loss)
        with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
          f.write(format_str % (epoch, loss, eval_loss, examples_per_sec, sec_per_epoch)+'\n')
 
        print (format_str % (epoch, loss, eval_loss,  examples_per_sec, sec_per_epoch))
        self.save()
      
      duration = time.time() - start_time
      with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
          format_str = ('Training took %.3f sec')
          f.write('\n'+format_str % (duration)+'\n')
          f.write('----------------------------------------------------------\n')
      print (format_str % (duration))
  
  def predict(self, X):
    with self._graph.as_default():
      for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print var.op.name
      test_loss=0.0
      evl_cost = self._construct_cost(targets=self.y, logits=self.logits, maxlen=np.max(X[2]), seqlens=self.seqlens)
      for i in xrange(len(X[0])/50):
        test_loss += self.sess.run(evl_cost, feed_dict={self.y : X[0][i*50:(i+1)*50], self.x : X[1][i*50:(i+1)*50], self.seqlens : X[2][i*50:(i+1)*50], self.batch_size : 50})
      test_loss /= np.sum(X[2])
      print 'Test PPL', np.exp(test_loss)
  
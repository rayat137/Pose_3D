# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

#import bnlstm
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import forward_kinematics as fk
#import rotmat_projection as rproj
# from epnp import epnp_tf_batch as epnp # FIXME make compatible with python 2.7
import os

class TemporalModel(object):
  """Sequence-to-sequence model with attention.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder.
  """

  def __init__(self,
               sgd,
               linear_size,
               batch_size,
               learning_rate,
               summaries_dir,
               dim_to_use_3d,
               data_mean,
               data_std,
               dim_to_ignore_3d,
               camera_frame, #Whether to estimate 3D locations in camera coordinate system
               seqlen,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """

    # Whether to randomly flip 2d points
    self.IM_W = 1000 # pixels in width
    self.IM_H = 1002 # pixels in height

    # FIXME doing experiments without root
    #self.HUMAN_2D_SIZE = 16 * 2 if use_prediction else 17 * 2
    self.HUMAN_2D_SIZE =  16 * 2
    self.HUMAN_3D_SIZE = 16 * 3

    self.input_size  = self.HUMAN_2D_SIZE
    self.output_size = self.HUMAN_3D_SIZE
    self.isTraining = tf.placeholder(tf.bool,name="isTrainingflags")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter( os.path.join(summaries_dir, 'train' ))
    self.test_writer  = tf.summary.FileWriter( os.path.join(summaries_dir, 'test' ))

    self.linear_size = linear_size
    self.batch_size  = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype, name="learning_rate")
    self.global_step = tf.Variable(0, trainable=False, name="global_step")
    self.learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,100000,0.96)
    #self.learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,1000000,0.96) # Use 1 million instead of 100K
    self.seqlen = seqlen
    self.dim_to_use = dim_to_use_3d
    self.mean =  data_mean
    self.std =   data_std
    self.dim_to_ignore = dim_to_ignore_3d


    # === Create the RNN that will keep the state ===
    print('linear_size = {0}'.format( linear_size ))

    # === Transform the inputs ===
    with vs.variable_scope("inputs"):

      enc_in  = tf.placeholder(dtype, shape=[None, seqlen, self.input_size], name="enc_in")
      dec_out = tf.placeholder(dtype, shape=[None, seqlen, self.output_size], name="dec_out")
      self.encoder_inputs  = enc_in
      self.decoder_outputs = dec_out
      #print(enc_in.get_shape,dec_out.get_shape)
      enc_in = enc_in[:,::-1,:]
      enc_in = tf.transpose(enc_in, [1, 0, 2])
      enc_in = tf.reshape(enc_in, [-1, self.input_size])
      enc_in = tf.split(enc_in, seqlen,axis=0)

      ###NOTE: TRYING instead of ones
      dec_in =tf.ones([self.batch_size,5,self.output_size])
      dec_in = tf.transpose(dec_in, [1, 0, 2])
      dec_in = tf.reshape(dec_in, [-1, self.output_size])
      dec_in = tf.split(dec_in,seqlen,axis=0)
      #enc_in = tf.cond(self.isTraining, lambda: enc_in + tf.random_normal(shape=tf.shape(enc_in),mean=0,stddev=0.1),lambda:enc_in)
    # === Create the linear + relu convos ===

    def lf(prev, i): # function for self-fed loss
      return prev

    cell1 = tf.contrib.rnn.LayerNormBasicLSTMCell(linear_size,dropout_keep_prob=self.dropout_keep_prob)

    cell1 = tf.contrib.rnn.DropoutWrapper(cell1,input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)

    cell1 = tf.contrib.rnn.InputProjectionWrapper(cell1,linear_size)
    cell1 = tf.contrib.rnn.OutputProjectionWrapper(cell1,self.input_size)


    cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(linear_size,dropout_keep_prob=self.dropout_keep_prob)
    cell2 = tf.contrib.rnn.DropoutWrapper(cell2,input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)

    cell2 = tf.contrib.rnn.InputProjectionWrapper(cell2,linear_size)
    cell2 = tf.contrib.rnn.OutputProjectionWrapper(cell2,self.output_size)
    cell2 = tf.contrib.rnn.ResidualWrapper(cell2)

    enc_outputs = []
    state = cell1.zero_state(batch_size,dtype)
    enc_state = []
    for inputs in enc_in:
      out,state = cell1(inputs,state)
      enc_outputs.append(out)
      enc_state.append(state)

    outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder( dec_in, enc_state[-1], cell2, loop_function=lf )

    # Revert the output
    enc_outputs = tf.concat( enc_outputs, axis=0 )
    enc_outputs = tf.reshape( enc_outputs, [seqlen, -1, self.input_size] )
    enc_outputs = tf.transpose( enc_outputs, [1, 0, 2] )


    outputs = tf.concat( outputs, axis=0 )
    #print(outputs.get_shape)
    outputs = tf.reshape( outputs, [seqlen, -1, self.output_size] )
    #outputs = tf.reshape( outputs[:,48:], [seqlen, -1, self.output_size] )
    outputs = tf.transpose( outputs, [1, 0, 2] )

    weights_diff = tf.concat([tf.ones([self.batch_size,seqlen-1,3]),2.5*tf.ones([self.batch_size,seqlen-1,3]),
    2.5*tf.ones([self.batch_size,seqlen-1,3]),tf.ones([self.batch_size,seqlen-1,3]),
    2.5*tf.ones([self.batch_size,seqlen-1,3]),2.5*tf.ones([self.batch_size,seqlen-1,3]),
    tf.ones([self.batch_size,seqlen-1,15]),4*tf.ones([self.batch_size,seqlen-1,3]),
    4*tf.ones([self.batch_size,seqlen-1,3]),tf.ones([self.batch_size,seqlen-1,3]),
    4*tf.ones([self.batch_size,seqlen-1,3]),4*tf.ones([self.batch_size,seqlen-1,3])],axis=2)


    #Put a Greater weight to smoothness 2.5 instead of 0.5
    self.std = self.std[self.dim_to_use]
    self.std = tf.reshape(self.std,[1,1,self.output_size])
    self.std = tf.tile(self.std,[self.batch_size,seqlen,1])
    self.std = tf.cast(self.std,tf.float32)
    self.mean = self.mean[self.dim_to_use]
    self.mean = tf.reshape(self.mean,[1,1,self.output_size])
    self.mean = tf.tile(self.mean,[self.batch_size,seqlen,1])
    self.mean = tf.cast(self.mean,tf.float32)
    un_norm_dec_gt = tf.multiply(dec_out, self.std) + self.mean
    un_norm_out = tf.multiply(outputs, self.std) + self.mean

    diff_outputs = tf.reduce_mean(tf.multiply(weights_diff,tf.square(tf.subtract(un_norm_out[:,1:,:], un_norm_out[:,:-1,:]))))
    self.loss  = tf.reduce_mean(tf.square(tf.subtract(un_norm_dec_gt ,un_norm_out))) + 5 * diff_outputs

    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)
    self.outputs = outputs
    # Just to keep track of the loss in mm
    self.err_mm = tf.placeholder( tf.float32, name="error_mm" )
    self.err_mm_summary = tf.summary.scalar( "loss/error_mm", self.err_mm )

    # Gradients and SGD update operation for training the model.
    if sgd:
      opt = tf.train.GradientDescentOptimizer( self.learning_rate )
    else:
      opt = tf.train.AdamOptimizer( self.learning_rate )


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      # Update all the trainable parameters

      gradients = opt.compute_gradients(self.loss)

      # FIXME this should not be necessary if all the inputs have gradients
      # The gradients go crazy if one of them is None, remove that
      #max_gradient_norm = tf.global_norm(gradients)

      self.gradients = [[] if i==None else i for i in gradients]
      self.updates = opt.apply_gradients(gradients, global_step=self.global_step)

    #self.updates = opt.minimize(self.loss,global_step=self.global_step)
    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=50 )

  def step(self, session, encoder_inputs, decoder_outputs, dropout_keep_prob, isTraining=True):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use
      encoder_inputs: list of numpy vectors to feed as encoder inputs
      decoder_outputs: list of numpy vectors that are the expected decoder outputs
      dropout_keep_prob: [0,1) dropout keep probability
      forward_only: whether to do the backward step or only forward

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      mean squared error, and the outputs
    """
    input_feed = {self.encoder_inputs: encoder_inputs,
                    self.decoder_outputs: decoder_outputs,
                    self.isTraining: isTraining,
                    self.dropout_keep_prob: dropout_keep_prob}

    # Output feed: depends on whether we do a backward step or not.
    if isTraining:

      # Training step
      output_feed = [self.updates,       # Update Op that does SGD
                     #self.gradient_norms, # Gradient norm
                     #self.gradients,       # Gradient norm
                     self.loss,
                     self.loss_summary,
                     self.learning_rate_summary,
                     self.outputs]

      #output_feed = [self.updates]

      outputs = session.run( output_feed, input_feed )

      #for i in range( len(outputs) ):
      #  print( i, outputs[i] )

      #print( 1, outputs[1] )

      #print( outputs )
      return outputs[1], outputs[2], outputs[3], outputs[4] # Gradient norm, loss, summaries

    else:
      # Validation step, not on Ashesh's seeds
      output_feed = [self.loss, # Loss for this batch.
                   self.loss_summary,
                   self.outputs]

      outputs = session.run(output_feed, input_feed)
      #print("######################ROTMAT",session.run(outputs[3]))
      return outputs[0], outputs[1], outputs[2]  # No gradient norm

  def get_all_batches( self, data_x, data_y, camera_frame,  training=True):
    """
    Obtain a list of all the batches, randomly permutted
    Args
      data_x: 2d inputs
      data_y: 3d expected outputs
      camera_frame: whether the 3d data is in camera coordinates
      training: True if this is a training batch. False otherwise.

    Returns
      encoder_inputs, decoder_outputs

    """
    # Figure out how many frames we have
    n = 0
    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d

    encoder_inputs = []
    decoder_outputs = []
    # Put all the data into big arrays
    n_sequences = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh') and camera_frame) else key3d
      if training:
        random_start = np.random.randint(self.seqlen-1) ##NOTE:IT WAS 10 initially
        pose_2d_list = data_x[ key2d ][random_start:,:]
        pose_3d_list = data_y[ key3d ][random_start:,:]

      else:
        pose_2d_list = data_x[ key2d ][:,:]
        pose_3d_list = data_y[ key3d ][:,:]
      n2d = pose_2d_list.shape[0]
      n_extra  = n2d % self.seqlen
      if n_extra>0:
        pose_2d_list  = pose_2d_list[:-n_extra, :]
        pose_3d_list = pose_3d_list[:-n_extra, :]
      n2d = pose_2d_list.shape[0]
      pose_2d_sliding = []
      pose_3d_sliding = []

      for i in range(n2d-self.seqlen+1):
        pose_2d_sliding.append(pose_2d_list[i:i+self.seqlen,:])
        pose_3d_sliding.append(pose_3d_list[i:i+self.seqlen,:])

      pose_2d_list = np.stack(pose_2d_sliding)
      pose_3d_list = np.stack(pose_3d_sliding)

      n_splits = n2d-self.seqlen+1
      encoder_inputs.append(pose_2d_list)
      decoder_outputs.append(pose_3d_list)
      n_sequences = n_sequences + n_splits

    # Randomly permute the sequences
    encoder_inputs = np.vstack(encoder_inputs)
    decoder_outputs = np.vstack(decoder_outputs)

    if training:
      idx = np.random.permutation( n_sequences )
      #print("###SHAPE OF idx",idx.shape)
      encoder_inputs  = encoder_inputs[idx,:,:]
      decoder_outputs = decoder_outputs[idx,:,:]

    # MAke the number of examples equally divide the batch size
    n_extra  = n_sequences % self.batch_size
    if n_extra > 0: # Otherwise we get nothing
      encoder_inputs  = encoder_inputs[:-n_extra, :,:]
      decoder_outputs = decoder_outputs[:-n_extra, :,:]
    n_batches = n_sequences // self.batch_size
    encoder_inputs  = np.split( encoder_inputs, n_batches )
    decoder_outputs = np.split( decoder_outputs, n_batches )

    return encoder_inputs, decoder_outputs

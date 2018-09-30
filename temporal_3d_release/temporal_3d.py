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

"""Predicting 3d poses from 2d joints
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import socket
import procrustes

import cameras

import temporal_model
import data_util as data_utils


# Learning
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "Learning rate")
tf.app.flags.DEFINE_boolean("sgd", False, "Whether to use SGD for optimization. Default is adam")
tf.app.flags.DEFINE_float("dropout", 1, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training")
tf.app.flags.DEFINE_integer("seqlen", 5, "Sequence length")
tf.app.flags.DEFINE_integer("epochs", 100, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
tf.app.flags.DEFINE_boolean("procrustes", False, "Apply procrustes analysis at test time")

# Data loading
tf.app.flags.DEFINE_boolean("use_sh", False, "Use 2d pose predictions from StackedHourglass")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of RNN hidden state.")
tf.app.flags.DEFINE_string("action","All", "The action to train on. All means all the actions")

# Directories
# FIXME hack to test code on rayats account
tf.app.flags.DEFINE_string("data_dir", "/ubc/cs/research/tracking-raid/rayat137/code/eyescream/tensorflow/pose_estimation/h36m/Training","Data directory")
tf.app.flags.DEFINE_string("train_dir","trained_model", "Training directory.")


tf.app.flags.DEFINE_boolean("evaluate", False, "Set to True for quantitative evaluation on the test set.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS



train_dir = os.path.join( FLAGS.train_dir,
  FLAGS.action,
  'dropout_{0}'.format(FLAGS.dropout),
  'epochs_{0}'.format(FLAGS.epochs) if FLAGS.epochs > 0 else '',
  'SGD' if FLAGS.sgd else 'adam',
  'lr_{0}'.format(FLAGS.learning_rate),
  'linear_size{0}'.format(FLAGS.linear_size),
  'batch_size_{0}'.format(FLAGS.batch_size),
  'use_stacked_hourglass' if FLAGS.use_sh else 'not_stacked_hourglass',
  'seqlen_{0}'.format(FLAGS.seqlen))

print( train_dir )
summaries_dir = os.path.join( train_dir, "log" ) # Directory for TB summaries
# Create right away to avoid race condition: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(summaries_dir))

def create_model( session, isTraining, dim_to_use_3d, batch_size, data_mean=0, data_std=0, dim_to_ignore_3d=0):
  """Create translation model and initialize or load parameters in session."""

  model = temporal_model.TemporalModel(
      FLAGS.sgd,
      FLAGS.linear_size,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      dim_to_use_3d,
      data_mean,
      data_std,
      dim_to_ignore_3d,
	  FLAGS.camera_frame,
      FLAGS.seqlen,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  if FLAGS.load <= 0:
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model

  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )
  if ckpt and ckpt.model_checkpoint_path:
    print(FLAGS.load)
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      print(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load)))
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    #print(ckpt.model_checkpoint_path)
    model.saver.restore( session, ckpt_name )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def train():

  """Train a Sequence to sequence model on human motion"""

  actions = data_utils.define_actions( FLAGS.action )

  number_of_actions = len( actions )

  rcams, vcams = cameras.load_cameras('cameras.h5', [1,5,6,7,8,9,11])

  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions, offsets_train, offsets_test = data_utils.read_3d_data(actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, vcams)
  if(FLAGS.use_sh):
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir )
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data(actions, FLAGS.data_dir, rcams, vcams)

  print( "done reading and normalizing data." )

  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options,
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    # === Create the model ===
    isTraining = True
    model = create_model( sess, isTraining, dim_to_use_3d, FLAGS.batch_size, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    model.train_writer.add_graph( sess.graph )
    print( "Model created" )

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    current_epoch = 0
    for _ in xrange( FLAGS.epochs ):

      current_epoch = current_epoch + 1
      # === Training for an epoch ===
      encoder_inputs, decoder_outputs = model.get_all_batches( train_set_2d, train_set_3d,  FLAGS.camera_frame, training=True)
      nbatches = len( encoder_inputs )
      print("There are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.
      for i in range( nbatches ):
        if (i+1) % 100 == 0:
          print("Working on epoch {0}, batch {1} / {2}... ".format( current_epoch, i+1, nbatches), end="" )
        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        isTraining = True
        step_loss, loss_summary, lr_summary,outputs =  model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining )

        if (i+1) % 100 == 0:
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / 100 ) )
        loss += step_loss
        current_step += 1
      loss = loss / nbatches
      print("==========================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "==========================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )

      # === Test for an epoch ===
      isTraining = False
      print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs
      cum_err = 0
      for action in actions:
        tot_act_err = 0
        print("{0:<12} ".format(action), end="")
        # Get 2d and 3d testing data for this action
        action_test_set_2d = get_action_subset( test_set_2d, action )
        action_test_set_3d = get_action_subset( test_set_3d, action )
        action_test_set_2d_gt = []
        for key2d in action_test_set_2d.keys():
          (subj, b, fname) = key2d
            # keys should be the same if 3d is in camera coordinates
          key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
          key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh') and FLAGS.camera_frame) else key3d
          #key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
          if fname.endswith('-sh'):
            fname = fname[:-3]

          enc_in = {}
          dec_out = {}
          enc_in[key2d]  = test_set_2d[ key2d ]
          dec_out[key3d] = test_set_3d[ key3d ]
          pose_2d_gt_list = []
          encoder_inputs, decoder_outputs = model.get_all_batches( enc_in, dec_out, FLAGS.camera_frame,  training=False)
          act_err, _, step_time, loss = evaluate_batches( sess, model, data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d, current_step, encoder_inputs, decoder_outputs )
          tot_act_err = tot_act_err + act_err
        print("{0:>6.2f}".format(tot_act_err/len(action_test_set_2d.keys())))
        cum_err = cum_err + tot_act_err/len(action_test_set_2d.keys())

      print("{0:<12} {1:>6.2f}".format("Average", cum_err/float(len(actions) )))
      print("{0:=^19}".format(''))
      # Log the error to tensorboard
      summaries = sess.run( model.err_mm_summary, {model.err_mm: float(cum_err/float(len(actions)))} )
      model.test_writer.add_summary( summaries, current_step )

      print( "Saving the model... ", end="" ); start_time = time.time()
      # Save the model
      model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step )
      print( "done in {0:02f} seconds".format(time.time() - start_time) )
    # Reset global time and loss
    step_time, loss = 0, 0
    sys.stdout.flush()

def evaluate_batches( sess, model,data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,current_step, encoder_inputs, decoder_outputs, current_epoch=0 ):

  n_joints = 17
  nbatches = len( encoder_inputs )
  #print(nbatches)
  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    dp = 1.0 # dropout keep probability is always 1 at test time

    step_loss, loss_summary, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

    if (i==0):
      dec_out = np.vstack([dec_out[0,:,:], dec_out[1:,FLAGS.seqlen-1,:]])
      poses3d = np.vstack([poses3d[0,:,:], poses3d[1:,FLAGS.seqlen-1,:]])
    else:
      dec_out = np.expand_dims(dec_out[:,FLAGS.seqlen-1,:],axis=0)
      poses3d = np.expand_dims(poses3d[:,FLAGS.seqlen-1,:],axis=0)

    dec_out = np.reshape(dec_out,[-1,(n_joints-1)*3])
    poses3d = np.reshape(poses3d,[-1,(n_joints-1)*3])

    ###NOTE: ADDED LINES FOR evaluation


    dec_out = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )

    # Keep only the relevant dimensions
    dtu3d = np.hstack( (np.arange(3), dim_to_use_3d) )

    dec_out = dec_out[:, dtu3d]
    poses3d = poses3d[:, dtu3d]


    if FLAGS.procrustes:
      # Apply per-frame procrustes alignment if asked to do so
      for j in range(poses3d.shape[0]):
        gt  = np.reshape(dec_out[j,:],[-1,3])
        out = np.reshape(poses3d[j,:],[-1,3])
        _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
        out = Z
        poses3d[j,:] = np.reshape(out,[-1,17*3] )
    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints*3, 3):
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx = dist_idx + 1

    all_dists.append(dists)
    #print(all_dists)
    #assert sqerr.shape[0] == FLAGS.batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time, loss

def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action
  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """

  return {k:v for k, v in poses_set.items() if k[1] == action}

def define_actions( action ):
  """
  Given an action string, returns a list of corresponding actions.
  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """
  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all":
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]

def evaluate(current_step=0):
  """Evaluate on all the test set"""
  if FLAGS.load <= 0:
    raise( ValueError, "Must give an iteration to read parameters from")

  actions = define_actions( FLAGS.action )
  rcams, vcams = cameras.load_cameras('cameras.h5', [1,5,6,7,8,9,11])
  # Load and normalize all the data
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions, offsets_train, offsets_test = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, vcams )

  if(FLAGS.use_sh):
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, _ , _ = data_utils.read_2d_predictions(actions, FLAGS.data_dir)

  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, _ , _  =  data_utils.create_2d_data(actions, FLAGS.data_dir, rcams, vcams)

  print( "done reading and normalizing data." )

  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  isTraining = False
  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:
    # === Create the model ===
    model = create_model( sess, isTraining, dim_to_use_3d, FLAGS.batch_size, data_mean_3d, data_std_3d, dim_to_ignore_3d)
    print("Model created")
    cum_err = 0
    print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs
    for action in actions:
      tot_act_err = 0
      print("{0:<12} ".format(action), end="")
      #print(test_set_2d_gt.keys())
      action_test_set_2d = get_action_subset( test_set_2d, action )
      action_test_set_3d = get_action_subset( test_set_3d, action )
      action_test_set_2d_gt = []
      for key2d in action_test_set_2d.keys():
        #print(key2d)
        (subj, b, fname) = key2d

        # keys should be the same if 3d is in camera coordinates
        key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
        key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh') and FLAGS.camera_frame) else key3d
        #key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
        if fname.endswith('-sh'):
          fname = fname[:-3]
        #print("###NAME OF THE FILE", fname[:-3])
        enc_in = {}
        dec_out = {}
        enc_in[key2d]  = test_set_2d[ key2d ]
        dec_out[key3d] = test_set_3d[ key3d ]
        pose_2d_gt_list = []

        encoder_inputs, decoder_outputs= model.get_all_batches( enc_in, dec_out, FLAGS.camera_frame,  training=False)
        act_err, _, step_time, loss = evaluate_batches( sess, model, data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d, current_step, encoder_inputs, decoder_outputs )
        tot_act_err = tot_act_err + act_err

      print("{0:>6.2f}".format(tot_act_err/len(action_test_set_2d.keys())))
      cum_err = cum_err + tot_act_err/len(action_test_set_2d.keys())

    print("{0:<12} {1:>6.2f}".format("Average", cum_err/float(len(actions) )))
    print("{0:=^19}".format(''))
    return cum_err/float(len(actions))


def main(_):
  if FLAGS.evaluate:
    errpr = evaluate()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()

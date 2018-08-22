"""
A bunch of utility functions for dealing with human3.6m data.
"""

from __future__ import division

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cameras
import viz
import h5py
import glob
import copy

#==============================#
#=== Bounding box functions ===#
#==============================#

"""
  For

  For human 3.6m
0  0  -- Hip
1  1  -- RHip
2  2  -- RKnee
3  3  -- RFoot
4  6  -- LHip
5  7  -- Lknee
6  8  -- Lfoot
7  12 -- Spine
8  13 -- Thorax
9  14 -- Neck/Nose
10  15 -- Head
11  17 -- LShoulder
12  18 -- LElbow
13  19 -- LWrist
14  25 -- RShoulder
15  26 -- RElbow
16  27 -- RWrist

  For StackedHourglass
  0  'RFoot'
  1  'RKnee'
  2  'RHip'
  3  'LHip'
  4  'LKnee'
  5  'LFoot',
  6  'Hip'
  7  'Spine'
  8  'Thorax'
  9  'Head'
  10  'RWrist'
  11  'RElbow'
  12  'Rshoulder'
  13  'LShoulder'
  14  'LElbow'
  15  'LWrist'



"""


def getBBcenter(bb):
  xmin, ymin, xlen, ylen = bb[:,0], bb[:,1], bb[:,2], bb[:,3]
  midx = xmin + xlen / 2
  midy = ymin + ylen / 2
  #midx = xmin
  #midy = ymin
  #print("### midxshape", midx.shape)
  midx = np.expand_dims(midx,1)
  midy = np.expand_dims(midy,1)
  center = np.concatenate((midx,midy),axis=1)
  return center


def normalizeBB( bb ):
  # Convert a bounding box to a square box
  # bb = [xmin, ymin, xlen, ylen]
  xmin, ymin, xlen, ylen = bb[0], bb[1], bb[2], bb[3]

  midx = xmin + xlen / 2
  midy = ymin + ylen / 2

  nbb = np.zeros_like( bb )

  if ylen > xlen:
    nbb[0], nbb[1], nbb[2], nbb[3] = midx-ylen/2, ymin, ylen, ylen
  else:
    nbb[0], nbb[1], nbb[2], nbb[3] = xmin, midy-xlen/2, xlen, xlen

  return nbb, midx, midy

def load_bb( bpath, subject, seqname ):
  # Load bounding boxes
  boxes = {}
  #print("###BPATH ",bpath)
  fname = os.path.join(bpath, 'S{0}'.format(subject), 'MySegmentsMat/ground_truth_bb/', '{0}.h5'.format(seqname.replace('_',' ')))
  with h5py.File( fname, 'r' ) as h5f:
    bbs = h5f['bbs'][:]
  return bbs

"""def load_bbs( bpath, subjects, seqnames ):
  # Load bounding boxes
  boxes = {}

  for s in subjects:
    for action in seqnames:
      fname = os.path.join(bpath, 'S{0}'.format(s), 'MySegmentsMat/ground_truth_bb/', '{0}.h5'.format(action.replace('_',' ')))
      with h5py.File( fname, 'r' ) as h5f:
        bbs = h5f['bbs'][:]

      # add to the dictionary
      boxes[(s, seqnames)] = bbs

  return boxes
"""
def load_data( bpath, subjects, actions, dim=3,verbose=True ):
  """
  Load data from disk, and put it in an easy-to-acess dictionary.

  Args:
    bpath. String. Base path where to load the data from,
    subjects. List of integers. Subjects whose data will be loaded.
    actions. List of strings. The actions to load.
	camera_frame. Boolean. Tells whether to retrieve data in camera coordinate system
  Returns:
    data. Dictionary with keys k=(subject, action, seqname)
          There will be 2 entries per subject/action if loading 3d data.
          There will be 8 entries per subject/action if loading 2d data.
  """

  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data = {}

  for subj in subjects:
    for action in actions:
      if verbose:
        print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoses/{0}D_positions'.format(dim), '{0}*.h5'.format(action) )
      print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )

        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        if seqname.startswith( action ):
          # This filters out e.g. walkDog and walkTogether
          if verbose:
            print( fname )
          loaded_seqs = loaded_seqs + 1

          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['{0}D_positions'.format(dim)][:]

          poses = poses.T
          #print('#####POSES_SHAPE',poses.shape)
          # Some early postprocesing (substracting root position from 3d)
          #  was happening here. Moved it out.

          data[ (subj, action, seqname) ] = poses

      if dim == 2:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format( loaded_seqs )
      else:
        assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format( loaded_seqs )

  return data

def normalization_stats(completeData, dim, predict_14=False ):
  """"Also borrowed for Ashesh. Computes mean, stdev and dimensions to ignore.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  # This line encodes which 17 2d-3d pairs we are predicting
  # NOTE We might want to bring this down to 14 to make it comparable to
  # https://arxiv.org/abs/1611.09010
  dimensions_to_ignore = []
  ### SHOULD REMOVE 14 (NOSE) from to use
  #dimensions_to_use    = np.array( [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27] )
  #dimensions_to_use    = np.array( [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27] )
  if dim == 2:
    # FIXME Removing for 2d although I don't think we should :/
    #dimensions_to_use = np.delete( dimensions_to_use, 0 )
    dimensions_to_use    = np.array( [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27] )
    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*2, dimensions_to_use*2+1)))
    dimensions_to_ignore = np.delete( np.arange(32*2), dimensions_to_use )
  else: # dim == 3
    # Do not predict the root node
    """For human 3.6m
    0  -- Hip
    1  -- RHip
    2  -- RKnee
    3  -- RFoot
    6  -- LHip
    7  -- Lknee
    8  -- Lfoot
    12 -- Spine
    13 -- Thorax
    14 -- Neck/Nose
    15 -- Head
    17 -- LShoulder
    18 -- LElbow
    19 -- LWrist
    25 -- RShoulder
    26 -- RElbow
    27 -- RWrist"""
    dimensions_to_use    = np.array( [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27] )
    if predict_14:
      dimensions_to_use = np.delete( dimensions_to_use, [0,7,9] )
    else:
      dimensions_to_use = np.delete( dimensions_to_use, 0 )
    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3, dimensions_to_use*3+1, dimensions_to_use*3+2)))
    dimensions_to_ignore = np.delete( np.arange(32*3), dimensions_to_use )

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use

def transform_world_to_camera(poses_set, cams, ncams=4 ):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args:
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted( poses_set.keys() ):
      subj, a, seqname = t3dk
      t3d_world = poses_set[ t3dk ]
      #print("#### SHAPE OF t3d_world::",t3d_world.shape)
      # FIXME this only works for real cameras
      for c in range( ncams ):
        R, T, f, c, k, p, name = cams[ (subj, c+1) ]
        camera_coord = cameras.world_to_camera_frame( np.reshape(t3d_world, [-1, 3]), R, T,f, c, k, p)
        camera_coord = np.reshape( camera_coord, [-1, 96] )
        #print("#### SHAPE OF CAMCOORD::",camera_coord.shape)
        sname = seqname[:-3]+"."+name+".h5" #Waiting 1.58860488.h5
        t3d_camera[ (subj, a, sname) ] = camera_coord

    return t3d_camera


def normalize_data( data, data_mean, data_std, dim_to_use, actions,dim=3):

  data_out = {}
  nactions = len(actions)
  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev )

    #data_out[ key ] = np.divide( (data[key] - data_mean), data_std )
    #data_out[ key ] = data_out[ key ][ :, dim_to_use ]
    #data_out[ key ] = np.hstack( (data_out[key], data[key][:,-nactions:]) )
  return data_out

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore ):
  """Borrowed from Ashesh. Unnormalizes a matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
  """
  T = normalizedData.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality
  #print(D)
  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  #print()
  dimensions_to_use = np.array(dimensions_to_use)
  #print(normalizedData.shape)
  #print(dimensions_to_ignore)
  origData[:, dimensions_to_use] = normalizedData

  # TODO this might be very inefficient? idk
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData

def load_offsets( bpath, subjects ):
  fname = os.path.join( bpath, 'offsets.h5' )

  with h5py.File( fname, 'r' ) as h5f:
    offsets = h5f['offsets'][:]

  data_out = {}
  for subj in subjects:
    data_out[ subj ] = offsets[:, subj-1 ]

  return data_out

def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all":
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]

def project_to_cameras( data_dir, poses_set, cams, ncams=4 ):
  """
  Project 3d poses to obtain 2d ones
  Args:
    poses_set: dictionary with 3d poses
    cams: dictionary with cameras
    ncams: number of cameras per subject
  Return:
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    # FIXME this only works for real cameras
    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )
      pts2d = np.reshape( pts2d, [-1, 64] )
      sname = seqname[:-3]+"."+name+".h5" #Waiting 1.58860488.h5
      t2d[ (subj, a, sname) ] = pts2d


  return t2d

def merge_two_dicts(x, y):
  """
  Given two dicts, merge them into a new dict as a shallow copy.
  """
  z = x.copy()
  z.update(y)
  return z

def load_stacked_hourglass(data_dir,subjects,actions,verbose=True):
  """
  Load data from disk, and put it in an easy-to-acess dictionary.

  Args:
    bpath. String. Base path where to load the data from,
    subjects. List of integers. Subjects whose data will be loaded.
    actions. List of strings. The actions to load.
    camera_frame. Boolean. Tells whether to retrieve data in camera coordinate system
  Returns:
    data. Dictionary with keys k=(subject, action, seqname)
          There will be 2 entries per subject/action if loading 3d data.
          There will be 8 entries per subject/action if loading 2d data.
  """
  data = {}
  for subj in subjects:
    for action in actions:
      if verbose:
        print('Reading subject {0}, action {1}'.format(subj, action))
      dpath = os.path.join( data_dir, 'S{0}'.format(subj), 'post_accept_sh_finetuned_10it/{0}*.h5'.format(action))
      print( dpath )
      fnames = glob.glob( dpath )
      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )
        seqname = seqname.replace('_',' ')
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue
        if seqname.startswith( action ):
          # This filters out e.g. walkDog and walkTogether
          if verbose:
            print( fname )
          loaded_seqs = loaded_seqs + 1
          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['poses'][:]
            permutation_idx = np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10])
            ### PERMUTE TO MAKE IT COMPATIBLE with h36m
            poses = poses[:,permutation_idx,:]
            ### Reshape into n times 16*2
            poses = np.reshape(poses,[poses.shape[0],-1])
            poses_final = np.zeros([poses.shape[0],32*2])
            dim_to_use_x    = np.array( [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27],dtype=np.int32 )*2
            dim_to_use_y    = dim_to_use_x+1
            dim_to_use = np.zeros(16*2,dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:,dim_to_use] = poses
            seqname = seqname+'-sh'
            data[ (subj, action, seqname) ] = poses_final

      # Make sure we loaded 8 sequences
      if (subj == 11 and action == 'Directions'): # <-- this video is damaged
        assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )
      else:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )

  return data

def read_2d_predictions(actions, data_dir):

  rcams, vcams = cameras.load_cameras('cameras.h5', [1,5,6,7,8,9,11], n_interpolations=0)
  train_set = load_stacked_hourglass(data_dir, [1, 5, 6, 7, 8], actions)
  test_set  = load_stacked_hourglass( data_dir, [9, 11], actions)
    #test_set  = load_stacked_hourglass( data_dir, [9], actions)

  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std,  dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )
  #print("DIM TO IGNORE",dim_to_ignore)
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, actions,2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, actions,2 )

  return train_set, test_set, data_mean, data_std, dim_to_ignore,dim_to_use


def create_2d_data( actions, data_dir, rcams, vcams, n_interpolations=0 ):
  """
  Creates 2d data from 3d points and real or virtual cameras.
  """

  # Load 3d data
  train_set = load_data( data_dir, [1, 5, 6, 7, 8], actions, dim=3 )
  test_set  = load_data( data_dir, [9, 11], actions, dim=3 )

  train_set_r = project_to_cameras( data_dir, train_set, rcams, ncams=4)
  train_set_v = project_to_cameras( data_dir, train_set, vcams, ncams=4*(n_interpolations))
  train_set = merge_two_dicts( train_set_r, train_set_v )

  test_set  = project_to_cameras( data_dir, test_set, rcams, ncams=4)
  # Apply 2d post-processing  ### FIXME
  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  # Divide every dimension independently (good if predicting 3d points directly)
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, actions )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, actions )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def postprocess_2d( poses_set,bbs ):
  """
  Center 2d points around Bounding_box
  """
  """for k in poses_set.keys():
    #(subj, a, sname) -- Key for 2d poses
    #print("###BBBS:   ",bbs[k].shape)
    center = getBBcenter(bbs[k])
    #print("###BBBS:   ",center.shape)
    poses = poses_set[k]
    poses = poses - np.tile( center, [1, 32] )
    poses_set[k] = poses
  """
  return poses_set

def read_3d_data( actions, data_dir, camera_frame=False,rcams=0,vcams=0,n_interpolations=0,predict_14=False):
  """
  Loads 3d data and normalizes it.
  """

  # Load 3d data
  train_set = load_data( data_dir, [1, 5, 6, 7, 8], actions, dim=3 )
  test_set  = load_data( data_dir, [9, 11], actions, dim=3 )

  if camera_frame:
    train_set_r = transform_world_to_camera(train_set,rcams,ncams=4)
    train_set_v = transform_world_to_camera( train_set, vcams, ncams=4*(n_interpolations) )
    train_set   = merge_two_dicts( train_set_r, train_set_v )
    # FIXME dirty test -- with Procrustes, this should not matter
    test_set    = transform_world_to_camera(test_set,rcams,ncams=4)
    # test_set    = transform_world_to_camera_dummy(test_set,rcams,ncams=4)

  # Apply 3d post-processing
  train_set, train_root_positions = postprocess_3d( train_set )
  test_set,  test_root_positions  = postprocess_3d( test_set )
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))

  # Compute normalization statistics
  if predict_14:
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3, predict_14=True )
  else:
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3 )



  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, actions )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, actions )
    # Load the offsets (bone lengths)
  offsets_train = load_offsets( data_dir, [1, 5, 6, 7, 8] )
  offsets_test  = load_offsets( data_dir, [9, 11] )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions, offsets_train, offsets_test

  #return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions

def postprocess_3d( poses_set ):
  """
  Center 3d points around root
  """
  root_positions = {}
  for k in poses_set.keys():
    # Keep track of the global position
    root_positions[k] = copy.deepcopy(poses_set[k][:,:3])

    # Remove the root from the 3d position
    poses = poses_set[k]
    poses = poses - np.tile( poses[:,:3], [1, 32] )
    poses_set[k] = poses

  return poses_set, root_positions

from __future__ import division

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data_util as data_utils
import viz
import socket
import itertools

def project_point_radial( P, R, T, f, c, k, p ):

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2

  # Fancy einsum Z fancy
  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) );
  tan = p[0]*XX[1,:] + p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2

def world_to_camera_frame(P,R,T, f, c, k, p ):
  """
  Convert a bunch of points from world to camera coordinates.

  Args
    P: 3d points in world coordinates
    R: Camera rotation matrix
	  T: Camera translation parameters
  Returns
    X_cam: 3d points in camera coordinates
  """
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]
  X_cam = R.dot( P.T - T ) # rotate and translate
  return X_cam.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame"""
  assert len(P.shape) == 2
  assert P.shape[1] == 3
  X_cam = R.T.dot( P.T ) + T # rotate and translate
  return X_cam.T

def compute_mean_A():
  [f,c] = obtain_mean_f_c()
  A = np.eye(3)
  A[0,0] = f[0,0]
  A[1,1] = f[1,0]
  A[0,2] = c[0,0]
  A[1,2] = c[1,0]

  return A


def obtain_mean_f_c():
  c_sum = np.zeros([2,1])
  f_sum = np.zeros([2,1])
  subjects=[1,5,6,7,8,9,11]
  with h5py.File('../cameras.h5','r') as hf:
    for s in subjects:
      for cams in range(4):
        [R,T,f,c,k,p,name] = load_camera_params(hf,'subject%d/camera%d/{0}' % (s,cams+1))
        c_sum = c_sum + c
        f_sum = f_sum + f

  return f_sum/28.0,c_sum/28.0



def load_camera_params( hf, path ):
  R = hf[ path.format('R') ][:]
  R = R.T

  T = hf[ path.format('T') ][:]
  f = hf[ path.format('f') ][:]
  c = hf[ path.format('c') ][:]
  k = hf[ path.format('k') ][:]
  p = hf[ path.format('p') ][:]

  name = hf[ path.format('Name') ][:]
  name = "".join( [chr(item) for item in name] )

  return R, T, f, c, k, p, name

def interpolate( cameras, n_inter ):
  '''
  Interpolate camera parameters to create virtual cameras
  Args:
    cams: list of cameras. Each entry is a tuple with camera params RTfckp
    n_inter: number of interpolations to make between each camera pair
  Returns:
    vcams: a list with all the interpolated virtual cameras
  '''
  from transformations import quaternion_from_matrix, quaternion_slerp, quaternion_matrix, interpolate_spherical

  ncams = len( cameras )

  inter_cameras = []
  for i in range(ncams):
    inter_cameras.append([])
    fractions = np.linspace(0, 1, n_inter+2)[1:-1]

    inter_idx = 1
    for inter in range(n_inter):
      # Interpolate rotation matrices (R) in quaternion space.
      q0 = quaternion_from_matrix(cameras[i][0])
      q1 = quaternion_from_matrix(cameras[(i+1)%ncams][0])
      q_inter = quaternion_slerp(q0, q1, fractions[inter])
      cam_inter = [quaternion_matrix(q_inter)[:3, :3]]

      # Interpolate translations (T) in spherical coords around the center
      # probably the ideal thing would be to use dual quaternion slerp.
      T_mid =interpolate_spherical(cameras[i][1].reshape(1, -1),
                                   cameras[(i+1)%ncams][1].reshape(1, -1), fractions[inter]).T
      cam_inter.append(T_mid)

      # Linear interpolation for the rest of numeric params (f, c, k, p)
      for j in range(2, 6):
        cam_inter.append((cameras[i][j] + cameras[(i+1)%ncams][j])/2.)

      # Give it a dummy name (name) - v for virtual
      cam_inter.append( cameras[i][6] + ".v{0}".format(inter_idx)  )
      inter_idx = inter_idx + 1
      inter_cameras[-1].append(cam_inter)

  # Put everything into one big list
  #allcams = sum([[cam]+intercam for cam,intercam in zip(cameras, inter_cameras)], [])
  vcams = list( itertools.chain(*inter_cameras) )

  return vcams




def load_cameras( bpath, subjects=[1,5,6,7,8,9,11], n_interpolations=0):
  # === Load 2d data the new way! ===
  rcams, vcams = {}, {}
  with h5py.File('../cameras.h5','r') as hf:
    for s in subjects:


      for c in range(4):
        rcams[(s, c+1)] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )


      if n_interpolations > 0:
        vcamsl = []
        for c in range(4):
          vcamsl.append( load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) ) )

        # Reorder so they make sense
        vcamsl = [vcamsl[i] for i in (0, 2, 3, 1)]

        # Interpolate to create new cameras
        vcamsl = interpolate( vcamsl, n_interpolations )

        # Add them to the virtual camera dictionary
        for c in range( len(vcamsl) ):
          vcams[ (s, c+1) ] = vcamsl[ c ]

    return rcams, vcams

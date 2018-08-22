""" Functions to visualize human poses """

import matplotlib.pyplot as plt
import numpy as np


def showBB( bb ):
  # Visualize a bounding box.

  # bb = [xmin, ymin, xlen, ylen]
  assert bb.size == 4, "bb should have 4 entries"
  plt.plot([bb[0], bb[0]+bb[2]],[bb[1], bb[1]], lw=3, c='b')
  plt.plot([bb[0], bb[0]+bb[2]],[bb[1]+bb[3], bb[1]+bb[3]], lw=3, c='b')
  plt.plot([bb[0], bb[0]],[bb[1], bb[1]+bb[3]], lw=3, c='b')
  plt.plot([bb[0]+bb[2], bb[0]+bb[2]],[bb[1], bb[1]+bb[3]], lw=3, c='b')


class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
        ## REMOVE 15 from I and J
        self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
        self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
        self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))
        ### 1 means right 0 means left
        # Make connection matrix
        self.plots = []
        for i in np.arange( len(self.I) ):
            x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
            y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
            z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
            #self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))
            self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def update(self, channels):
        assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
        vals = np.reshape( channels, (32, -1) )

        for i in np.arange( len(self.I) ):
            x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
            y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
            z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)


        r = 750;
        xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
        self.ax.set_xlim3d([-r+xroot, r+xroot])
        self.ax.set_zlim3d([-r+zroot, r+zroot])
        self.ax.set_ylim3d([-r+yroot, r+yroot])

        self.ax.set_aspect('equal')


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c"): # blue, orange
  # Visualize a 3d skeleton.

  assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (32, -1) )

  ## REMOVE 15 from I and J
  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  # I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1
  # J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1
  # LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  ### 1 means right 0 means left
  # Make connection matrix
  for i in np.arange( len(I) ):
    x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
    y = np.array( [vals[I[i], 1], vals[J[i], 1]] )
    z = np.array( [vals[I[i], 2], vals[J[i], 2]] )
    ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  #print( vals[:,0] )
  # ax.scatter( vals[:,0], vals[:,1], vals[:,2], marker='o', s=8 )

  r = 750;
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-r+xroot, r+xroot]); ax.set_xlabel("x")
  ax.set_zlim3d([-r+zroot, r+zroot]); ax.set_ylabel("y")
  ax.set_ylim3d([-r+yroot, r+yroot]); ax.set_zlabel("z")
  #ax.view_init(elev=45., azim=0)

  # ax.get_xaxis().set_ticklabels([])
  # ax.get_yaxis().set_ticklabels([])
  # ax.set_zticklabels([])
  ax.set_aspect('equal')

def show2DposePrediction(channels, ax, lcolor="#3498db", rcolor="#e74c3c"):
  # Visualize a 2d skeleton.

  assert channels.size == 32, "channels should have 32 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (16, -1) )
  #I  = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
  #J  = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1

  """local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri'}"""

  #I  = np.array([1,2,3,4,4,5,7,9, 14,11,12,13,14,15])-1
  #J  = np.array([2,3,7,5,7,6,9,10, 9,12,13, 9,15,16])-1
  #LR = np.array([0,0,0,1,1,1,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  I  = np.array([1,2,3,1,5,6,1,8, 9,9, 11,12,9, 14,15])-1
  J  = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])-1
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  ### 1 means right 0 means left
  # Make connection matrix
  #len(I)
  for i in np.arange(len(I)):
    x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
    y = np.array( [vals[I[i], 1], vals[J[i], 1]] )
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  r = 350
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-r+xroot, r+xroot]); ax.set_xlabel("x")
  ax.set_ylim([-r+yroot, r+yroot]); ax.set_ylabel("z")
  ax.set_aspect('equal')


class Ax2DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
        vals = np.zeros((32, 2))
        self.ax = ax

        self.I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1
        self.J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1
        self.LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        ### 1 means right 0 means left
        # Make connection matrix
        self.plots = []
        for i in np.arange( len(self.I) ):
          x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
          y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
          self.plots.append(self.ax.plot(x, y, lw=2, c=lcolor if self.LR[i] else rcolor))

        self.ax.set_aspect('equal')
        #self.ax.invert_yaxis()
        self.ax.set_ylabel("z")
        self.ax.set_xlabel("x")


    def update(self, im, channels):

        if not im:
            #print("Empty")
            pass
        elif not hasattr(self, 'im_data'):
            self.im_data = self.ax.imshow(im)
        else:
            self.im_data.set_data(im)

        assert channels.size == 64, "channels should have 64 entries, it has %d instead" % channels.size
        vals = np.reshape( channels, (32, -1) )


        # Make connection matrix
        for i in np.arange( len(self.I) ):
            x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
            y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)

        r = 350
        xroot, yroot = vals[0,0], vals[0,1]
        self.ax.set_xlim([-r+xroot, r+xroot])
        self.ax.set_ylim([-r+yroot, r+yroot])
        self.ax.invert_yaxis()



def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c"):
  # Visualize a 2d skeleton.

  assert channels.size == 64, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (32, -1) )


  ## REMOVE 15 from I and J
  #  I  = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
  #  J  = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
  #LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  ### 1 means right 0 means left
  # Make connection matrix
  for i in np.arange( len(I) ):
    x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
    y = np.array( [vals[I[i], 1], vals[J[i], 1]] )
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  r = 350
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-r+xroot, r+xroot]); ax.set_xlabel("x")
  ax.set_ylim([-r+yroot, r+yroot]); ax.set_ylabel("z")
  ax.set_aspect('equal')

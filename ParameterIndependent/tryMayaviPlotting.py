#!/usr/bin/env python3
# Hayley Wragg 2018-05-29
import numpy as np
import vtk
import sys
import os
import math as ma
from mayavi.core.api import Engine
from mayavi.sources.vtk_file_reader import VTKFileReader
from mayavi.modules.surface import Surface
from mayavi import mlab
from pyface.api import GUI

def PlotRays():
    ##Plot the obstacles and the room first

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nrao,Nre,h     =np.load('Parameters/Raytracing.npy')
    Nra=int(np.sqrt(Nrao/2.0)-1)*int(np.sqrt(2.0*Nrao))+1

    ##---Retrieve the Ray points ---------------------------------------
    data_matrix=np.load('RayMeshPoints'+str(int(Nrao))+'Refs'+str(int(Nre))+'n.npy')

    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')
    Tx            =np.load('Parameters/Origin.npy')
    Nob           =len(Oblist)
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')
    Nob2          =len(OuterBoundary)
    Room          =np.concatenate((Oblist,OuterBoundary),axis=0)
    RoomP=Oblist[0]
    for j in range(1,Nob):
      RoomP=np.concatenate((RoomP,Oblist[j]),axis=0)
    for j in range(0,Nob2):
      RoomP=np.concatenate((RoomP,OuterBoundary[j]),axis=0)
    mlab.clf()
    #mlab.figure(1)
    for k in range(0,int(len(RoomP)/3)):
      j=k*3
      x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
      y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
      z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
      mlab.plot3d(x,y,z)
    for j in range(0,int(Nra)):
      #j=int(Nre)*k
      x=np.array([data_matrix[j][0][0]])
      y=np.array([data_matrix[j][0][1]])
      z=np.array([data_matrix[j][0][2]])
      s=np.array([data_matrix[j][0][3]])
      for l in range(1,int(Nre)+1):
          #FIXME problem finding intersection for directly up
          if all( ma.isnan(x) for x in data_matrix[j][l]):
            pass
          else:
            x=np.append(x,[data_matrix[j][l][0]])
            y=np.append(y,[data_matrix[j][l][1]])
            z=np.append(z,[data_matrix[j][l][2]])
            s=np.append(s,[data_matrix[j][l][3]])
      mlab.plot3d(x,y,z,color= (0, 1, 1))
    mlab.savefig('Rayreflections.jpg')
    gui = GUI()
    gui.start_event_loop()
    return

if __name__=='__main__':
  PlotRays()
  print('Running  on python version')
  print(sys.version)
exit()

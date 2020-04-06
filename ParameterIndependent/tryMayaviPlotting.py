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
    Nrao,Nre,h ,L    =np.load('Parameters/Raytracing.npy')
    #Nra=98
    #Nre=5

    #L=1
    #Nra=int(np.sqrt(Nrao/2.0)-1)*int(np.sqrt(2.0*Nrao))+1
    #Nra=Nrao

    ##---Retrieve the Ray points ---------------------------------------
    data_matrix=L*np.load('./Mesh/RayMeshPoints'+str(int(Nrao))+'Refs'+str(int(Nre))+'m.npy')
    #data_matrix=data_matrix #print(data_matrix)

    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')
    Tx            =np.load('Parameters/Origin.npy')
    Nob           =len(Oblist)
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')
    Nob2          =len(OuterBoundary)
    Room          =np.concatenate((Oblist,OuterBoundary),axis=0)
    Oblist        =Room
    RoomP=Oblist[0]
    for j in range(1,Nob):
      RoomP=np.concatenate((RoomP,Oblist[j]),axis=0)
    for j in range(0,Nob2):
      RoomP=np.concatenate((RoomP,OuterBoundary[j]),axis=0)
    mlab.clf()
    #mlab.figure(1)
    mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.25)
    for j in range(0,int(Nrao)):
      #j=int(Nre)*k
      x=np.array([data_matrix[j][0][0]])
      y=np.array([data_matrix[j][0][1]])
      z=np.array([data_matrix[j][0][2]])
      s=np.array([data_matrix[j][0][3]])
      #print(x,y,z)
      # for l in range(1,2):#int(Nre)+1):
          # #FIXME problem finding intersection for directly up
          # if all( ma.isnan(x) for x in data_matrix[j][l]):
            # pass
          # else:
            # x=np.append(x,[data_matrix[j][l][0]])
            # y=np.append(y,[data_matrix[j][l][1]])
            # z=np.append(z,[data_matrix[j][l][2]])
            # s=np.append(s,[data_matrix[j][l][3]])
      x=np.append(x,[data_matrix[j][1][0]])
      y=np.append(y,[data_matrix[j][1][1]])
      z=np.append(z,[data_matrix[j][1][2]])
      mlab.plot3d(x,y,z,color= (0, 1, 1))
    for k in range(0,int(len(RoomP)/3)):
      j=k*3
      x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
      y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
      z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
      mlab.plot3d(x,y,z)
    if not os.path.exists('./ConeFigures'):
      os.makedirs('./ConeFigures')
    mlab.savefig('ConeFigures/Room.jpg',size=(1000,1000))
    mlab.close()
    #gui = GUI()
    #gui.start_event_loop()
    return

def PlotCones():
    ##Plot the obstacles and the room first

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nrao,Nre,h ,L    =np.load('Parameters/Raytracing.npy')
    data_matrix         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
    # Take Tx to be 0,0,0
    delangle      =np.load('Parameters/delangle.npy')

    mlab.points3d(0,0,0,scale_factor=0.1)
    mulfac=4
    xysteps=73
    iternum=int(Nrao)
    #mlab.savefig('ConeFigures/Rays.jpg',size=(1000,1000))
    #mlab.clf()
    #mlab.close()
    delth=2*np.arcsin(np.sqrt(2)*ma.sin(delangle/2))
    ta=np.sqrt(1-ma.tan(delth/2)**2)   # Nra>2 and an integer. Therefore tan(theta) exists.
    s=ma.sin(delth/2)
    beta=mulfac*ta*s
    if beta<h:
        Ncon=0
    else:
      Ncon=int(1+np.pi/np.arcsin(h/(beta)))
    anglevec=np.linspace(0.0,2*ma.pi,num=int(Ncon), endpoint=False) # Create an array of all the angles
    Norm=np.zeros((Ncon,3),dtype=np.float) # Initialise the matrix of normals
    Cones=np.zeros((iternum*Ncon,2,3))
    for j in range(0,iternum):#int(Nrao)):
      cx,cy,cz,cs=mulfac*data_matrix[j]/np.linalg.norm(data_matrix[j])
      d=data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3])                  # Normalise the direction of the ray
      if abs(d[2])>0 and Ncon>0:
       Norm[0]=np.array([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
       Norm[0]/=np.linalg.norm(Norm[0]) # Normalise the vector
       yax=np.cross(Norm[0],d)            # Compute another vector in the plane for the axis.
       yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
      elif Ncon>0:
       Norm[0]=np.array([0,0,1])        # If d_z is 0 then this vector is always in the plane.
       yax=np.cross(Norm[0],d)            # Compute another vector in the plane to form co-ordinate axis.
       yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
      if Ncon>0:
        Norm=np.outer(np.cos(anglevec),Norm[0])+np.outer(np.sin(anglevec),yax) # Use the outer product to multiple the axis
      for k in range(0,Ncon):
          Cones[j*Ncon+k][0]=np.array([cx,cy,cz])
          Cones[j*Ncon+k][1]=(mulfac*data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3]))+beta*Norm[k]
    if not os.path.exists('./ConeFigures'):
      os.makedirs('./ConeFigures')
    #mlab.savefig('ConeFigures/Rays.jpg',size=(1000,1000))
    N=int(len(Cones)/(xysteps*Ncon))
    for l in range(0,N-1):
      #j=int(Nre)*k
      if l==0:
        j=l
        x=0
        y=0
        z=0
        x2,y2,z2=mulfac*data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3])
        x=np.append(x,[x2])
        y=np.append(y,[y2])
        z=np.append(z,[z2])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
        x=0
        y=0
        z=0
        x2,y2,z2=mulfac*data_matrix[-1][0:3]/np.linalg.norm(data_matrix[-1][0:3])
        x=np.append(x,[x2])
        y=np.append(y,[y2])
        z=np.append(z,[z2])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
        for k in range(0, Ncon):
          j=k
          x=np.array([Cones[j][0][0]])
          y=np.array([Cones[j][0][1]])
          z=np.array([Cones[j][0][2]])
          xp=np.append(x,[Cones[j][1][0]])
          yp=np.append(y,[Cones[j][1][1]])
          zp=np.append(z,[Cones[j][1][2]])
          mlab.plot3d(xp,yp,zp,color= (1,0,1))
          j=-1-k
          x=np.array([Cones[j][0][0]])
          y=np.array([Cones[j][0][1]])
          z=np.array([Cones[j][0][2]])
          xp=np.append(x,[Cones[j][1][0]])
          yp=np.append(y,[Cones[j][1][1]])
          zp=np.append(z,[Cones[j][1][2]])
          mlab.plot3d(xp,yp,zp,color= (1,0,1))
      else:
        for k in range(0,xysteps): #int(Nrao)):
          j=(l-1)*xysteps+k+1
          x=0
          y=0
          z=0
          x2,y2,z2=mulfac*data_matrix[j][0:3]
          x=np.append(x,[x2])
          y=np.append(y,[y2])
          z=np.append(z,[z2])
          mlab.plot3d(x,y,z,color= (0, 1, 1))
        for k in range(0,xysteps*Ncon):
          j=(l)*xysteps*Ncon+k
          x=np.array([Cones[j][0][0]])
          y=np.array([Cones[j][0][1]])
          z=np.array([Cones[j][0][2]])
          xp=np.append(x,[Cones[j][1][0]])
          yp=np.append(y,[Cones[j][1][1]])
          zp=np.append(z,[Cones[j][1][2]])
          mlab.plot3d(xp,yp,zp,color= (1,0,1))
      filename=str('ConeFigures/Cone'+str(l)+'.jpg')
      mlab.savefig(filename,size=(1000,1000))
      mlab.clf()
    mlab.close()
    #gui = GUI()
    #gui.start_event_loop()
    return

if __name__=='__main__':
  PlotCones()
  PlotRays
  print('Running  on python version')
  print(sys.version)
exit()

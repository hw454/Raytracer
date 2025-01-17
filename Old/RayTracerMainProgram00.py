#!/usr/bin/env python3
# Updated Hayley Wragg 2019-03-15
''' Code to trace rays around a room using cones to account for
spreading. This version will compute this independent of object
parameters and object an array with signatures for the field'''
import numpy as np
import matplotlib.pyplot as mp
import Room01 as rom
import raytracerfunction as rayt
import DictionarySparseMatrix as DSM
import sys

if __name__=='__main__':
  #print('Running  on python version')
  #print(sys.version)
  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h     =np.load('Parameters/Raytracing.npy')

  ##----Retrieve the environment--------------------------------------
  #Oblist        =np.load('Parameters/Obstacles.npy')
  Tx            =np.load('Parameters/Origin.npy')
  Nob           =0#=len(Oblist)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')
  #print(Oblist, OuterBoundary)
  Oblist        =OuterBoundary#=np.concatenate((Oblist,OuterBoundary),axis=0)
  Room=rom.room(Oblist,Nob)

  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)

  # Room contains all the obstacles and walls.

  #Mesh=DSM.DS(Nx,Ny,Nz,int(Nre*(Nra+1)),int(Nob*(Nre+1)))
  # This large mesh is initialised as empty. It contains reference to
  # every segment at every position in the room.
  # The history of the ray up to that point is stored in a vector at that reference point.

  # Calculate the Ray trajectories
  Rays=Room.ray_bounce(Tx, int(Nre), int(Nra))
  np.save('RayPoints'+str(int(Nra))+'Refs'+str(int(Nre))+'n.npy',Rays)
  exit()


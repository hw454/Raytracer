#!/usr/bin/env python3
# Updated Hayley Wragg 2019-03-15
''' Code to trace rays around a room. This code computes the trajectories only.'''
import numpy as np
import Room  as rom
import raytracerfunction as rayt
import sys
import ParameterInput as PI
import DictionarySparseMatrix as DSM
import time as t
import matplotlib.pyplot as mp

#FIXME write a new program with a similar structure for storing the information in a DSM
# Is it possible to use this function and build on top? -Calculation is
# reduced if the rays don't have to be iterated through after being saved.

def RayTracer():

  # Run the ParameterInput file
  out=PI.DeclareParameters()

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h     =np.load('Parameters/Raytracing.npy')

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy')             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of intial ray directions for Nra rays.
  Oblist        =np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
  #Nob           =len(Oblist)                                 # The number of obstacles in the room

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)

  # Calculate the Ray trajectories
  print('Starting trajectory calculation')
  print('-------------------------------')
  Rays=Room.ray_bounce(Tx, int(Nre), int(Nra), Direc)
  print('-------------------------------')
  print('Trajectory calculation completed')
  print('Time taken',Room.time)
  print('-------------------------------')
  np.save('RayPoints'+str(int(Nra))+'Refs'+str(int(Nre))+'n.npy',Rays)
  # The "Rays" file is Nra+1 x Nre+1 x 4 array containing the
  # co-ordinate and obstacle number for each reflection point corresponding
  # to each source ray.

  return 1

def MeshProgram():
  print('-------------------------------')
  print('Building Mesh')
  print('-------------------------------')
  # Run the ParameterInput file
  out=PI.DeclareParameters()

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h     =np.load('Parameters/Raytracing.npy')

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy')             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  Oblist        =np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
  #Nob           =len(Oblist)                                 # The number of obstacles in the room

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob

  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  Mesh=DSM.DS(Nx,Ny,Nz,int(Nob*Nre+1),int((Nre)*(Nra)+1))
  print('-------------------------------')
  print('Mesh built')
  print('-------------------------------')
  print('Starting the ray bouncing and information storage')
  print('-------------------------------')
  Rays, Mesh=Room.ray_mesh_bounce(Tx,int(Nre),int(Nra),Direc,Mesh)
  np.save('RayMeshPoints'+str(int(Nra))+'Refs'+str(int(Nre))+'n.npy',Rays)
  print('-------------------------------')
  print('Ray-launching complete')
  print('Time taken',Room.time)
  print('-------------------------------')
  # This large mesh is initialised as empty. It contains reference to
  # every segment at every position in the room.
  # The history of the ray up to that point is stored in a vector at that reference point.
  return Mesh

def RefCoefComputation(Mesh):
  out=PI.ObstacleCoefficients()
  FreeSpace=np.load('Parameters/FreeSpace.npy')
  freq         =np.load('Parameters/frequency.npy')
  Znob         =np.load('Parameters/Znob.npy')
  refindex     =np.load('Parameters/refindex.npy')
  print('-------------------------------')
  print('Computing the reflection coeficients')
  print('-------------------------------')
  RefCoefPerp, RefCoefPar=DSM.ref_coef(Mesh,FreeSpace,freq,Znob,refindex)
  print('-------------------------------')
  print('Reflection coeficients found')
  print('-------------------------------')
  return RefCoef

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  out=RayTracer()
  Mesh=MeshProgram()
  RefCoef=RefCoefComputation(Mesh)
  exit()


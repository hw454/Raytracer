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
  ''' Refect rays and output the Mesh containing ray information.

  Parameters for the raytracer are input in :py:mod:`ParameterInput`
  The raytracing parameters defined in this module are saved and then loaded.

  * 'Raytracing.npy' - An array of 3 floats which is saved to \
  [Nra (number of rays), Nre (number of reflections), h (relative meshwidth)]
  * 'Obstacles.npy'  - An array for 3x3x1 arrays containing co-ordinates \
  forming triangles which form the obstacles. This is saved to Oblist \
  (The obstacles which are within the outerboundary )
  * 'Origin.npy'     - A 3x1 array for the co-ordinate of the source. \
  This is saved to Tx  (The location of the source antenna and origin \
  of every ray)
  * 'OuterBoundary.npy' - An array for 3x3x1 arrays containing \
  co-ordinates forming triangles which form the obstacles. This is \
  saved to OuterBoundary   (The Obstacles forming the outer boundary of \
  the room )
  Put the two arrays of obstacles into one array

  .. code::

     Oblist=[Oblist,OuterBoundary]

  * 'Directions.npy' - An Nrax3x1 array containing the vectors which \
  correspond to the initial direction of each ray. This is save to Direc.

  A room is initialised with *Oblist* using the :class:`room` \
  class in :py:mod:`Room`.

  The number of obstacles and the number of x, y and z steps is found

  .. code::

      Nob=Room.Nob
      Nx=int(Room.maxxleng()/h)
      Ny=int(Room.maxyleng()/h)
      Nz=int(Room.maxzleng()/h)

  Initialise a :py:mod:`DictionarySparseMatrix`. :py:class:`DS` with the \
  number of spaces in the x, y and z axis Nx, Ny, Nz, the number of \
  obstacles Nob, the number of reflections Nre and the number of rays Nra.

  .. code::

    Mesh=DSM.DS(Nx,Ny,Nz,int(Nob*Nre+1),int((Nre)*(Nra)+1))

  Find the reflection points of the rays and store the distance and \
  reflection angles of the rays in the Mesh. Use the :py:class:`room`. \
  :func:`ray_mesh_bounce(Tx,Nre,Nra,Direc,Mesh)` function.

  .. code::

     Rays, Mesh=Room.ray_mesh_bounce(Tx,int(Nre),int(Nra),Direc,Mesh)

  Save the reflection points in Rays to \
  'RayMeshPoints\ **Nra**\ Refs\ **Nre**\ n.npy' making the \
  substitution for **Nra** and **Nre** with their parameter values.

  :return: Mesh
  '''
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
  ''' Compute the mesh of reflection coefficients.

  :param Mesh: The DS mesh which contains the angles and distances rays \
  have travelled.

  Load the physical parameters using :py:mod:`ParameterInput`.\
  :func:`.ObstacleCoefficients()`
  * Znobrat - is the vector of characteristic impedances for obstacles \
  divided by the characteristic impedance of air.
  * refindex - if the vector of refractive indexes for the obstacles.

  Compute the Reflection coefficients (RefCoefper,Refcoefpar) using:
  :py:mod:`DictionarySparseMatrix`.\
  :func:`ref_coef((Mesh,Znobrat,refindex)`
  :return: (RefCoefper,Refcoefpar)
  '''
  out=PI.ObstacleCoefficients()
  #FreeSpace=np.load('Parameters/FreeSpace.npy')
  #freq         =np.load('Parameters/frequency.npy')
  Znobrat      =np.load('Parameters/Znobrat.npy')
  refindex     =np.load('Parameters/refindex.npy')
  print('-------------------------------')
  print('Computing the reflection coeficients')
  print('-------------------------------')
  start=t.time()
  RefCoefPerp, RefCoefPar=DSM.ref_coef(Mesh,Znobrat,refindex)
  end=t.time()
  print('-------------------------------')
  print('Reflection coeficients found')
  print('Computation time',end-start)
  print('-------------------------------')
  return RefCoefPerp, RefCoefPar

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  out=RayTracer()
  Mesh=MeshProgram()
  RefCoefperp, Refcoefpar=RefCoefComputation(Mesh)
  exit()


#!/usr/bin/env python3
# Updated Hayley Wragg 2019-03-15
''' Code to trace rays around a room. This code uses:

  * the function :py:func:`RayTracer` to compute the points for \
  the ray trajectories.
  * the function :py:func:`MeshProgram` to compute the points for \
  the ray trajectories and iterate along the rays storing the \
  information in a :py:class:`DictionarySparseMatrix.DS` and outputing \
  the points and mesh.
  * the function :py:func:`power_grid` which loads the last saved \
  and loads the antenna and obstacle physical parameters from \
  :py:func:`ParameterInput.ObstacleCoefficients`. It uses these and \
  the functions :py:func:`RefCoefComputation` which output Rper \
  and Rpar the perpendicular and parallel to polarisation reflection \
  coefficients, and the function :py:func:`RefCombine` to \
  get the loss from reflection for each ray segment entering each grid \
  point. This is then combine with the distance of each raysegments \
  travel from the mesh and the antenna gains to get the Power in \
  decibels.

  * The ray points from :py:func:`RayTracer` are saved as:
    'RayPoints\ **Nra**\ Refs\ **Nre**\ n.npy' with **Nra** replaced by the \
    number of rays and **Nre** replaced by the number of reflections.
  * The ray points from :py:func:`MeshProgram` are saved as:
    'RayMeshPoints\ **Nra**\ Refs\ **Nre**\ n.npy' with **Nra** replaced \
    by the \
    number of rays and **Nre** replaced by the number of reflections. \
    The mesh is saved as 'DSM\ **Nra**\ Refs\ **Nre**\ m.npy'.

  '''
import numpy as np
import Room  as rom
import raytracerfunction as rayt
import sys
import ParameterInput as PI
import DictionarySparseMatrix as DSM
import time as t
import matplotlib.pyplot as mp
import os

epsilon=sys.float_info.epsilon

def RayTracer():
  ''' Refect rays and output the points of reflection.

  Parameters for the raytracer are input in \
  :py:func:`ParameterInput.DeclareParameters()` The raytracing \
  parameters defined in this function are saved and then loaded.

  * 'Raytracing.npy' - An array of 4 floats which is saved to \
  [Nra (number of rays), Nre (number of reflections), \
  h (relative meshwidth), \
  L (room length scale, the longest axis has been rescaled to 1 and this \
  is it's original length)]
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

  Find the reflection points of the rays using \
  :py:func:`room.ray_bounce` function.

  .. code::

     Rays, Mesh=Room.ray_bounce(Tx,Nre,Nra,Direc)

  Save the reflection points in Rays to \
  'RayPoints\ **Nra**\ Refs\ **Nre**\ n.npy' making the \
  substitution for **Nra** and **Nre** with their parameter values.

  :return: 0 to mark a successful run
  '''

  # Run the ParameterInput file
  out=PI.DeclareParameters()

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h,L     =np.load('Parameters/Raytracing.npy')
  Nre=int(Nre)
  Nra=int(Nra)

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
  Rays=Room.ray_bounce(Tx, Nre, Nra, Direc)
  print('-------------------------------')
  print('Trajectory calculation completed')
  print('Time taken',Room.time)
  print('-------------------------------')
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  filename=str('./Mesh/RayPoints'+str(int(Nra))+'Refs'+str(int(Nre))+'m.npy')
  np.save(filename,Rays)
  # The "Rays" file is Nra+1 x Nre+1 x 4 array containing the
  # co-ordinate and obstacle number for each reflection point corresponding
  # to each source ray.

  return 0



def MeshProgram():
  ''' Refect rays and output the Mesh containing ray information.

  Parameters for the raytracer are input in :py:mod:`ParameterInput`
  The raytracing parameters defined in this module are saved and then loaded.

  * 'Raytracing.npy' - An array of 4 floats which is saved to \
  [Nra (number of rays), Nre (number of reflections), \
  h (relative meshwidth), \
  L (room length scale, the longest axis has been rescaled to 1 and this \
  is it's original length)]
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

  A room is initialised with *Oblist* using the py:class:`Room.room` \
  class in :py:mod:`Room`.

  The number of obstacles and the number of x, y and z steps is found

  .. code::

      Nob=Room.Nob
      Nx=int(Room.maxxleng()/h)
      Ny=int(Room.maxyleng()/h)
      Nz=int(Room.maxzleng()/h)

  Initialise a `DSM`. \
  :py:class:`DictionarySparseMatrix.DS` with the \
  number of spaces in the x, y and z axis Nx, Ny, Nz, the number of \
  obstacles Nob, the number of reflections Nre and the number of rays Nra.

  .. code::

    Mesh=DSM.DS(Nx,Ny,Nz,int(Nob*Nre+1),int((Nre)*(Nra)+1))

  Find the reflection points of the rays and store the distance and \
  reflection angles of the rays in the Mesh. Use the \
  py:func:`Room.room.ray_mesh_bounce` function.

  .. code::

     Rays, Mesh=Room.ray_mesh_bounce(Tx,int(Nre),int(Nra),Direc,Mesh)

  Save the reflection points in Rays to \
  'RayMeshPoints\ **Nra**\ Refs\ **Nre**\ n.npy' making the \
  substitution for **Nra** and **Nre** with their parameter values.

  :return: Mesh

  '''
  #print('-------------------------------')
  #print('Building Mesh')
  #print('-------------------------------')
  # Run the ParameterInput file
  out=PI.DeclareParameters()

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra=int(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy')/L             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  Nx=int(Room.maxxleng()/(h))
  Ny=int(Room.maxyleng()/(h))
  Nz=int(Room.maxzleng()/(h))
  Mesh=DSM.DS(Nx,Ny,Nz,Nob*Nre+1,Nra*(Nre+1))
  #print('-------------------------------')
  #print('Mesh built')
  print('-------------------------------')
  print('Starting the ray bouncing and information storage')
  print('-------------------------------')
  t0=t.time()
  Rays, Mesh=Room.ray_mesh_bounce(Tx,Nre,Nra,Direc,Mesh,deltheta)
  t1=t.time()
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  np.save('./Mesh/RayMeshPoints'+str(Nra)+'Refs'+str(Nre)+'m.npy',Rays)
  meshname=str('./Mesh/DSM'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  Mesh.save_dict(meshname)
  print('-------------------------------')
  print('Ray-launching complete')
  print('Time taken',t1-t0)
  print('-------------------------------')
  return Mesh

def StdProgram(index=0):
  ''' Refect rays and input object information output the power.

  Parameters for the raytracer are input in :py:mod:`ParameterInput`
  The raytracing parameters defined in this module are saved and then loaded.

  * 'Raytracing.npy' - An array of 4 floats which is saved to \
  [Nra (number of rays), Nre (number of reflections), \
  h (relative meshwidth), \
  L (room length scale, the longest axis has been rescaled to 1 and this \
  is it's original length)]
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

  A room is initialised with *Oblist* using the py:class:`Room.room` \
  class in :py:mod:`Room`.

  The number of obstacles and the number of x, y and z steps is found

  .. code::

      Nob=Room.Nob
      Nx=int(Room.maxxleng()/h)
      Ny=int(Room.maxyleng()/h)
      Nz=int(Room.maxzleng()/h)

  Initialise a `DSM`. \
  :py:class:`DictionarySparseMatrix.DS` with the \
  number of spaces in the x, y and z axis Nx, Ny, Nz, the number of \
  obstacles Nob, the number of reflections Nre and the number of rays Nra.

  .. code::

    Mesh=DSM.DS(Nx,Ny,Nz,int(Nob*Nre+1),int((Nre)*(Nra)+1))

  Find the reflection points of the rays and store the power Use the \
  py:func:`Room.room.ray_mesh_bounce` function.

  .. code::

     Rays, Mesh=Room.ray_mesh_power_bounce(Tx,int(Nre),int(Nra),Direc,Mesh)

  Save the reflection points in Rays to \
  'RayMeshPoints\ **Nra**\ Refs\ **Nre**\ n.npy' making the \
  substitution for **Nra** and **Nre** with their parameter values.

  :return: Mesh

  '''
  print('-------------------------------')
  print('Building Mesh')
  print('-------------------------------')
  # Run the ParameterInput file
  out1=PI.DeclareParameters()
  out2=PI.ObstacleCoefficients()
  if out1==0 & out2==0: pass
  else:
      raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra=int(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy')/L             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/TxGains'+str(index)+'.npy')
  freq          = np.load('Parameters/frequency'+str(index)+'.npy')
  Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')
  Pol           = np.load('Parameters/Pol'+str(index)+'.npy')
  c             =Freespace[3]
  khat          =freq*L/c
  lam           =(2*np.pi*c)/freq
  Antpar        =np.array([khat,lam,L])

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat'+str(index)+'.npy')
  refindex     =np.load('Parameters/refindex'+str(index)+'.npy')

  Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a zero for placement in the LOS row
  refindex=np.insert(refindex,0,1.0+0.0j)

  Nx=int(Room.maxxleng()/(h)+1)
  Ny=int(Room.maxyleng()/(h)+1)
  Nz=int(Room.maxzleng()/(h)+1)
  Mesh=np.zeros((Nx,Ny,Nz,2),dtype=np.complex128)
  print('-------------------------------')
  print('Mesh for Std program built')
  print('-------------------------------')
  print('Starting the ray bouncing and field storage')
  print('-------------------------------')
  Rays, Grid=Room.ray_mesh_power_bounce(Tx,Nre,Nra,Direc,Mesh,Znobrat,refindex,Antpar,Gt,Pol,deltheta)
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  np.save('./Mesh/RayMeshPointsstd'+str(int(Nra))+'Refs'+str(int(Nre))+'m.npy',Rays*L)
  np.save('./Mesh/Power_gridstd'+str(Nra)+'Refs'+str(int(Nre))+'m'+str(int(index))+'.npy',Grid)
  # print('-------------------------------')
  # print('Ray-launching complete')
  # print('Time taken',Room.time)
  # print('-------------------------------')
  return Grid

def power_grid(Roomnum=0):
  ''' Calculate the field on a grid using enviroment parameters and the \
  ray Mesh.

  Loads:

  * (*Nra*\ = number of rays, *Nre*\ = number of reflections, \
  *h*\ = meshwidth, *L*\ = room length scale)=`Paramters/Raytracing.npy`
  * (*Nob*\ =number of obstacles)=`Parameters/Nob.npy`
  * (*Gt*\ =transmitter gains)=`Parameters/TxGains.npy`
  * (*freq*\ = frequency)=`Parameters/frequency.npy`
  * (*Freespace*\ = permittivity, permeabilty \
  and spead of light)=`Parameters/Freespace.npy`
  * (*Znobrat*\ = Znob/Z0, the ratio of the impedance of obstacles and \
  the impedance in freespace.) = `Parameters/Znobrat.npy`
  * (*refindex*\ = the refractive index of the obstacles)=\
  Paramerters/refindex.npy`
  * (*Mesh*)=`DSM\ **Nra**\ Refs\ **Nre**\ m.npy`

  Method:
  * Initialise Grid using the number of x, y, and z steps in *Mesh*.
  * Use the function :py:func:`DictionarySparseMatrix.DS.power_compute`
  to compute the power.

  :rtype: Nx \ Ny x Nz numpy array of floats.

  :returns: Grid

  '''

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra=int(Nra)
  Nre=int(Nre)
  #Roomnum        =int(input('How many combinations of room values do you want to test?'))
  Nob            =np.load('Parameters/Nob.npy')

  ##----Retrieve the Mesh--------------------------------------
  meshname=str('./Mesh/DSM'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  Mesh= DSM.load_dict(meshname)

  ##----Initialise Grid For Power------------------------------------------------------
  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  Ns=max(Nx,Ny,Nz)
  Grid=np.zeros((Nx,Ny,Nz),dtype=float)
  t0=t.time()
  for index in range(0,Roomnum):
    PI.ObstacleCoefficients(index)
    ##----Retrieve the antenna parameters--------------------------------------
    Gt            = np.load('Parameters/TxGains'+str(index)+'.npy')
    freq          = np.load('Parameters/frequency'+str(index)+'.npy')
    Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')
    Pol           = np.load('Parameters/Pol'+str(index)+'.npy')

    ##----Retrieve the Obstacle Parameters--------------------------------------
    Znobrat      =np.load('Parameters/Znobrat'+str(index)+'.npy')
    refindex     =np.load('Parameters/refindex'+str(index)+'.npy')
    # Make the refindex, impedance and gains vectors the right length to
    # match the matrices.
    Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nob*Nre+1. Repeat Nob
    Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a zero for placement in the LOS row
    refindex=np.tile(refindex,(Nre,1))
    refindex=np.insert(refindex,0,1.0+0.0j)
    Gt=np.tile(Gt,(Nre+1,1))

    # Calculate the necessry parameters for the power calculation.
    c             =Freespace[3]
    khat          =freq*L/c
    lam           =(2*np.pi*c)/freq
    Antpar        =np.array([khat,lam,L])
    if index==0:
      Grid,ind=DSM.power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra,Nre,Ns)
    else:
      Grid,ind=DSM.power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra,Nre,Ns,ind)
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
    np.save('./Mesh/Power_grid'+str(Nra)+'Refs'+str(Nre)+'m'+str(index)+'.npy',Grid) #.compressed())
  t1=t.time()
  print('-------------------------------')
  print('Power from DSM complete')
  print('Time taken',t1-t0)
  print('-------------------------------')
  return Grid

def Quality(Roomnum=0):
  ''' Calculate the field on a grid using enviroment parameters and the \
  ray Mesh.

  Loads:

  * (*Nra*\ = number of rays, *Nre*\ = number of reflections, \
  *h*\ = meshwidth, *L*\ = room length scale)=`Paramters/Raytracing.npy`
  * (*Nob*\ =number of obstacles)=`Parameters/Nob.npy`
  * (*Gt*\ =transmitter gains)=`Parameters/TxGains.npy`
  * (*freq*\ = frequency)=`Parameters/frequency.npy`
  * (*Freespace*\ = permittivity, permeabilty \
  and spead of light)=`Parameters/Freespace.npy`
  * (*Znobrat*\ = Znob/Z0, the ratio of the impedance of obstacles and \
  the impedance in freespace.) = `Parameters/Znobrat.npy`
  * (*refindex*\ = the refractive index of the obstacles)=\
  Paramerters/refindex.npy`
  * (*Mesh*)=`DSM\ **Nra**\ Refs\ **Nre**\ m.npy`

  Method:
  * Initialise Grid using the number of x, y, and z steps in *Mesh*.
  * Use the function :py:func:`DictionarySparseMatrix.DS.power_compute`
  to compute the power.

  :rtype: Nx \ Ny x Nz numpy array of floats.

  :returns: Grid

  '''

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra=int(Nra)
  Nre=int(Nre)
  #Roomnum        =int(input('How many combinations of room values do you want to test?'))
  Nob            =np.load('Parameters/Nob.npy')

  ##----Retrieve the Mesh--------------------------------------
  meshname=str('./Mesh/DSM'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  Mesh= DSM.load_dict(meshname)

  ##----Initialise Grid For Power------------------------------------------------------
  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  Ns=max(Nx,Ny,Nz)
  Grid=np.zeros((Nx,Ny,Nz),dtype=float)
  t0=t.time()
  for index in range(0,Roomnum):
    PI.ObstacleCoefficients(index)
    ##----Retrieve the antenna parameters--------------------------------------
    Gt            = np.load('Parameters/TxGains'+str(index)+'.npy')
    freq          = np.load('Parameters/frequency'+str(index)+'.npy')
    Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')
    Pol           = np.load('Parameters/Pol'+str(index)+'.npy')

    ##----Retrieve the Obstacle Parameters--------------------------------------
    Znobrat      =np.load('Parameters/Znobrat'+str(index)+'.npy')
    refindex     =np.load('Parameters/refindex'+str(index)+'.npy')
    # Make the refindex, impedance and gains vectors the right length to
    # match the matrices.
    Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nob*Nre+1. Repeat Nob
    Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a zero for placement in the LOS row
    refindex=np.tile(refindex,(Nre,1))
    refindex=np.insert(refindex,0,1.0+0.0j)
    Gt=np.tile(Gt,(Nre+1,1))

    # Calculate the necessry parameters for the power calculation.
    c             =Freespace[3]
    khat          =freq*L/c
    lam           =(2*np.pi*c)/freq
    Antpar        =np.array([khat,lam,L])
    if index==0:
     Q,ind=DSM.quality_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra,Nre,Ns)
    else:
     Q,ind=DSM.quality_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra,Nre,Ns,ind)
    if not os.path.exists('./Quality'):
      os.makedirs('./Quality')
    np.save('./Quality/Quality'+str(Nra)+'Refs'+str(Nre)+'m'+str(index)+'.npy',Q)
  t1=t.time()
  print('-------------------------------')
  print('Quality from DSM complete', Q)
  print('Time taken',t1-t0)
  print('-------------------------------')
  return Grid

def RefCoefComputation(Mesh):
  ''' Compute the mesh of reflection coefficients.

  :param Mesh: The DS mesh which contains the angles and distances rays \
  have travelled.

  Load the physical parameters using
  :py:func:`ParameterInput.ObstacleCoefficients`

  * Znobrat - is the vector of characteristic impedances for obstacles \
  divided by the characteristic impedance of air.
  * refindex - if the vector of refractive indexes for the obstacles.

  Compute the Reflection coefficients (RefCoefper,Refcoefpar) using: \
  :py:func:`DictionarySparseMatrix.ref_coef`

  :rtype: (:py:class:`DictionarySparseMatrix.DS` (Nx,Ny,Nz,na,nb)\
    , :py:class:`DictionarySparseMatrix.DS` (Nx,Ny,Nz,na,nb))

  :returns: (RefCoefper,Refcoefpar)

  '''
  out=PI.ObstacleCoefficients()
  Znobrat      =np.load('Parameters/Znobrat'+str(index)+'.npy')
  refindex     =np.load('Parameters/refindex'+str(index)+'.npy')
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

def RefCombine(Rper,Rpar):
  ''' Combine reflection coefficients to get the loss from reflection \
  coefficient for each ray segment.

  Take in the DS's (:py:mod:`DictionarySparseMatrix`. :py:class:`DS`)\
  corresponding to the reflection coefficients for all the ray \
  interactions (:py:mod:`DictionarySparseMatrix`. :py:func:`ref_coef(Mesh)`).

  Use the function :py:mod:`DictionarySparseMatrix`. :py:class:`DS`. \
  :py:func:`dict_col_mult()` to multiple reflection coefficients in the same column.

  .. code::

     Combper=[
     [prod(nonzero terms in column 0 in Rper[0,0,0]),
     prod(nonzero terms in column 1 in Rper[0,0,0]),
     ...,
     prod(nonzero terms in column nb in Rper[0,0,0]
     ],
     ...,
     [prod(nonzero terms in column 0 in Rper[Nx-1,Ny-1,Nz-1]),
     prod(nonzero terms in column 1 in Rper[Nx-1,Ny-1,Nz-1]),
     ...,
     prod(nonzero terms in column nb in Rper[Nx-1,Ny-1,Nz-1]
     ]
     ]

  .. code::

     Combpar=[
     [prod(nonzero terms in column 0 in Rpar[0,0,0]),
     prod(nonzero terms in column 1 in Rpar[0,0,0]),
     ...,
     prod(nonzero terms in column nb in Rpar[0,0,0]
     ],
     ...,
     [prod(nonzero terms in column 0 in Rpar[Nx-1,Ny-1,Nz-1]),
     prod(nonzero terms in column 1 in Rpar[Nx-1,Ny-1,Nz-1]),
     ...,
     prod(nonzero terms in column nb in Rpar[Nx-1,Ny-1,Nz-1]
     ]
     ]

  :param Rper: The mesh corresponding to reflection coefficients \
  perpendicular to the polarisation.

  :param Rpar: The mesh corresponding to reflection coefficients \
  parallel to the polarisation.

  :rtype: (:py:class:`DictionarySparseMatrix.DS` (Nx,Ny,Nz,1,nb), \
  :py:class:`DictionarySparseMatrix.DS` (Nx,Ny,Nz,na,nb))

  :return: Combper, Combpar

  '''
  Combper=Rper.dict_col_mult()
  Combpar=Rpar.dict_col_mult()
  return Combper, Combpar

def plot_grid(index=0):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  pstr=str('./Mesh/Power_grid'+str(int(Nra))+'Refs'+str(int(Nre))+'m'+str(index)+'.npy')
  pstrstd=str('./Mesh/Power_gridstd'+str(int(Nra))+'Refs'+str(int(Nre))+'m'+str(index)+'.npy')
  truestr=str('Parameters/True.npy')
  P3=np.load(truestr)
  #P3/=np.abs(np.average(P3))
  #print(P2)
  P=np.load(pstr)
  #P/=np.abs(np.average(P))
  #P2=np.load(pstrstd)
  #P2/=np.abs(np.average(P2))
  #Pdiff=np.zeros((P.shape),dtype=np.single)
  #Pdiff=abs(np.divide(P-P2,P, where=(abs(P)>epsilon)))
  Pdifftil=abs(np.divide(P-P3,P, where=(abs(P)>epsilon)))
  #Pdiffhat=abs(np.divide(P2-P3,P, where=(abs(P)>epsilon)))
  #err=np.sum(Pdiff)/(P.shape[0]*P.shape[1]*P.shape[2])
  #print('Residual GRL to std',err)
  err2=np.sum(Pdifftil)/(P.shape[0]*P.shape[1]*P.shape[2])
  print('Residual GRL to true',err2)
  #err3=np.sum(Pdiffhat)/(P.shape[0]*P.shape[1]*P.shape[2])
  #print('Residual of std to true',err3)
  n=P.shape[2]
  #n2=P2.shape[2]
  #n3=Pdiff.shape[2]
  lb=np.amin(P)
  #lb2=np.amin(P2)
  lb3=np.amin(P3)
  lb=min(lb,lb3)
  ub=np.amax(P)
  #ub2=np.amax(P2)
  ub3=np.amax(P3)
  ub=max(ub,ub3)
  if not os.path.exists('./GeneralMethodPowerFigures'):
    os.makedirs('./GeneralMethodPowerFigures')
  for i in range(n):
    mp.figure(i)
    #
    #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
    mp.imshow(P[:,:,i], cmap='viridis', vmax=ub,vmin=lb)
    mp.colorbar()
    filename=str('GeneralMethodPowerFigures/NoBoxPowerSliceNra'+str(int(Nra))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.eps')
    mp.savefig(filename)
    mp.clf()
  # for i in range(n2):
    # mp.figure(n+i)
    # #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
    # mp.imshow(P2[:,:,i], cmap='viridis',  vmax=ub,vmin=lb)
    # mp.colorbar()
    # filename=str('GeneralMethodPowerFigures/NoBoxPowerSliceNrastd'+str(int(Nra))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.eps')
    # mp.savefig(filename)
    # mp.clf()
  for i in range(n):
    mp.figure(n+i)
    #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
    mp.imshow(P3[:,:,i], cmap='viridis',  vmax=ub,vmin=lb)
    mp.colorbar()
    filename=str('GeneralMethodPowerFigures/NoBoxTrueSliceNra'+str(int(Nra))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.eps')
    mp.savefig(filename)
    mp.clf()
  # for i in range(n3):
    # mp.figure(n2+n+i)
    # #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
    # mp.imshow(Pdiff[:,:,i], cmap='viridis')#, vmax=max(ub2,ub),vmin=min(lb,lb2))
    # mp.colorbar()
    # filename=str('GeneralMethodPowerFigures/NoBoxPowerDiffSliceNra'+str(int(Nra))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.eps')
    # mp.savefig(filename)
    # mp.clf()
  for i in range(n):
    mp.figure(2*n+i)
    #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
    mp.imshow(Pdifftil[:,:,i], cmap='viridis')#, vmax=max(ub2,ub),vmin=min(lb,lb2))
    mp.colorbar()
    filename=str('GeneralMethodPowerFigures/NoBoxPowerDifftilSliceNra'+str(int(Nra))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.eps')
    mp.savefig(filename)
    mp.clf()
  # for i in range(n3):
    # mp.figure(n2+n+i)
    # #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
    # mp.imshow(Pdiffhat[:,:,i], cmap='viridis')#, vmax=max(ub2,ub),vmin=min(lb,lb2))
    # mp.colorbar()
    # filename=str('GeneralMethodPowerFigures/NoBoxPowerDiffhatSliceNra'+str(int(Nra))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.eps')
    # mp.savefig(filename)
    # mp.clf()
  #mp.show()
  return


if __name__=='__main__':
  np.set_printoptions(precision=3)
  print('Running  on python version')
  print(sys.version)
  #out=RayTracer() # To compute just the rays with no storage uncomment this line.
  timetest=1
  testnum=1
  roomnumstat=1
  Timemat=np.zeros((testnum,6))
  for j in range(0,timetest):
    Roomnum=roomnumstat
    #Timemat[0,0]=Roomnum
    for count in range(0,testnum):
      start=t.time()
      Mesh1=MeshProgram() # Shoot the rays and store the information
      mid=t.time()
      Grid=power_grid(Roomnum)  # Use the ray information to compute the power
      Q=Quality(Roomnum)
      # plot_grid()        # Plot the power in slices.
      end=t.time()
      Timemat[count,0]+=Roomnum
      Timemat[count,1]+=mid-start
      Timemat[count,2]+=(end-mid)/(Roomnum)
      if count !=0:
        Timemat[0,2]+=(end-mid)/(Roomnum)
      Timemat[count,3]+=end-start
      start=t.time()
      #for i in range(0,Roomnum):
      #  Mesh2=StdProgram(i) # Shoot the rays and store the information
      end=t.time()
      Timemat[count,4]+=end-start
      Timemat[count,5]+=(end-start)/(Roomnum)
      if count !=0:
        Timemat[0,5]+=(end-start)/(Roomnum)
      Roomnum*=2
      del Mesh1, Grid
  Timemat[0,2]/=(testnum)
  Timemat[0,5]/=(testnum)
  Timemat/=(timetest)
  plot_grid()        # Plot the power in slices.
  mp.clf()
  print('-------------------------------')
  print('Time to complete program')
  #print(Timemat)
  print('-------------------------------')
  Nra,Nre,h,L     =np.load('Parameters/Raytracing.npy')
  if not os.path.exists('./Times'):
    os.makedirs('./Times')
  timename=('./Times/TimesNra'+str(int(Nra))+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
  np.save(timename,Timemat)
  np.save('roomnumstat.npy',roomnumstat)
  np.save('Roomnum.npy',Roomnum)
  exit()

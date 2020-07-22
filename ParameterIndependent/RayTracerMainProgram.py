#!/usr/bin/env python3
# Updated Hayley Wragg 2020-06-30
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
import pickle
import csv

epsilon=sys.float_info.epsilon

def RayTracer():
  ''' Reflect rays and output the points of reflection.

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
  Nre,h,L,split     =np.load('Parameters/Raytracing.npy')
  Nre=int(Nre)
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy')             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Oblist        =np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
  #Nob           =len(Oblist)                                 # The number of obstacles in the room

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)

  for j in range(0,nra):
    # Calculate the Ray trajectories
    directionname=str('Parameters/Directions'+str(int(j))+'.npy')
    Direc         =np.load(directionname)         # Matrix of intial ray directions for Nra rays.
    print('Starting trajectory calculation')
    print('-------------------------------')
    Rays=Room.ray_bounce(Tx, Nre, Nra[j], Direc)
    print('-------------------------------')
    print('Trajectory calculation completed')
    print('Time taken',Room.time)
    print('-------------------------------')
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
    if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
    filename=str('./Mesh/'+plottype+'/RayPoints'+str(int(Nra[j]))+'Refs'+str(int(Nre))+'m.npy')
    np.save(filename,Rays)
    # The "Rays" file is Nra+1 x Nre+1 x 4 array containing the
    # co-ordinate and obstacle number for each reflection point corresponding
    # to each source ray.

  return 0


def MeshProgram(repeat=0,plottype=str()):
  ''' Reflect rays and output the Mesh containing ray information. \
  This mesh contains the distance rays have travelled and the angles of reflection.

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
  # Run the ParameterInput file, if this is a repeat run then we know
  # the parameters are already saved so this does not need to be run again.
  if repeat==1:
    pass
  else:
    out=PI.DeclareParameters()


  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)
  timesmat=np.zeros((nra))

  ##----Retrieve the environment--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')/L         # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy')/L             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')/L      # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  print('Number of obstacles',Nob)
  # This number will be used in further functions so resaving will ensure the correct number is used.
  np.save('Parameters/Nob.npy',Nob)

  # -------------Find the number of cells in the x, y and z axis.-------
  Nx=int(Room.maxxleng()/(h))
  Ny=int(Room.maxyleng()/(h))
  Nz=int(Room.maxzleng()/(h))

  #--------------Run the ray tracer for each ray number-----------------
  for j in range(0,nra):
    #------------Initialise the Mesh------------------------------------
    Mesh=DSM.DS(Nx,Ny,Nz,Nob*Nre+1,Nra[j]*(Nre+1),np.complex128,split)
    print('-------------------------------')
    print('Starting the ray bouncing and information storage')
    print('-------------------------------')
    t0=t.time()
    #-----------The input directions changes for each ray number.-------
    directionname=str('Parameters/Directions'+str(int(j))+'.npy')
    Direc=np.load(directionname)
    '''Find the ray points and the mesh storing their information
    The rays are reflected Nre times in directions Direc from Tx then
    the information about their paths is stored in Mesh.'''
    Rays, Mesh=Room.ray_mesh_bounce(Tx,Nre,Nra[j],Direc,Mesh,deltheta[j])
    if Mesh.check_nonzero_col(Nre,Nob):
      pass
    else:
      print('Too many nonzero terms in column')
    #Mesh,ind=Mesh.__del_doubles__(h,Nob)
    # if Mesh.check_nonzero_col(Nre):
      # pass
    # else:
      # print('Too many nonzero terms in column after del')
    t1=t.time()
    timesmat[j]=t1-t0
    #----------Save the Mesh for further calculations
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
    if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
    np.save('./Mesh/'+plottype+'/RayMeshPoints'+str(Nra[j])+'Refs'+str(Nre)+'m.npy',Rays)
    meshname=str('./Mesh/'+plottype+'/DSM'+str(Nra[j])+'Refs'+str(Nre)+'m.npy')
    Mesh.save_dict(meshname)
    meshnamecsv=str('./Mesh/RayMeshPoints'+str(Nra[j])+'Refs'+str(Nre)+'m.csv')
    myFile = open(meshnamecsv, 'w')
    with myFile:
      writer = csv.writer(myFile)
      writer.writerows(Rays)
    print('-------------------------------')
    print('Ray-launching complete')
    print('Time taken',t1-t0)
    print('-------------------------------')
  return Mesh

def StdProgram(plottype,index=0):
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
  Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  timemat=np.zeros((nra,1))
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy')/L             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
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
  for j in range(0,nra):
    directionname=str('Parameters/Directions'+str(int(j))+'.npy')
    Direc=np.load(directionname)
    gainname=str('Parameters/Tx'+str(int(Nra[j]))+'Gains'+str(index)+'.npy')
    Gt            = np.load(gainname)
    Rays, Grid=Room.ray_mesh_power_bounce(Tx,Nre,Nra[j],Direc,Mesh,Znobrat,refindex,Antpar,Gt,Pol,deltheta)
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
    if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
    np.save('./Mesh/'+plottype+'/RayMeshPointsstd'+str(int(Nra[j]))+'Refs'+str(int(Nre))+'m.npy',Rays*L)
    np.save('./Mesh/'+plottype+'/Power_gridstd'+str(Nra[j])+'Refs'+str(int(Nre))+'m'+str(int(index))+'.npy',Grid)
  # print('-------------------------------')
  # print('Ray-launching complete')
  # print('Time taken',Room.time)
  # print('-------------------------------')
  return Grid

def power_grid(repeat=0,plottype=str(),Roomnum=0):
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
  * (*LOS*)=`LOS.npy`, 1 if a Line of Sight Calculation and 0 if not.

  Method:
  * Initialise Grid using the number of x, y, and z steps in *Mesh*.
  * Use the function :py:func:`DictionarySparseMatrix.DS.power_compute`
  to compute the power.

  :rtype: Nx \ Ny x Nz numpy array of floats.

  :returns: Grid

  '''

  ##----Retrieve the Raytracing Parameters-----------------------------
  PerfRef    =np.load('Parameters/PerfRef.npy')
  LOS        =np.load('Parameters/LOS.npy')
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  timemat=np.zeros((nra,1))
  Nre=int(Nre)
  Nob            =np.load('Parameters/Nob.npy')

  # Initialise variable for counting zeros
  G_z=np.zeros((1,nra))

  #-------Run the power calculations for each ray number----------------
  for j in range(0,nra):
    ##----Retrieve the Mesh--------------------------------------
    meshname=str('./Mesh/'+plottype+'/DSM'+str(Nra[j])+'Refs'+str(Nre)+'m.npy')
    Mesh= DSM.load_dict(meshname)
    #print('power func')
    #print(Mesh[3,3,3])

    ##----Initialise Grid For Power-------------------------------------
    Nx=Mesh.Nx
    Ny=Mesh.Ny
    Nz=Mesh.Nz
    Ns=max(Nx,Ny,Nz)
    Grid=np.zeros((Nx,Ny,Nz),dtype=float)
    t0=t.time()
    #-------If multiple room variations are desired then run for all----
    for index in range(0,Roomnum):
      if repeat==0:
        PI.ObstacleCoefficients(index)
      ##----Retrieve the antenna parameters--------------------------------------
      gainname=str('Parameters/Tx'+str(Nra[j])+'Gains'+str(index)+'.npy')
      Gt            = np.load(gainname)
      freq          = np.load('Parameters/frequency'+str(index)+'.npy')
      Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')
      Pol           = np.load('Parameters/Pol'+str(index)+'.npy')

      ##----Retrieve the Obstacle Parameters--------------------------------------
      Znobrat      =np.load('Parameters/Znobrat'+str(index)+'.npy')
      refindex     =np.load('Parameters/refindex'+str(index)+'.npy')
      # Make the refindex, impedance and gains vectors the right length to
      # match the matrices.
      Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nob*Nre+1. Repeat Znobrat to match Mesh dimensions
      Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
      refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nob*Nre+1. Repeat refindex to match Mesh dimensions
      refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
      Gt=np.tile(Gt,(Nre+1,1))

      # Calculate the necessry parameters for the power calculation.
      c             =Freespace[3]            # Speed of Light
      Antpar        =np.load('Parameters/Antpar'+str(index)+'.npy')
      khat          =Antpar[0]
      lam           =Antpar[1]
      L             =Antpar[2]
      if index==0:
        Grid,RadA,RadB,ind=DSM.power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra[j],Nre,Ns,LOS,PerfRef)
      else:
        Grid,RadA,RadB,ind=DSM.power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra[j],Nre,Ns,LOS,PerfRef,ind)
      if not os.path.exists('./Mesh'):
        os.makedirs('./Mesh')
        os.makedirs('./Mesh/'+plottype)
      if not os.path.exists('./Mesh/'+plottype):
        os.makedirs('./Mesh/'+plottype)
      np.save('./Mesh/'+plottype+'/Power_grid'+str(int(Nra[j]))+'Refs'+str(Nre)+'m'+str(index)+'.npy',Grid)
      np.save('./Mesh/'+plottype+'/RadA_grid'+str(int(Nra[j]))+'Refs'+str(Nre)+'m'+str(index)+'.npy',RadA)
      np.save('./Mesh/'+plottype+'/RadB_grid'+str(int(Nra[j]))+'Refs'+str(Nre)+'m'+str(index)+'.npy',RadB) #.compressed())
      G_z[0,j]=np.count_nonzero((Grid==0))
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Power from DSM complete')
  print('Time taken',timemat)
  print('-------------------------------')
  return Grid,G_z

def Quality(plottype=str(),Roomnum=0):
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
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  Qmat=np.zeros((1,nra))
  timemat=np.zeros((nra,1))
  Nre=int(Nre)
  #Roomnum        =int(input('How many combinations of room values do you want to test?'))
  Nob            =np.load('Parameters/Nob.npy')
  LOS            =np.load('Parameters/LOS.npy')
  PerfRef        =np.load('Parameters/PerfRef.npy')


  for j in range(0,nra):
    ##----Retrieve the Mesh--------------------------------------
    meshname=str('./Mesh/'+plottype+'/DSM'+str(Nra[j])+'Refs'+str(Nre)+'m.npy')
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
      gainname=str('Parameters/Tx'+str(int(Nra[j]))+'Gains'+str(index)+'.npy')
      Gt            = np.load(gainname)
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
       Q,ind=DSM.quality_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra[j],Nre,Ns,LOS,PerfRef)
      else:
       Q,ind=DSM.quality_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra[j],Nre,Ns,LOS,PerfRef,ind)
      Qmat[0,j]=Q
      if not os.path.exists('./Quality'):
        os.makedirs('./Quality')
      if not os.path.exists('./Quality/'+plottype):
        os.makedirs('./Quality/'+plottype)
      np.save('./Quality/'+plottype+'/Quality'+str(Nra[j])+'Refs'+str(Nre)+'m'+str(index)+'.npy',Q)
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Quality from DSM complete', Qmat)
  print('Time taken',timemat)
  print('-------------------------------')
  truestr=str('Parameters/'+plottype+'/True.npy')
  P3=np.load(truestr)
  Q2=np.sum(P3)/(Mesh.Nx*Mesh.Ny*Mesh.Nz)
  return Qmat, Q2

def plot_grid(plottype=str(),index=0):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  for j in range(0,nra):
    pstr=str('./Mesh/'+plottype+'/Power_grid'+str(int(Nra[j]))+'Refs'+str(int(Nre))+'m'+str(index)+'.npy')
    RadAstr=str('./Mesh/'+plottype+'/RadA_grid'+str(int(Nra[j]))+'Refs'+str(int(Nre))+'m'+str(index)+'.npy')
    RadBstr=str('./Mesh/'+plottype+'/RadB_grid'+str(int(Nra[j]))+'Refs'+str(int(Nre))+'m'+str(index)+'.npy')
    pstrstd=str('./Mesh/'+plottype+'/Power_gridstd'+str(int(Nra[j]))+'Refs'+str(int(Nre))+'m'+str(index)+'.npy')
    truestr=str('Parameters/'+plottype+'/True.npy')#SinglePerfect.npy')
    P3  =np.load(truestr)
    P   =np.load(pstr)
    RadA=np.load(RadAstr)
    RadB=np.load(RadBstr)
    Pdifftil=abs(np.divide(P-P3,P, where=(abs(P)>epsilon)))  # Normalised Difference Mesh
    pdiffstr=str('./Mesh/'+plottype+'/PowerDiff_grid'+str(int(Nra[j]))+'Refs'+str(int(Nre))+'m'+str(index)+'.npy')
    np.save(pdiffstr,Pdifftil)
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
    rlb=min(np.amin(RadA),np.amin(RadB))
    ub=np.amax(P)
    #ub2=np.amax(P2)
    ub3=np.amax(P3)
    ub=max(ub,ub3)
    rub=max(np.amax(RadA),np.amax(RadB))
    if not os.path.exists('./GeneralMethodPowerFigures'):
      os.makedirs('./GeneralMethodPowerFigures')
    if not os.path.exists('./GeneralMethodPowerFigures/'+plottype):
      os.makedirs('./GeneralMethodPowerFigures/'+plottype)
    for i in range(n):
      mp.figure(i)
      #
      #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
      mp.imshow(P[:,:,i], cmap='viridis', vmax=ub,vmin=lb)
      mp.colorbar()
      rayfolder=str('./GeneralMethodPowerFigures/'+plottype+'/PowerSlice/Nra'+str(int(Nra[j])))
      if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/PowerSlice'):
        os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/PowerSlice')
        os.makedirs(rayfolder)
      elif not os.path.exists(rayfolder):
        os.makedirs(rayfolder)
      filename=str(rayfolder+'/NoBoxPowerSliceNra'+str(int(Nra[j]))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.jpg')#.eps')
      mp.savefig(filename)
      mp.clf()
      mp.figure(i)
      mp.imshow(RadA[:,:,i], cmap='viridis', vmax=rub,vmin=rlb)
      mp.colorbar()
      rayfolder=str('./GeneralMethodPowerFigures/'+plottype+'/RadSlice/Nra'+str(int(Nra[j])))
      if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/RadSlice'):
        os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/RadSlice')
        os.makedirs(rayfolder)
      elif not os.path.exists(rayfolder):
        os.makedirs(rayfolder)
      filename=str(rayfolder+'/NoBoxRadASliceNra'+str(int(Nra[j]))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.jpg')#.eps')
      mp.savefig(filename)
      mp.clf()
      mp.figure(i)
      mp.imshow(RadB[:,:,i], cmap='viridis', vmax=rub,vmin=rlb)
      mp.colorbar()
      filename=str(rayfolder+'/NoBoxRadBSliceNra'+str(int(Nra[j]))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.jpg')#.eps')
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
      mp.imshow(Pdifftil[:,:,i], cmap='viridis', vmax=1,vmin=0)
      mp.colorbar()
      Difffolder=str('./GeneralMethodPowerFigures/'+plottype+'/DiffSlice/Nra'+str(int(Nra[j])))
      if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/DiffSlice'):
        os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/DiffSlice')
        os.makedirs(Difffolder)
      elif not os.path.exists(Difffolder):
        os.makedirs(Difffolder)
      filename=str(Difffolder+'/NoBoxPowerDifftilSliceNra'+str(int(Nra[j]))+'Nref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.jpg')#.eps')
      mp.savefig(filename)
      mp.clf()
  for i in range(n):
      mp.figure(n+i)
      #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
      mp.imshow(P3[:,:,i], cmap='viridis',  vmax=ub,vmin=lb)
      mp.colorbar()
      truefolder=str('./GeneralMethodPowerFigures/'+plottype+'/TrueSlice')
      if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/TrueSlice'):
        os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/TrueSlice')
        os.makedirs(truefolder)
      elif not os.path.exists(truefolder):
        os.makedirs(truefolder)
      filename=str(truefolder+'/NoBoxTrueSliceNref'+str(int(Nre))+'slice'+str(int(i+1))+'of'+str(int(n))+'.jpg')#.eps')
      mp.savefig(filename)
      mp.clf()
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
  Nra =np.load('Parameters/Nra.npy')
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Qmat   =np.zeros((testnum,nra))
  Qtruemat=np.zeros((testnum,nra))
  G_zeros =np.zeros((testnum,nra))
  repeat=1

  for j in range(0,timetest):
    Roomnum=roomnumstat
    #Timemat[0,0]=Roomnum
    for count in range(0,testnum):
      start=t.time()
      Mesh1=MeshProgram(repeat,plottype) # Shoot the rays and store the information
      #('In main',Mesh1[3,3,3])
      mid=t.time()
      Grid,G_z=power_grid(repeat,plottype,Roomnum)  # Use the ray information to compute the power
      repeat=1
      G_zeros[count,:]=G_z
      #Q,Q2=Quality(plottype,Roomnum)
      #Qmat[count,:]=Q
      #Qtruemat[count,:]=Q2
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
      Roomnum*=1 #FIXME      to increase roomnumber

      #del Mesh1, Grid
  Timemat[0,2]/=(testnum)
  Timemat[0,5]/=(testnum)
  Timemat/=(timetest)
  plot_grid(plottype)        # Plot the power in slices.
  print('-------------------------------')
  print('Time to complete program')
  print(Timemat)
  print('-------------------------------')
  Nra         =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=Nra
  else:
      nra=Nra[0]
  Nre,h,L     =np.load('Parameters/Raytracing.npy')[0:3]
  if not os.path.exists('./Times'):
    os.makedirs('./Times')
  if not os.path.exists('./Times/'+plottype):
    os.makedirs('./Times/'+plottype)
  timename=('./Times/'+plottype+'/TimesNra'+str(int(nra))+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
  # if not os.path.exists('./Quality'):
    # os.makedirs('./Quality')
  # if not os.path.exists('./Quality/'+plottype):
    # os.makedirs('./Quality/'+plottype)
  # qualityname=('./Quality/'+plottype+'/QualityNra'+str(int(nra))+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
  # np.save(qualityname,Qmat[0,:])

  #print(Qmat)
  #mp.plot(Nra,Qmat[0,:])
  #mp.plot(Nra,Qtruemat[0,:])
  #filename=str('Quality/'+plottype+'Quality'+str(int(Nra[0]))+'to'+str(int(Nra[-1]))+'Nref'+str(int(Nre))+'.jpg')#.eps').
  #mp.savefig(filename)
  np.save(timename,Timemat)
  np.save('roomnumstat.npy',roomnumstat)
  np.save('Roomnum.npy',Roomnum)
  exit()

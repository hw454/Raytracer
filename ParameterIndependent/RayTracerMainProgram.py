#!/usr/bin/env python3
# Updated Hayley Wragg 2020-06-30
''' Code to trace rays around a room. This code uses:

  * the function :py:func:`RayTracer` to compute the points for \
  the ray trajectories.
  * the function :py:func:`MeshProgram` to compute the points for \
  the ray trajectories and iterate along the rays storing the \
  information in a :py:mod:`DictionarySparseMatrix.DS` and outputing \
  the points and mesh.
  * the function :py:func:`StdProgram` uses a standard ray-tracing approach\
  iterating along a ray and calculating the field as each step is taken \
  then converting to power at the end. This is for comparison.
  * the function :py:func:`power_grid` which loads the last saved power grid\
  and loads the antenna and obstacle physical parameters from \
  :py:func:`ParameterInput.ObstacleCoefficients`. It uses these and \
  the functions :py:func:`power_compute` in \
  :py:class:`DictionarySparseMatrix.DS` to compute the power in db \
  at all the non-zero positions of the grid.
  * the function :py:func:`Quality` uses the grid of values saved in \
  './Mesh/'+plottype+'/DSM'+str(Nra[j])+'Refs'+str(Nre)+'m.npy' and the function \
  :py:func:`quality_compute` in :py:mod:`DictionarySparseMatrix` to compute \
  the quality on the environment.

  * The ray points from :py:func:`RayTracer` are saved as:
    'RayPoints\ **Nra**\ Refs\ **Nre**\ n.npy' with **Nra** replaced by the \
    number of rays and **Nre** replaced by the number of reflections.
  * The ray points from :py:func:`MeshProgram` are saved as:
    'RayMeshPoints\ **Nra**\ Refs\ **Nre**\ n.npy' with **Nra** replaced \
    by the \
    number of rays and **Nre** replaced by the number of reflections. \
    The mesh is saved as 'DSM\ **Nra**\ Refs\ **Nre**\ m.npy'.

  When arrays are referrred to these are numpy arrays.

  '''
import numpy as np
import Room  as rom
import raytracerfunction as rayt
import sys
import ParameterLoad as PI
import DictionarySparseMatrix as DSM
import time as t
import os
import pickle
import csv
import logging
import pdb

epsilon=sys.float_info.epsilon
xcheck=1
ycheck=0
zcheck=0

def RayTracer():
  ''' Reflect rays and output the points of reflection.

  Parameters for the raytracer are input in \
  :py:func:`ParameterInput.DeclareParameters()` The raytracing \
  parameters defined in this function are saved and then loaded.

  * 'Raytracing.npy' - An array of shape (4,) of floats which is saved to \
  [Nra (number of rays), Nre (number of reflections), \
  h (relative meshwidth), \
  L (room length scale, the longest axis has been rescaled to 1 and this \
  is it's original length)]
  * 'Obstacles.npy'  - An array of shape (3,3) containing co-ordinates \
  forming triangles which form the obstacles. This is saved to Oblist \
  (The obstacles which are within the outerboundary )
  * 'Origin.npy'     - An array of shape (3,) for the co-ordinate of the source. \
  This is saved to Tx  (The location of the source antenna and origin \
  of every ray)
  * 'OuterBoundary.npy' - An array of shape (3,3) containing \
  co-ordinates forming triangles which form the obstacles. This is \
  saved to OuterBoundary   (The Obstacles forming the outer boundary of \
  the room )

  Put the two arrays of obstacles into one array

  .. code::

     Oblist=[Oblist,OuterBoundary]

  * 'Directions.npy' - An array of shape (Nra,3) containing the vectors which \
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
  Nre        =int(Nre)
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
    j=int(j)
    directionname='Parameters/Directions%d.npy'%j
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
    Nr=int(Nra[j])
    filename='./Mesh/'+plottype+'/RayPoints%dRefs%dm.npy'%(Nr,Nre)
    np.save(filename,Rays)
    # The "Rays" file is and array of shape (Nra+1 ,Nre+1 , 4)  containing the
    # co-ordinate and obstacle number for each reflection point corresponding
    # to each source ray.

  return 0


def MeshProgram(SN,repeat=0,plottype=str()):
  ''' Reflect rays and output the Mesh containing ray information. \
  This mesh contains the distance rays have travelled and the angles of reflection.

  Parameters for the raytracer are input in :py:mod:`ParameterInput`
  The raytracing parameters defined in this module are saved and then loaded.

  * 'Raytracing.npy' - An array of shape (4,) of floats which is saved to \
  [Nra (number of rays), Nre (number of reflections), \
  h (relative meshwidth), \
  L (room length scale, the longest axis has been rescaled to 1 and this \
  is it's original length)]
  * 'Obstacles.npy'  - An array of shape (3,3) containing co-ordinates \
  forming triangles which form the obstacles. This is saved to Oblist \
  (The obstacles which are within the outerboundary )
  * 'Origin.npy'     - An array of shape (3,) for the co-ordinate of the source. \
  This is saved to Tx  (The location of the source antenna and origin \
  of every ray)
  * 'OuterBoundary.npy' - An array of shape (3,3) containing \
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
    out=PI.DeclareParameters(SN)


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
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy').astype(float)         # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy').astype(float)  # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')             # Array of
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  NtriOut       =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  InnerOb       =np.load('Parameters/InnerOb.npy')              # Whether the innerobjects should be included or not.
  if InnerOb:
    Oblist=np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
    Ntri=np.append(NtriOb,NtriOut)
  else:
    Oblist=OuterBoundary
    Ntri=NtriOut
  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist,Ntri)
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
    j=int(j)
    #------------Initialise the Mesh------------------------------------
    Mesh=DSM.DS(Nx,Ny,Nz,Nob*Nre+1,Nra[j]*(Nre+1),np.complex128,split)
    print('-------------------------------')
    print('Starting the ray bouncing and information storage')
    print('-------------------------------')
    t0=t.time()
    #-----------The input directions changes for each ray number.-------
    directionname='Parameters/Directions%d.npy'%j
    Direc=np.load(directionname)
    '''Find the ray points and the mesh storing their information
    The rays are reflected Nre times in directions Direc from Tx then
    the information about their paths is stored in Mesh.'''
    Rays, Mesh=Room.ray_mesh_bounce(Tx,Nre,Nra[j],Direc,Mesh,deltheta[j])
    if Mesh.check_nonzero_col(Nre,Nob):
      pass
    else:
      raise ValueError('Too many nonzero terms in column')
    #logging.info('Before doubles deleted'+str(Mesh[xcheck,ycheck,zcheck]))
    #print('Deleting doubles')
    Mesh,ind=Mesh.__del_doubles__(h,Nob,Ntri=Room.Ntri)
    #logging.info('After doubles deleted'+str(Mesh[xcheck,ycheck,zcheck]))
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
    Nr=int(Nra[j])
    rayname='./Mesh/'+plottype+'/RayMeshPoints%dRefs%dm.npy'%(Nr,Nre)
    np.save(rayname,Rays)
    meshname='./Mesh/'+plottype+'/DSM%dRefs%dm.npy'%(Nr,Nre)
    Mesh.save_dict(meshname)
    meshnamecsv='./Mesh/'+plottype+'RayMeshPoints%dRefs%dm.csv'%(Nr,Nre)
    myFile = open(meshnamecsv, 'w')
    #with myFile:
    #  writer = csv.writer(myFile)
    #  writer.writerows(Rays)
    print('-------------------------------')
    print('Ray-launching complete')
    print('Time taken',t1-t0)
    print('-------------------------------')
  return Mesh

def StdProgram(plottype,index=0):
  ''' Refect rays and input object information output the power.

  Parameters for the raytracer are input in :py:mod:`ParameterInput`
  The raytracing parameters defined in this module are saved and then loaded.

  * 'Raytracing.npy' - An array of shape (4,) floats which is saved to \
  [Nra (number of rays), Nre (number of reflections), \
  h (relative meshwidth), \
  L (room length scale, the longest axis has been rescaled to 1 and this \
  is it's original length)]
  * 'Obstacles.npy'  - An array of shape (3,3) containing co-ordinates \
  forming triangles which form the obstacles. This is saved to Oblist \
  (The obstacles which are within the outerboundary )
  * 'Origin.npy'     - An array of shape (3,) for the co-ordinate of the source. \
  This is saved to Tx  (The location of the source antenna and origin \
  of every ray)
  * 'OuterBoundary.npy' - An array of shape (3,3) containing \
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
  Nre,h,L,split =np.load('Parameters/Raytracing.npy')
  Nra           =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=1
      Nra=np.array([Nra])
  else:
      nra=len(Nra)
  timemat=np.zeros((nra,1))
  Nre=int(Nre)

  ##----Retrieve the environment parameters--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin.npy').astype(float)         # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy').astype(float)  # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')             # Array of
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  NtriOut       =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  InnerOb       =np.load('Parameters/InnerOb.npy')              # Whether the innerobjects should be included or not.
  if InnerOb:
    Oblist=np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
    Ntri=np.append(NtriOb,NtriOut)
  else:
    Oblist=OuterBoundary
    Ntri=NtriOut

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  #np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
  ##----Retrieve the Raytracing Parameters-----------------------------
  PerfRef    =np.load('Parameters/PerfRef.npy')
  LOS        =np.load('Parameters/LOS.npy')
  freq          = np.load('Parameters/frequency%d.npy'%index)
  Freespace     = np.load('Parameters/Freespace%d.npy'%index)
  Pol           = np.load('Parameters/Pol%d.npy'%index)
  c             =Freespace[3]
  Antpar        =np.load('Parameters/Antpar%d.npy'%index)
  khat          =Antpar[0]
  lam           =Antpar[1]
  L             =Antpar[2]

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%d.npy'%index)
  refindex     =np.load('Parameters/refindex%d.npy'%index)

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
    directionname=str('Parameters/Directions%d.npy'%j)
    Direc=np.load(directionname)
    Nr=int(Nra[j])
    gainname      ='Parameters/Tx%dGains%d.npy'%(Nr,index)
    Gt            = np.load(gainname)
    Rays, Grid=Room.ray_mesh_power_bounce(Tx,Nre,Nra[j],Direc,Mesh,Znobrat,refindex,Antpar,Gt,Pol,deltheta)
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
    if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
    stdraypointname='./Mesh/'+plottype+'/RayMeshPointsstd%dRefs%dm.npy'%(Nr,Nre)
    np.save(stdraypointname,Rays*L)
    stdGridname='./Mesh/'+plottype+'/Power_gridstd%dRefs%dm%d.npy'%(Nr,Nre,index)
    np.save(stdGridname,Grid)
  # print('-------------------------------')
  # print('Ray-launching complete')
  # print('Time taken',Room.time)
  # print('-------------------------------')
  return Grid

def power_grid(SN,repeat=0,plottype=str(),Roomnum=0):
  ''' Calculate the field on a grid using enviroment parameters and the \
  ray Mesh.

  Loads:

  * (*Nra*\ = number of rays, *Nre*\ = number of reflections, \
  *h*\ = meshwidth, *L*\ = room length scale, *split*\ =number of steps through a mesh element)\
  =`Paramters/Raytracing.npy`
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

  :rtype: numpy array of shape (Nx,Ny,Nz)

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

  print('---------------------------------')
  print('Starting the power calculation')
  #-------Run the power calculations for each ray number----------------
  for j in range(0,nra):
      ##----Retrieve the Mesh--------------------------------------
    Nr=int(Nra[j])
    meshname='./Mesh/'+plottype+'/DSM%dRefs%dm.npy'%(Nr,Nre)
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
      print(index,Roomnum)
      if repeat==0:
        PI.ObstacleCoefficients(SN,index)
      ##----Retrieve the antenna parameters--------------------------------------
      gainname      ='Parameters/Tx%dGains%d.npy'%(Nr,index)
      Gt            = np.load(gainname)
      freq          = np.load('Parameters/frequency%d.npy'%index)
      Freespace     = np.load('Parameters/Freespace%d.npy'%index)
      Pol           = np.load('Parameters/Pol%d.npy'%index)

      ##----Retrieve the Obstacle Parameters--------------------------------------
      Znobrat      =np.load('Parameters/Znobrat%d.npy'%index)
      refindex     =np.load('Parameters/refindex%d.npy'%index)
      # Make the refindex, impedance and gains vectors the right length to
      # match the matrices.
      Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nob*Nre+1. Repeat Znobrat to match Mesh dimensions
      Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
      refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nob*Nre+1. Repeat refindex to match Mesh dimensions
      refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
      Gt=np.tile(Gt,(Nre+1,1))

      # Calculate the necessry parameters for the power calculation.
      c             =Freespace[3]            # Speed of Light
      Antpar        =np.load('Parameters/Antpar%d.npy'%index)
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
      np.save('./Mesh/'+plottype+'/Power_grid%dRefs%dm%d.npy'%(Nr,Nre,index),Grid)
      np.save('./Mesh/'+plottype+'/RadA_grid%dRefs%dm%d.npy'%(Nr,Nre,index),RadA)
      np.save('./Mesh/'+plottype+'/RadB_grid%dRefs%dm%d.npy'%(Nr,Nre,index),RadB)
      G_z[0,j]=np.count_nonzero((Grid==0))
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Power from DSM complete')
  print('Time taken',timemat)
  print('-------------------------------')
  return Grid,G_z

def Residual(plottype=str(),Roomnum=0):
  ''' Compute the residual between the computed mesh and the true mesh summed over x,y,z and averaged
  '''
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  LOS=np.load('Parameters/LOS.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  err=np.zeros(nra)
  for index in range(0,Roomnum):
    for j in range(0,nra):
      Nr=int(Nra[j])
      Nre=int(Nre)
      pstr       ='./Mesh/'+plottype+'/Power_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
      truestr    ='Mesh/True/'+plottype+'/True.npy'
      Pt   =np.load(truestr)
      Pthat=DSM.db_to_Watts(Pt)
      P    =np.load(pstr)
      Phat =DSM.db_to_Watts(P)
      rat  =Phat/Pthat
      err[j]+=DSM.Watts_to_db(np.sum(Phat/Pthat)/(P.shape[0]*P.shape[1]*P.shape[2]))
      pratstr='./Mesh/'+plottype+'/PowerRat_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
      np.save(pratstr,rat)
      ResStr  ='./Errors/'+plottype+'/Residual%dRefs%dm%d.npy'%(Nr,Nre,index)
      np.save(ResStr,err[j])
  return err

def Quality(SN,repeat=0,plottype=str(),Roomnum=0):
  ''' Calculate the field on a grid using enviroment parameters and the \
  ray Mesh.

  Loads:

  * (*Nra*\ = number of rays, *Nre*\ = number of reflections, \
  *h*\ = meshwidth, *L*\ = room length scale, *split*\ =number of steps through a mesh element)\
  =`Paramters/Raytracing.npy`
  * *P*\ = power grid.
  * *Ptr*\ = true power grid.

  Method:
  * Use the function :py:func:`DictionarySparseMatrix.QualityFromPower(P)`
  to compute the power.

  :rtype: A numpy array of floats with shape (nra,) where nra is the number of rays.

  :returns: Grid

  '''

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  Qmat=np.zeros(nra)
  timemat=np.zeros(nra)
  Nre=int(Nre)

  for j in range(0,nra):
    ##----Retrieve the Mesh--------------------------------------
    Nr=int(Nra[j])
    meshname='./Mesh/'+plottype+'/DSM%dRefs%dm.npy'%(Nr,Nre)
    Mesh= DSM.load_dict(meshname)

    ##----Initialise Grid For Power------------------------------------------------------
    Nx=Mesh.Nx
    Ny=Mesh.Ny
    Nz=Mesh.Nz
    Ns=max(Nx,Ny,Nz)
    Grid=np.zeros((Nx,Ny,Nz),dtype=float)
    t0=t.time()
    for index in range(0,Roomnum):
      pstr       ='./Mesh/'+plottype+'/Power_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
      P=np.load(pstr)
      Q=DSM.QualityFromPower(P)
      Qmat[j]=Q
      if not os.path.exists('./Quality'):
        os.makedirs('./Quality')
      if not os.path.exists('./Quality/'+plottype):
        os.makedirs('./Quality/'+plottype)
      np.save('./Quality/'+plottype+'/Quality%dRefs%dm%d.npy'%(Nr,Nre,index),Q)
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Quality from DSM complete', Qmat)
  print('Time taken',timemat)
  print('-------------------------------')
  truestr='Mesh/True/'+plottype+'/True.npy'
  P3=np.load(truestr)
  Q2=DSM.QualityFromPower(P3)
  return Qmat, Q2


if __name__=='__main__':
  np.set_printoptions(precision=3)
  print('Running  on python version')
  print(sys.version)
  Sheetname='InputSheet.xlsx'
  out=PI.DeclareParameters(Sheetname)
  #out=RayTracer() # To compute just the rays with no storage uncomment this line.
  timetest    =np.load('Parameters/timetest.npy')
  testnum    =np.load('Parameters/testnum.npy')
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  Nra =np.load('Parameters/Nra.npy')
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  Timemat=np.zeros((testnum,6))
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Qmat   =np.zeros((testnum,nra))
  Qtruemat=np.zeros((testnum,nra))
  G_zeros =np.zeros((testnum,nra)) # Number of nonzero terms
  Reserr  =np.zeros((testnum,nra))
  repeat=1
  logname='RayTracer'+plottype+'.log'
  j=1
  while os.path.exists(logname):
    logname='RayTracer'+plottype+'%d.log'%j
    j+=1
  logging.basicConfig(filename=logname,filemode='w',format="[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
  logging.info(sys.version)
  for j in range(0,timetest):
    Roomnum=roomnumstat
    #Timemat[0,0]=Roomnum
    for count in range(0,testnum):
      start=t.time()
      Mesh1=MeshProgram(Sheetname,repeat,plottype) # Shoot the rays and store the information
      #('In main',Mesh1[3,3,3])
      mid=t.time()
      Grid,G_z=power_grid(Sheetname,repeat,plottype,Roomnum)  # Use the ray information to compute the power
      repeat=1
      G_zeros[count,:]=G_z
      Q,Q2=Quality(Sheetname,repeat,plottype,Roomnum)
      Qmat[count,:]=Q
      Qtruemat[count,:]=Q2
      end=t.time()
      Reserr[count,:]+=Residual(plottype,Roomnum)/Roomnum
      Timemat[count,0]+=Roomnum
      Timemat[count,1]+=mid-start
      Timemat[count,2]+=(end-mid)/(Roomnum)
      if count !=0:
        Timemat[0,2]+=(end-mid)/(Roomnum)
      Timemat[count,3]+=end-start
      start=t.time()
      for i in range(0,Roomnum):
        Mesh2=StdProgram(i) # Shoot the rays and store the information
      end=t.time()
      Timemat[count,4]+=end-start
      Timemat[count,5]+=(end-start)/(Roomnum)
      #if count !=0:
      #  Timemat[0,5]+=(end-start)/(Roomnum)
      Roomnum*=2 #FIXME      to increase roomnumber

      #del Mesh1, Grid
  Timemat[0,2]/=(testnum)
  Timemat[0,5]/=(testnum)
  Timemat/=(timetest)
  Reserr/=(timetest)
  print('-------------------------------')
  print('Time to complete program') # Roomnum, ray time, average power time, total time, total time averaged by room
  print(Timemat)
  print('-------------------------------')
  print('-------------------------------')
  print('Residual to the True calculation')
  print(Reserr)
  print('-------------------------------')
  Nra         =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=1
  else:
      nra=len(Nra)
  Nre,h,L     =np.load('Parameters/Raytracing.npy')[0:3]
  if not os.path.exists('./Times'):
    os.makedirs('./Times')
  if not os.path.exists('./Times/'+plottype):
    os.makedirs('./Times/'+plottype)
  timename='./Times/'+plottype+'/TimesNra%dRefs%dRoomnum%dto%d.npy'%(nra,Nre,roomnumstat,Roomnum)

  if not os.path.exists('./Quality'):
    os.makedirs('./Quality')
  if not os.path.exists('./Quality/'+plottype):
    os.makedirs('./Quality/'+plottype)
  if not os.path.exists('./Errors'):
    os.makedirs('./Errors')
  if not os.path.exists('./Errors/'+plottype):
    os.makedirs('./Errors/'+plottype)

  for j in range(testnum):
    qualityname=('./Quality/'+plottype+'/QualityNrays'+str(int(nra))+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
    np.save(qualityname,Qmat[j,:])
    mp.figure(2*j)
    mp.plot(Nra,Qmat[j,:])
    mp.plot(Nra,Qtruemat[j,:])
    filename=str('Quality/'+plottype+'/Quality'+str(int(Nra[0]))+'to'+str(int(Nra[-1]))+'Nref'+str(int(Nre))+'.jpg')#.eps').
    mp.savefig(filename)
    errorname=('./Errors/'+plottype+'/ErrorsNrays'+str(int(nra))+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
    np.save(qualityname,Reserr[j,:])
    mp.figure(2*j+1)
    mp.plot(Nra,Reserr[j,:])
    filename=str('./Errors/'+plottype+'/Residual'+str(int(Nra[0]))+'to'+str(int(Nra[-1]))+'Nref'+str(int(Nre))+'.jpg')#.eps').
    mp.savefig(filename)
  np.save(timename,Timemat)
  np.save('roomnumstat.npy',roomnumstat)
  np.save('Roomnum.npy',Roomnum)
  exit()

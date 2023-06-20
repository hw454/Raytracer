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
import openpyxl as wb
import xlrd as rd
from itertools import product
import num2words as nw

epsilon=sys.float_info.epsilon
xcheck=2
ycheck=9
zcheck=5
dbg=0
if dbg:
  logon=1
else:
  logon=np.load('Parameters/logon.npy')

def RayTracer(job=0):
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
  nra=1
  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job) # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Oblist        =np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
  #Nob           =len(Oblist)                                 # The number of obstacles in the room

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)

  for j in range(0,nra):
    # Calculate the Ray trajectories
    j=int(j)
    Nr=Nra[j]
    directionname='Parameters/Directions%d.npy'%Nr
    Direc         =np.load(directionname)         # Matrix of intial ray directions for Nra rays.
    print('Starting trajectory calculation')
    print('-------------------------------')
    Rays=Room.ray_bounce(Tx, Nre, Nr, Direc)
    print('-------------------------------')
    print('Trajectory calculation completed')
    print('Time taken',Room.time)
    print('-------------------------------')
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
    if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
    filename='./Mesh/'+plottype+'/RayPoints%03dRefs%03dm_tx%03d.npy'%(Nr,Nre,job)
    np.save(filename,Rays)
    # The "Rays" file is and array of shape (Nra+1 ,Nre+1 , 4)  containing the
    # co-ordinate and obstacle number for each reflection point corresponding
    # to each source ray.

  return 0


def MeshProgram(Nr=22,index=1,job=0,Nre=3,PerfRef=0,LOS=0,InnerOb=0,Nrs=2,Ns=5):
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

    Mesh=DSM.DS(Nx,Ny,Nz,int(Nsur*Nre+1),int((Nre)*(Nra)+1))

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


  ##----Retrieve the Raytracing Parameters-----------------------------
  t0=t.time()
  h=1.0/Ns
  _,_,L,split    =np.load('Parameters/Raytracing.npy')
  splitinv=1.0/split
  # Nra        =np.load('Parameters/Nra.npy')
  # if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      # Nra=np.array([Nra])
      # nra=1
  # else:
      # nra=len(Nra)
  Nre=int(Nre)
  timesmat=np.zeros(1)

  ##----Retrieve the environment--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)# The location of the source antenna (origin of every ray)
  #OuterBoundary =np.load('Parameters/OuterBoundary.npy').astype(float)  # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')             # Array of
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  AngChan       =np.load('Parameters/AngChan.npy')              # switch for whether angles should be corrected for the received points on cones and voxel centre.
  if Nre>1:
    Refstr=nw.num2words(Nre)+'Ref'
  else:
    Refstr='NoRef'
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index).astype(float)      # The obstacles which are within the outerboundary
  Nsur       =np.load('Parameters/Nsur%d.npy'%index)
  refindex=np.load('Parameters/refindex%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  MaxInter     =np.load('Parameters/MaxInter%d.npy'%index)
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  refindex     =np.load('Parameters/refindex%03d.npy'%index)
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  obnumbers=np.zeros((Nrs,1))
  k=0
  Obstr=''
  if 0<Nrs<Nsur:
    obnumbers=np.zeros(Nrs)
    for ob, refind in enumerate(refindex):
      if abs(refind)>epsilon:
        obnumbers[k]=ob
        k+=1
        Obstr=Obstr+'Ob%02d'%ob
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'PerfRef'
      else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'Ref'
      else:
        LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  foldtype=Refstr+boxstr
  #Room contains all the obstacles and walls.
  if InnerOb:
    Ntri=np.append(Ntri,NtriOb)
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Room.__set_MaxInter__(MaxInter)
  np.save('Parameters/Nob%d.npy'%index,Nob)
  np.save('Parameters/Nsur%d.npy'%index,Nsur)
  Nsur=int(Room.Nsur)
  Nx=int(Room.maxxleng()//h+1)
  Ny=int(Room.maxyleng()//h+1)
  Nz=int(Room.maxzleng()//h+1)
  Ns=max(Nx,Ny,Nz)
  #for j in range(0,nra):
  #Nr=Nra[j]
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  meshname=meshfolder+'/DSM_tx%03d'%(job)
  print(meshname+'%02dx%02dy%02dz.npz'%(0,0,0))
  if os.path.isfile(meshname+'%02dx%02dy%02dz.npz'%(0,0,0)):
    Mesh=DSM.load_dict(meshname,Nx,Ny,Nz)
    print('Tx',Tx)
    print('Mesh loaded from store')
    timemat=t.time()-t0
    return Mesh,timesmat,Room
  #------------Initialise the Mesh------------------------------------
  Mesh=DSM.DS(Nx,Ny,Nz,Nsur*Nre+1,Nr*(Nre+1),np.complex128,split)
  if not Room.CheckTxInner(Tx):
    if logon:
        logging.info('Tx=(%f,%f,%f) is not a valid transmitter location'%(Tx[0],Tx[1],Tx[2]))
    print('This is not a valid transmitter location')
    timemat=t.time()-t0
    return Mesh,timesmat,Room
  print('-------------------------------')
  print('Starting the ray bouncing and information storage')
  print('-------------------------------')
  #-----------The input directions changes for each ray number.-------
  directionname='Parameters/Directions%03d.npy'%Nr
  Direc=np.load(directionname)
  '''Find the ray points and the mesh storing their information
  The rays are reflected Nre times in directions Direc from Tx then
  the information about their paths is stored in Mesh.'''
  if logon:
      logging.info('Tx at(%f,%f,%f)'%(Tx[0],Tx[1],Tx[2]))
  print('Tx',Tx)
  if Nr==22:
     j=0
  else:
     j=1
  programterms=np.array([Nr,Nre,AngChan,split,splitinv,deltheta[j]])
  Rays, Mesh=Room.ray_mesh_bounce(Tx,Direc,Mesh,programterms)
  if dbg:
      assert Mesh.check_nonzero_col(Nre,Nsur)
      print('before del',Mesh[xcheck,ycheck,zcheck])
      for l,m in zip(Mesh[xcheck,ycheck,zcheck].nonzero()[0],Mesh[xcheck,ycheck,zcheck].nonzero()[1]):
        print(l,m,abs(Mesh[xcheck,ycheck,zcheck][l,m]))
      Mesh,ind=Mesh.__del_doubles__(h,Nsur,Ntri=Room.Ntri)
      print('after del',Mesh[xcheck,ycheck,zcheck])
  #----------Save the Mesh for further calculations
  if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+foldtype)
      os.makedirs(meshfolder)
  if not os.path.exists('./Mesh/'+foldtype):
      os.makedirs('./Mesh/'+foldtype)
      os.makedirs(meshfolder)
  if not os.path.exists(meshfolder):
      os.makedirs(meshfolder)
  rayname='./Mesh/'+foldtype+'/RayMeshPoints%03dRefs%03d_tx%03d.npy'%(Nr,Nre,job)
  np.save(rayname,Rays)
  Mesh.save_dict(meshname)
  t1=t.time()
  timesmat[0]=t1-t0
  print('-------------------------------')
  print('Ray-launching complete')
  print('Time taken',t1-t0)
  print('-------------------------------')
  return Mesh,timesmat,Room

def StdProgram(SN,index=1,job=0,Nre=3,PerfRef=0,LOS=0,InnerOb=0):
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

    Mesh=DSM.DS(Nx,Ny,Nz,int(Nsur*Nre+1),int((Nre)*(Nra)+1))

  Find the reflection points of the rays and store the power Use the \
  py:func:`Room.room.ray_mesh_bounce` function.

  .. code::

     Rays, Mesh=Room.ray_mesh_power_bounce(Tx,int(Nre),int(Nra),Direc,Mesh)

  Save the reflection points in Rays to \
  'RayMeshPoints\ **Nra**\ Refs\ **Nre**\ n.npy' making the \
  substitution for **Nra** and **Nre** with their parameter values.

  :return: Mesh

  '''
  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates
  ##----Retrieve the Raytracing Parameters-----------------------------
  _,h,L,split    =np.load('Parameters/Raytracing.npy')
  splitinv=1.0/split
  # Nra        =np.load('Parameters/Nra.npy')
  # if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      # Nra=np.array([Nra])
      # nra=1
  # else:
      # nra=len(Nra)
  timesmat=np.zeros(1)

  ##----Retrieve the environment--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)         # The location of the source antenna (origin of every ray)
  #OuterBoundary =np.load('Parameters/OuterBoundary.npy').astype(float)  # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')             # Array of
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index).astype(float)      # The obstacles which are within the outerboundary
  refindex      =np.load('Parameters/refindex%03d.npy'%index)
  Pol           =np.load('Parameters/Pol%03d.npy'%index)
  Nx=int(Room.maxxleng()//h+1)
  Ny=int(Room.maxyleng()//h+1)
  Nz=int(Room.maxzleng()//h+1)
  Ns=max(Nx,Ny,Nz)
  if Nr==22:
    j=0
  else:
    j=1

  # Room contains all the obstacles and walls as well as the number of max intersections any ray segment can have,
  #the number of obstacles (each traingle is separate in this list), number of triangles making up each surface,
  #the number of surfaces.
  Nob=Room.Nob
  Nsur=Room.Nsur
  print('Number of obstacles',Nob)
  # This number will be used in further functions so resaving will ensure the correct number is used.

  Mesh=np.zeros((Nx,Ny,Nz,2),dtype=np.complex128)
  # There are two terms in each grid point to account for the polarisation directions.
  print('-------------------------------')
  print('Mesh for Std program built')
  print('-------------------------------')
  print('Starting the ray bouncing and field storage')
  print('-------------------------------')
  #for j in range(0,nra):
  t0=t.time()
  #Nr=Nra[j]
  # Indicates whether all terms after reflection should be ignored and line of sight modelled (1). (0) if reflection should be included.
  # The permmittivty and permeability of air
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  # Polarisation of the antenna.
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  # Antenna parameters=[khat,lam,L]
  # Length scale of the environment. (length of the longest axis)
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  # Ratio of the impedance of each obstacle divided by the impedance of air.
  refindex     =np.load('Parameters/refindex%03d.npy'%index)
  # The refractive index of each obstacle
  Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
  refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
  if not Room.CheckTxInner(Tx): # Check that the transmitter location is not inside an obstacle
      if logon:
        logging.info('Tx=(%f,%f,%f) is not a valid transmitter location'%(Tx[0],Tx[1],Tx[2]))
      print('This is not a valid transmitter location')
      return Mesh,timesmat,Room
  directionname=str('Parameters/Directions%03d.npy'%(Nr))
  Direc=np.load(directionname)
  gainname      ='Parameters/Tx%03dGains%03d.npy'%(Nr,index)
  Gt            = np.load(gainname)
  programterms=np.array([Nr,Nr,AngChan,split,splitinv,deltheta[j]])
  Rays, Grid=Room.ray_mesh_power_bounce(Tx,Direc,Mesh,Znobrat,refindex,Antpar,Gt,Pol,programterms)
  if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
  if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
  stdraypointname='./Mesh/'+plottype+'/RayMeshPointsstd%03dRefs%03dm_tx%03d.npy'%(Nr,Nre,job)
  np.save(stdraypointname,Rays*L)
  stdGridname='./Mesh/'+plottype+'/Power_gridstd%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
  np.save(stdGridname,Grid)
  t1=t.time()
  timemat[0]=t1-t0
  print('-------------------------------')
  print('Ray-launching std complete')
  print('Time taken',t1-t0)
  print('-------------------------------')
  return Grid,timemat

def power_grid(Room,Mesh,Nr=22,index=0,job=0,Nre=3,PerfRef=0,LOS=0,InnerOb=0,Nrs=2,Ns=5):
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
  _,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)# The location of the source antenna (origin of every ray)
  #-------If multiple room variations are desired then run for all----
  refindex=np.load('Parameters/refindex%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  refindex     =np.load('Parameters/refindex%03d.npy'%index)
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  h=1.0/Ns
  obnumbers=np.zeros((Nrs,1))
  k=0
  Obstr=''
  Nsur=Room.Nsur
  if 0<Nrs<Nsur:
    obnumbers=np.zeros(Nrs)
    for ob, refind in enumerate(refindex):
      if abs(refind)>epsilon:
        obnumbers[k]=ob
        k+=1
        Obstr=Obstr+'Ob%02d'%ob
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'PerfRef'
      else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'Ref'
      else:
        LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  if Nre>1:
    Refstr=nw.num2words(Nre)+'Ref'
  else:
    Refstr='NoRef'
  foldtype=Refstr+boxstr
  # if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      # nra=np.array([Nra])
  # else:
      # nra=len(Nra)
  # nra=1
  if Nr==22:
    j=0
  else:
    j=1
  timemat=np.zeros(1)
  Nre=int(Nre)
  if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
    loca='Centre'
  else:
    loca='OffCentre'
  plottype=LOSstr+boxstr+loca
  # Initialise variable for counting zeros
  Nsur=int(Room.Nsur)
  Nx=int(Room.maxxleng()//h+1)
  Ny=int(Room.maxyleng()//h+1)
  Nz=int(Room.maxzleng()//h+1)
  if not Room.CheckTxInner(Tx):
    if logon:
        logging.info('Tx=(%f,%f,%f) is not a valid transmitter location'%(Tx[0],Tx[1],Tx[2]))
    print('This is not a valid transmitter location')
    timemat=0
    Grid=np.zeros((Nx,Ny,Nz))
    return Grid,timemat
  Ns=max(Nx,Ny,Nz)
  print('---------------------------------')
  print('Starting the power calculation')
  #-------Run the power calculations for each ray number----------------
  #for j in range(0,nra):
      ##----Retrieve the Mesh--------------------------------------
  #  Nr=int(Nra[j])
  t0=t.time()
  ##----Initialise Grid For Power-------------------------------------
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  pstr=powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
  # if os.path.isfile(pstr):
    # Grid= np.load(pstr)
    # print('Power loaded from store')
    # print('Power file'+pstr)
  # else:
  c=1
  if c:
    # Make the refindex, impedance and gains vectors the right length to
    # match the matrices.
    Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1. Repeat Znobrat to match Mesh dimensions
    Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
    refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nsur*Nre+1. Repeat refindex to match Mesh dimensions
    refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
    # Calculate the necessry parameters for the power calculation.
    Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
    gainname      ='Parameters/Tx%03dGains%03d.npy'%(Nr,index)
    Gt            = np.load(gainname)
    Grid,ind=DSM.power_compute(foldtype,plottype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,job,index,LOS,PerfRef)
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
      os.makedirs(powerfolder)
    if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
      os.makedirs(powerfolder)
    if not os.path.exists(powerfolder):
      os.makedirs(powerfolder)
    np.save(pstr,Grid)
    print('Power grid saved at, ',pstr)
  t1=t.time()
  timemat[0]=t1-t0
  print('-------------------------------')
  print('Job number',job)
  print('Power from DSM complete')
  print('Time taken',timemat)
  print('-------------------------------')
  return Grid,timemat

def optimum_gains(Room,Mesh,Nr=22,index=0,job=0,Nre=3,PerfRef=0,LOS=0,InnerOb=0,Nrs=2,Ns=5):
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
  t0=t.time()
  ##----Retrieve the Raytracing Parameters-----------------------------
  _,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)# The location of the source antenna (origin of every ray)
  Nra        =np.load('Parameters/Nra.npy')
  #-------If multiple room variations are desired then run for all----
  refindex=np.load('Parameters/refindex%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  refindex     =np.load('Parameters/refindex%03d.npy'%index)
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  h=1.0/Ns
  obnumbers=np.zeros((Nrs,1))
  k=0
  Obstr=''
  Nsur=Room.Nsur
  if 0<Nrs<Nsur:
    obnumbers=np.zeros(Nrs)
    for ob, refind in enumerate(refindex):
      if abs(refind)>epsilon:
        obnumbers[k]=ob
        k+=1
        Obstr=Obstr+'Ob%02d'%ob
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'PerfRef'
      else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'Ref'
      else:
        LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  # if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      # nra=np.array([Nra])
  # else:
      # nra=len(Nra)
  timemat=np.zeros(1)
  Nre=int(Nre)
  if Nre>1:
    Refstr=nw.num2words(Nre)+''
  else:
    Refstr='NoRef'
  if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
    loca='Centre'
  else:
    loca='OffCentre'
  # Initialise variable for counting zeros
  Nsur=int(Room.Nsur)
  Nx=int(Room.maxxleng()//h+1)
  Ny=int(Room.maxyleng()//h+1)
  Nz=int(Room.maxzleng()//h+1)
  Ns=max(Nx,Ny,Nz)
  print('---------------------------------')
  print('Starting the optimal gains calculation')
  foldtype=Refstr+boxstr
  plottype=LOSstr+boxstr+loca
  #-------Run the power calculations for each ray number----------------
  #for j in range(0,nra):
      ##----Retrieve the Mesh--------------------------------------
  #  Nr=int(Nra[j])
  t0=t.time()
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  OptiStr=powerfolder+'/'+boxstr+Obstr+'OptimalGains%03dRefs%03dm%03d_tx%03d'%(Nr,Nre,index,job)
  # if os.path.isfile(OptiStr+'.npy'):
    # Gt=np.load(OptiStr+'.npy')
    # print('Optimal loaded')
    # timemat[0]=t.time()-t0
    # return Gt, timemat
  # Make the refindex, impedance and gains vectors the right length to
  # match the matrices.
  Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1. Repeat Znobrat to match Mesh dimensions
  Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
  refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nsur*Nre+1. Repeat refindex to match Mesh dimensions
  refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
  # Calculate the necessry parameters for the power calculation.
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  Gt=DSM.optimum_gains(foldtype,plottype,Mesh,Room,Znobrat,refindex,Antpar,Pol,Nr,Nre,job,index)
  if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
  if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
  np.save(OptiStr+'.npy',Gt)
  text_file = open(OptiStr+'.txt', 'w')
  n = text_file.write('Optimal antenna gains are')
  n = text_file.write(str(Gt))
  text_file.close()
  t1=t.time()
  timemat[0]=t1-t0
  print('-------------------------------')
  print('Job number',job)
  print('Optimum gains from DSM complete')
  print('Time taken',timemat)
  print('-------------------------------')
  return Gt,timemat


def Residual(Room,Nr=22,index=0,job=0,Nre=3,PerfRef=0,LOS=0,InnerOb=0,Nrs=2,Ns=5):
  ''' Compute the residual between the computed mesh and the true mesh summed over x,y,z and averaged
  '''
  Tx         =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)# The location of the source antenna (origin of every ray)
  _,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  refindex=np.load('Parameters/refindex%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  refindex     =np.load('Parameters/refindex%03d.npy'%index)
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  myfile     = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt    = myfile.read()         # read the entire file into a string
  myfile.close()
  h=1.0/Ns
  Obstr=''
  Nsur=Room.Nsur
  if 0<Nrs<Nsur:
    obnumbers=np.zeros((Nrs,1))
    k=0
    for ob, refin in enumerate(refindex):
      if abs(refin)>epsilon:
        obnumbers[k]=ob
        k+=1
        ObStr=Obstr+'Ob%02d'%ob
  # if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      # Nra=np.array([Nra])
      # nra=1
  # else:
      # nra=len(Nra)
  if Nre>1:
    Refstr=nw.num2words(Nre)+'Ref'
  else:
    Refstr='NoRef'
  if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
    loca='Centre'
  else:
    loca='OffCentre'
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'PerfRef'
      else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'Ref'
      else:
        LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  foldtype=Refstr+boxstr
  plottype=LOSstr+boxstr+loca
  Nsur=Room.Nsur
  Nx=int(Room.maxxleng()//h+1)
  Ny=int(Room.maxyleng()//h+1)
  Nz=int(Room.maxzleng()//h+1)
  Ns=max(Nx,Ny,Nz)
  err=np.zeros(1)
  #for j in range(0,nra):
  #  Nr=Nra[j]
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  pstr       =powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
  truestr    ='./Mesh/True/'+plottype+'/'+boxstr+Obstr+'True_tx%03d.npy'%job
  if os.path.isfile(truestr) and os.path.isfile(pstr):
      Pt   =np.load(truestr)
      Pthat=DSM.db_to_Watts(Pt)
      P    =np.load(pstr)
      Phat =DSM.db_to_Watts(P)
      rat  =abs(Phat-Pthat)
      err[0]+=DSM.Watts_to_db(np.sum(rat)/(P.shape[0]*P.shape[1]*P.shape[2]))
      pratstr='./Mesh/'+plottype+'/'+boxstr+Obstr+'PowerRes_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
      np.save(pratstr,rat)
      if not os.path.exists('./Errors'):
        mkdir('./Errors/')
        mkdir('./Errors/'+plottype)
      if not os.path.exists('./Errors/'+plottype):
        mkdir('./Errors/'+plottype)
      ResStr  ='./Errors/'+plottype+'/'+boxstr+Obstr+'Residual%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
      np.save(ResStr,err[0])
  return err

def Quality(Room,Nr=22,index=0,job=0,Nre=3,PerfRef=0,LOS=0,InnerOb=0,Nrs=2,Ns=5):
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
  _,_,L,split =np.load('Parameters/Raytracing.npy')
  Nra           =np.load('Parameters/Nra.npy')
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)# The location of the source antenna (origin of every ray)
  refindex=np.load('Parameters/refindex%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  Nsur=Room.Nsur
  h=1.0/Ns
  obnumbers=np.zeros((Nrs,1))
  k=0
  Obstr=''
  if 0<Nrs<Nsur:
    obnumbers=np.zeros(Nrs)
    for ob, refind in enumerate(refindex):
      if abs(refind)>epsilon:
        obnumbers[k]=ob
        k+=1
        Obstr=Obstr+'Ob%02d'%ob
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'PerfRef'
      else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'Ref'
      else:
        LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  Nsur=Room.Nsur
  # if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
    # nra=np.array([Nra])
  # else:
    # nra=len(Nra)
  timemat=np.zeros(1)
  Nre=int(Nre)
  if Nre>1:
    Refstr=nw.num2words(Nre)+''
  else:
    Refstr='NoRef'
  if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
    loca='Centre'
  else:
    loca='OffCentre'
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  nra=1
  Qmat=np.zeros((1,3))
  timemat=np.zeros(1)
  Nx=int(Room.maxxleng()//h+1)
  Ny=int(Room.maxyleng()//h+1)
  Nz=int(Room.maxzleng()//h+1)
  Ns=max(Nx,Ny,Nz)
  foldtype=Refstr+boxstr
  plottype=LOSstr+boxstr+loca
  #for j in range(0,nra):
    ##----Retrieve the Mesh--------------------------------------
   # Nr=int(Nra[j])
  t0=t.time()
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  pstr=powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
  if not Room.CheckTxInner(Tx):
    if logon:
        logging.info('Tx=(%f,%f,%f) is not a valid transmitter location'%(Tx[0],Tx[1],Tx[2]))
    print('This is not a valid transmitter location')
    return 0.0
  P=np.load(pstr)
  Q=DSM.QualityFromPower(P)
  Qmat[0,0]=Q
  if not os.path.exists('./Quality'):
      os.makedirs('./Quality')
  if not os.path.exists('./Quality/'+plottype):
      os.makedirs('./Quality/'+plottype)
  np.save('./Quality/'+plottype+'/'+boxstr+Obstr+'Quality%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),Q)
  QM=DSM.QualityMinFromPower(P)
  Qmat[0,1]=Q
  if not os.path.exists('./Quality'):
      os.makedirs('./Quality')
  if not os.path.exists('./Quality/'+plottype):
      os.makedirs('./Quality/'+plottype)
  np.save('./Quality/'+plottype+'/'+boxstr+Obstr+'QualityMin%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),QM)
  QP=DSM.QualityPercentileFromPower(P)
  Qmat[0,2]=QP
  if not os.path.exists('./Quality'):
      os.makedirs('./Quality')
  if not os.path.exists('./Quality/'+plottype):
      os.makedirs('./Quality/'+plottype)
  np.save('./Quality/'+plottype+'/'+boxstr+Obstr+'QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),QP)
  t1=t.time()
  timemat[0]=t1-t0
  print('-------------------------------')
  print('Job number',job)
  print('Quality from DSM complete', Qmat)
  print('Time taken',timemat)
  print('-------------------------------')
  #truestr='Mesh/True/'+plottype+'/True.npy'
  #P3=np.load(truestr)
  #Q2=DSM.QualityFromPower(P3)
  return Qmat #, Q2

def MoveTx(job,Nx,Ny,Nz,h):
  Tx=np.array([(job//Ny)*h+h/2,(job%Ny)*h+h/2,(job//(Nx*Ny))*h+h/2])
  if Tx[0]>Nx*h: Tx[0]=((job%(Nx*Ny))//Ny)*h+h/2
  if Tx[1]>Ny*h: Tx[1]=h/2
  if Tx[2]>Nz*h: Tx[2]=h/2
  np.save('Parameters/Origin_job%03d.npy'%job,Tx)
  return Tx

def jobfromTx(Tx,h):
  Ns=int(1.0//h+1)
  H=round((Tx[2]-0.5*h)/h)
  t=round((Tx[0]-0.5*h)/h)
  u=round((Tx[1]-0.5*h)/h)
  return int(H*(Ns**2)+t*Ns+u)

def main(argv,scriptcall=False):
  job=0 # default job
  if len(argv)>1:
    job=int(argv[1])
    scriptcall=True
  fn='ray_trace_output_%03d.txt'%job
  if scriptcall:
    print('main called with job=%3d, using output filename=%s'%(job,fn,))
  else:
    Tx=np.load('Parameters/Origin.npy')
    h=np.load('Parameters/Raytracing.npy')[1]
    job=jobfromTx(Tx,h)
  np.set_printoptions(precision=3)
  print('Running  on python version')
  print(sys.version)
  timetest=1
  testnum=1
  roomnumstat=1
  Timemat=np.zeros((testnum,6))
  Nra =np.load('Parameters/Nra.npy')
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  Sheetname='InputSheet.xlsx'
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Qmat   =np.zeros((testnum,nra))
  Qtruemat=np.zeros((testnum,nra))
  G_zeros =np.zeros((testnum,nra)) # Number of nonzero terms
  Reserr  =np.zeros((testnum,nra))
  repeat=0
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
      # plot_grid()        # Plot the power in slices.
      end=t.time()
      Reserr[count,:]+=Residual(plottype,Roomnum)/Roomnum
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
  ResOn      =np.load('Parameters/ResOn.npy')
  if not os.path.exists('./Quality'):
    os.makedirs('./Quality')
  if not os.path.exists('./Errors'):
    os.makedirs('./Errors')
  for j in range(testnum):
    qualityname=('./Quality/'+plottype+'/QualityNrays'+str(int(nra))+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
    np.save(qualityname,Qmat[j,:])
+plottype+'/Quality'+str(int(Nra[0]))+'to'+str(int(Nra[-1]))+'Nref'+str(int(Nre))+'.jpg')#.eps').
    errorname=('./Errors/'+plottype+'/ErrorsNrays'+str(int(nra))+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
    np.save(errorname,Reserr[j,:])
  np.save(timename,Timemat)
  np.save('roomnumstat.npy',roomnumstat)
  np.save('Roomnum.npy',Roomnum)
  parameters=  np.load('Parameters/Parameterarray.npy')
  _,_,L,split    =np.load('Parameters/Raytracing.npy')
  for arr in parameters:
    InnerOb,Nr,Nrs,LOS,Nre,PerfRef,Ns,Q,Par,index=arr.astype(int)
    #Tx=np.load('Parameters/Origin.npy')
    ##----Retrieve the environment--------------------------------------
    h=1.0/Ns
    ##----The lengths are non-dimensionalised---------------------------
    Oblist     =np.load('Parameters/Obstacles%d.npy'%index).astype(float)      # The obstacles which are within the outerboundary
    Ntri       =np.load('Parameters/NtriOut.npy')                      # Number of triangles forming the surfaces of the outerboundary
    Room       =rom.room(Oblist,Ntri)
    # -------------Find the number of cells in the x, y and z axis.-------
    Nx=int(Room.maxxleng()//h+1)
    Ny=int(Room.maxyleng()//h+1)
    Nz=int(Room.maxzleng()//h+1)
    # Call another function which moves the transmitter using job.
    if scriptcall:
      Tx=MoveTx(job,Nx,Ny,Nz,h)
    else:
      Tx=np.load('Parameters/Origin.npy')
      job=jobfromTx(Tx,h)
      job=652
      Tx=MoveTx(job,Nx,Ny,Nz,h)
      np.save('Parameters/Origin_job%03d.npy'%job,Tx)
    #InBook     =rd.open_workbook(filename=Sheetname)#,data_only=True)
    #SimParstr  ='SimulationParameters'
    #SimPar     =InBook.sheet_by_name(SimParstr)
    #InBook.save(filename=Sheetname)
    if Ns==11:
      if job==665 or job==652:
        pass
      else:
        continue
    if job>125 and Ns==5:
        continue
    if Nr==337 or Nre==6:
      if job==55:
        continue
      elif job==652 or job==665:
        pass
      else:
        continue
    Mesh1,timemesh,Room=MeshProgram(Nr,index,job,Nre,PerfRef,LOS,InnerOb,Nrs,Ns) # Shoot the rays and store the information
    Grid,timep     =power_grid(Room,Mesh1,Nr,index,job,Nre,PerfRef,LOS,InnerOb,Nrs,Ns)  # Use the ray information to compute the power
    if job==55:
      Gtout,timeo      =optimum_gains(Room,Mesh1,Nr,index,job,Nre,PerfRef,LOS,InnerOb,Nrs,Ns)
    Q=Quality(Room,Nr,index,job,Nre,PerfRef,LOS,InnerOb,Nrs,Ns)
    if ResOn:
      Residual(Room,Nr,index,job,Nre,PerfRef,LOS,InnerOb,Nrs,Ns)
    print('-------------------------------')
    print('Time to complete program') # Roomnum, ray time, average power time,total power time,
    #total time, total time averaged by room, std total time, std total time averaged per room.
    print(timemesh,timep)
    print('-------------------------------')
    print('-------------------------------')
    print('-------------------------------')
  np.save('Parameters/Numjobs.npy',job)
  return 0

if __name__=='__main__':
  main(sys.argv)
  exit()

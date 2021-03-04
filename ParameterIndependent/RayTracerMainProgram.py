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
    directionname='Parameters/Directions%d.npy'%Nra[j]
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
    filename='./Mesh/'+plottype+'/RayPoints%03dRefs%03dm_tx%03d.npy'%(Nr,Nre,job)
    np.save(filename,Rays)
    # The "Rays" file is and array of shape (Nra+1 ,Nre+1 , 4)  containing the
    # co-ordinate and obstacle number for each reflection point corresponding
    # to each source ray.

  return 0


def MeshProgram(SN,repeat=0,plottype=str(),job=0):
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
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  splitinv=1.0/split
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)
  timesmat=np.zeros(nra)

  ##----Retrieve the environment--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)# The location of the source antenna (origin of every ray)
  #OuterBoundary =np.load('Parameters/OuterBoundary.npy').astype(float)  # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')             # Array of
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  InnerOb       =np.load('Parameters/InnerOb.npy')              # Whether the innerobjects should be included or not.
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.
  AngChan       =np.load('Parameters/AngChan.npy')              # switch for whether angles should be corrected for the received points on cones and voxel centre.
  LOS           =np.load('Parameters/LOS.npy')
  Nrs           =np.load('Parameters/Nrs.npy')
  Nsur          =np.load('Parameters/Nsur.npy')

  if LOS:
    LOSstr='LOS'
  else:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+''
      else:
        LOSstr='Multi'
    else:
      LOSstr='Single'
  if InnerOb:
    Box='Box'
  else:
    Box='NoBox'
  foldtype=LOSstr+Box
  if InnerOb:
    Ntri=np.append(Ntri,NtriOb)

    # Oblist=OuterBoundary
    # Ntri=NtriOut
  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Room.__set_MaxInter__(MaxInter)
  Nsur=Room.Nsur
  print('Number of obstacles',Nob)
  # This number will be used in further functions so resaving will ensure the correct number is used.
  np.save('Parameters/Nob.npy',Nob)
  np.save('Parameters/Nsur.npy',Nsur)
  # -------------Find the number of cells in the x, y and z axis.-------
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  if InnerOb:
    DumbyMesh=DSM.DS(Nx,Ny,Nz)
    rom.FindInnerPoints(Room,DumbyMesh)
  #--------------Run the ray tracer for each ray number-----------------
  Ns=max(Nx,Ny,Nz)
  for j in range(0,nra):
    j=int(j)
    meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nra[j],Nre,Ns)
    meshname=meshfolder+'/DSM_tx%03d'%(job)
    if os.path.isfile(meshname+'%02dx%02dy%02dz.npz'%(0,0,0)):
      if j==nra-1:
        Mesh=DSM.load_dict(meshname,Nx,Ny,Nz)
        print('Tx',Tx)
        print('Mesh loaded from store')
      continue
    #------------Initialise the Mesh------------------------------------
    Mesh=DSM.DS(Nx,Ny,Nz,Nsur*Nre+1,Nra[j]*(Nre+1),np.complex128,split)
    #print(Room.obst)
    #print('Inside P',Room.inside_points)
    if not Room.CheckTxInner(Tx):
      if logon:
        logging.info('Tx=(%f,%f,%f) is not a valid transmitter location'%(Tx[0],Tx[1],Tx[2]))
      print('This is not a valid transmitter location')
      return Mesh,timesmat,Room
    print('-------------------------------')
    print('Starting the ray bouncing and information storage')
    print('-------------------------------')
    t0=t.time()
    #-----------The input directions changes for each ray number.-------
    directionname='Parameters/Directions%03d.npy'%Nra[j]
    Direc=np.load(directionname)
    '''Find the ray points and the mesh storing their information
    The rays are reflected Nre times in directions Direc from Tx then
    the information about their paths is stored in Mesh.'''
    if logon:
      logging.info('Tx at(%f,%f,%f)'%(Tx[0],Tx[1],Tx[2]))
    print('Tx',Tx)
    programterms=np.array([Nra[j],Nre,AngChan,split,splitinv,deltheta[j]])
    Rays, Mesh=Room.ray_mesh_bounce(Tx,Direc,Mesh,programterms)
    if dbg:
      assert Mesh.check_nonzero_col(Nre,Nsur)
      print('before del',Mesh[xcheck,ycheck,zcheck])
      for l,m in zip(Mesh[xcheck,ycheck,zcheck].nonzero()[0],Mesh[xcheck,ycheck,zcheck].nonzero()[1]):
        print(l,m,abs(Mesh[xcheck,ycheck,zcheck][l,m]))
      Mesh,ind=Mesh.__del_doubles__(h,Nsur,Ntri=Room.Ntri)
      print('after del',Mesh[xcheck,ycheck,zcheck])
    t1=t.time()
    timesmat[j]=t1-t0
    #----------Save the Mesh for further calculations
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+plottype)
      os.makedirs('./Mesh/'+foldtype)
      os.makedirs(meshfolder)
    if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
      os.makedirs(meshfolder)
    if not os.path.exists(meshfolder):
      os.makedirs(meshfolder)
    if not os.path.exists('./Mesh/'+foldtype):
      os.makedirs('./Mesh/'+foldtype)
    Nr=int(Nra[j])
    rayname='./Mesh/'+foldtype+'/RayMeshPoints%03dRefs%03d_tx%03d.npy'%(Nr,Nre,job)
    np.save(rayname,Rays)
    Mesh.save_dict(meshname)
    #meshnamecsv='./Mesh/'+plottype+'RayMeshPoints%03dRefs%03d_tx%03d.csv'%(Nr,Nre,job)
    #myFile = open(meshnamecsv, 'w')
    #with myFile:
    #  writer = csv.writer(myFile)
    #  writer.writerows(Rays)
    print('-------------------------------')
    print('Ray-launching complete')
    print('Time taken',t1-t0)
    print('-------------------------------')
  return Mesh,timesmat,Room

def StdProgram(SN,Roomnum=1,repeat=0,plottype=str(),job=0):
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
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  splitinv=1.0/split
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)
  timesmat=np.zeros(nra)

  ##----Retrieve the environment--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Tx            =np.load('Parameters/Origin_job%03d.npy'%job).astype(float)         # The location of the source antenna (origin of every ray)
  #OuterBoundary =np.load('Parameters/OuterBoundary.npy').astype(float)  # The Obstacles forming the outer boundary of the room
  deltheta      =np.load('Parameters/delangle.npy')             # Array of
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  InnerOb       =np.load('Parameters/InnerOb.npy')              # Whether the innerobjects should be included or not.
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.

  if InnerOb:
    Ntri=np.append(Ntri,NtriOb)

  # Room contains all the obstacles and walls as well as the number of max intersections any ray segment can have,
  #the number of obstacles (each traingle is separate in this list), number of triangles making up each surface,
  #the number of surfaces.
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Room.__set_MaxInter__(MaxInter)
  Nsur=Room.Nsur
  print('Number of obstacles',Nob)
  # This number will be used in further functions so resaving will ensure the correct number is used.
  np.save('Parameters/Nob.npy',Nob)
  np.save('Parameters/Nsur.npy',Nsur)

  ##----Retrieve the antenna parameters--------------------------------------
  PerfRef    =np.load('Parameters/PerfRef.npy')
  # Indicates whether reflection coefficents are calculated (0) or whether a perfect reflection should be used (1).
  LOS        =np.load('Parameters/LOS.npy')

  Nx=int(Room.maxxleng()/(h)+1)
  Ny=int(Room.maxyleng()/(h)+1)
  Nz=int(Room.maxzleng()/(h)+1)
  Mesh=np.zeros((Nx,Ny,Nz,2),dtype=np.complex128)
  # There are two terms in each grid point to account for the polarisation directions.
  print('-------------------------------')
  print('Mesh for Std program built')
  print('-------------------------------')
  print('Starting the ray bouncing and field storage')
  print('-------------------------------')
  for j in range(0,nra):
    for index in range(Roomnum):
      # Indicates whether all terms after reflection should be ignored and line of sight modelled (1). (0) if reflection should be included.
      freq          = np.load('Parameters/frequency%03d.npy'%index)
      # The frequency in GHz associated with the antenna.
      Freespace     = np.load('Parameters/Freespace%03d.npy'%index)
      # The permmittivty and permeability of air
      Pol           = np.load('Parameters/Pol%03d.npy'%index)
      # Polarisation of the antenna.
      c             =Freespace[3]
      # Speed of light in freespace
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
      t0=t.time()
      directionname=str('Parameters/Directions%03d.npy'%(Nra[j]))
      Direc=np.load(directionname)
      Nr=int(Nra[j])
      gainname      ='Parameters/Tx%dGains%d.npy'%(Nr,index)
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
      timemat[j]=t1-t0
      print('-------------------------------')
      print('Ray-launching std complete')
      print('Time taken',t1-t0)
      print('-------------------------------')
  return Grid,timemat

def power_grid(SN,room,Mesh,repeat=0,plottype=str(),Roomnum=0,job=0):
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
  InnerOb    =np.load('Parameters/InnerOb.npy')
  Nob        =np.load('Parameters/Nob.npy')
  Nsur       =np.load('Parameters/Nsur.npy')
  Nrs        =np.load('Parameters/Nrs.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  timemat=np.zeros(nra)
  Nre=int(Nre)
  if InnerOb:
      Box='Box'
  else:
      Box='NoBox'
  Nx=int(room.maxxleng()/h)
  Ny=int(room.maxyleng()/h)
  Nz=int(room.maxzleng()/h)
  Ns=max(Nx,Ny,Nz)

  # Initialise variable for counting zeros
  G_z=np.zeros((1,nra))

  print('---------------------------------')
  print('Starting the power calculation')
  #-------Run the power calculations for each ray number----------------
  for j in range(0,nra):
      ##----Retrieve the Mesh--------------------------------------
    Nr=int(Nra[j])
    meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    meshname=meshfolder+'/DSM_tx%03d'%(job)
    if os.path.isfile(meshname):
      Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)

    ##----Initialise Grid For Power-------------------------------------
    Ns=max(Nx,Ny,Nz)
    Grid=np.zeros((Nx,Ny,Nz),dtype=float)
    t0=t.time()
    #-------If multiple room variations are desired then run for all----
    for index in range(0,Roomnum):
      ##----Retrieve the antenna parameters--------------------------------------
      gainname      ='Parameters/Tx%dGains%d.npy'%(Nr,index)
      Gt            = np.load(gainname)
      freq          = np.load('Parameters/frequency%03d.npy'%index)
      Freespace     = np.load('Parameters/Freespace%03d.npy'%index)
      Pol           = np.load('Parameters/Pol%03d.npy'%index)

      ##----Retrieve the Obstacle Parameters--------------------------------------
      Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
      refindex     =np.load('Parameters/refindex%03d.npy'%index)
      Obstr=''
      if Nrs<Nsur:
        obnumbers=np.zeros((Nrs,1))
        k=0
        for ob, refin in enumerate(refindex):
          if abs(refin)>epsilon:
            obnumbers[k]=ob
            k+=1
            Obstr=Obstr+'Ob%02d'%ob
      # Make the refindex, impedance and gains vectors the right length to
      # match the matrices.
      Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1. Repeat Znobrat to match Mesh dimensions
      Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
      refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nsur*Nre+1. Repeat refindex to match Mesh dimensions
      refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
      # Calculate the necessry parameters for the power calculation.
      c             =Freespace[3]            # Speed of Light
      Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
      khat          =Antpar[0]
      lam           =Antpar[1]
      L             =Antpar[2]
      if index==0:
        Grid,ind=DSM.power_compute(plottype,Mesh,room,Znobrat,refindex,Antpar,Gt,Pol,Nra[j],Nre,Ns,LOS,PerfRef)
      else:
        Grid,ind=DSM.power_compute(plottype,Mesh,room,Znobrat,refindex,Antpar,Gt,Pol,Nra[j],Nre,Ns,LOS,PerfRef,ind)
      meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nra[j],Nre,Ns)
      if not os.path.exists('./Mesh'):
        os.makedirs('./Mesh')
        os.makedirs('./Mesh/'+plottype)
        os.makedirs(meshfolder)
      if not os.path.exists('./Mesh/'+plottype):
        os.makedirs('./Mesh/'+plottype)
        os.makedirs(meshfolder)
      if not os.path.exists(meshfolder):
        os.makedirs(meshfolder)
      meshname=meshfolder+'/DSM_tx%03d'%(job)
      pstr=meshfolder+'/'+Box+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nra[j],Nre,index,job)
      np.save(pstr,Grid)
      print('Power grid save at, ',pstr)
      RadAstr=meshfolder+'/RadA_grid%dRefs%dm%d.npy'%(Nra[j],Nre,0)
      if os.path.isfile(RadAstr):
        os.rename(r''+meshfolder+'/RadA_grid%dRefs%dm%d.npy'%(Nra[j],Nre,0),r''+meshfolder+'/'+Box+'RadA_grid%dRefs%dm%d_tx%03d.npy'%(Nra[j],Nre,0,job))
      if not LOS:
        Angstr=meshfolder+'/AngNpy.npy'
        if os.path.isfile(Angstr):
          os.rename(r''+meshfolder+'/AngNpy.npy',r''+meshfolder+'/'+Box+'AngNpy%03dRefs%03dNs%03d_tx%03d.npy'%(Nra[j],Nre,Ns,job))
        for su in range(0,Nsur):
          RadSstr=meshfolder+'/RadS%d_grid%dRefs%dm%d.npy'%(su,Nra[j],Nre,0)
          if os.path.isfile(RadSstr):
            os.rename(r''+meshfolder+'/RadS%d_grid%dRefs%dm%d.npy'%(su,Nra[j],Nre,0),r''+meshfolder+'/'+Box+'RadS%d_grid%dRefs%dm%d_tx%03d.npy'%(su,Nra[j],Nre,0,job))
       rstr='rad%dRefs%dNs%d'%(Nra[j],Nre,Ns)
       rfile=meshfolder+rstr
       angstr='/ang%03dRefs%03dNs%0d'%(Nra[j],Nre,Ns)
       angfile=meshfolder+angstr
       for (x,y,z) in product(range(Nx),range(Ny),range(Nz)):
          rfilename=rfile+'%02dx%02dy%02dz.npz'(x,y,z)
          angfilename=angfile+'%02dx%02dy%02dz.npz'(x,y,z)
          os.rename(r''+rfilename,r''+meshfolder+'/'+Box+rstr+'%02dx%02dy%02dz_tx%03d.npz'%(x,y,z,job))
          os.rename(r''+angfilename,r''+meshfolder+'/'+Box+angstr+'%02dx%02dy%02dz_tx%03d.npz'%(x,y,z,job))


      G_z[0,j]=np.count_nonzero((Grid==0))
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Job number',job)
  print('Power from DSM complete')
  print('Time taken',timemat)
  print('-------------------------------')
  return Grid,G_z,timemat

def optimum_gains(SN,room,Mesh,repeat=0,plottype=str(),Roomnum=0,job=0):
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
  InnerOb    =np.load('Parameters/InnerOb.npy')
  Nob        =np.load('Parameters/Nob.npy')
  Nsur       =np.load('Parameters/Nsur.npy')
  Nrs        =np.load('Parameters/Nrs.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  nra=1
  timemat=np.zeros(nra)
  Nre=int(Nre)
  if InnerOb:
      Box='Box'
  else:
      Box='NoBox'
  Nx=int(room.maxxleng()/h)
  Ny=int(room.maxyleng()/h)
  Nz=int(room.maxzleng()/h)
  Ns=max(Nx,Ny,Nz)

  # Initialise variable for counting zeros
  G_z=np.zeros((1,nra))

  print('---------------------------------')
  print('Starting the optimal gains calculation')
  #-------Run the power calculations for each ray number----------------
  for j in range(0,nra):
      ##----Retrieve the Mesh--------------------------------------
    Nr=int(Nra[j])
    meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    meshname=meshfolder+'/DSM_tx%03d'%(job)
    if os.path.isfile(meshname):
      Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
    ##----Initialise Grid For Power-------------------------------------
    Ns=max(Nx,Ny,Nz)
    Grid=np.zeros((Nx,Ny,Nz),dtype=float)
    t0=t.time()
    #-------If multiple room variations are desired then run for all----
    for index in range(0,Roomnum):
      ##----Retrieve the antenna parameters--------------------------------------
      gainname      ='Parameters/Tx%dGains%d.npy'%(Nr,index)
      Gt            = np.load(gainname)
      freq          = np.load('Parameters/frequency%03d.npy'%index)
      Freespace     = np.load('Parameters/Freespace%03d.npy'%index)
      Pol           = np.load('Parameters/Pol%03d.npy'%index)

      ##----Retrieve the Obstacle Parameters--------------------------------------
      Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
      refindex     =np.load('Parameters/refindex%03d.npy'%index)
      Obstr=''
      if Nrs<Nsur:
        obnumbers=np.zeros((Nrs,1))
        k=0
        for ob, refin in enumerate(refindex):
          if abs(refin)>epsilon:
            obnumbers[k]=ob
            k+=1
            Obstr=Obstr+'Ob%02d'%ob
      # Make the refindex, impedance and gains vectors the right length to
      # match the matrices.
      Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1. Repeat Znobrat to match Mesh dimensions
      Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
      refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nsur*Nre+1. Repeat refindex to match Mesh dimensions
      refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
      # Calculate the necessry parameters for the power calculation.
      c             =Freespace[3]            # Speed of Light
      Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
      khat          =Antpar[0]
      lam           =Antpar[1]
      L             =Antpar[2]
      if index==0:
        Gt=DSM.optimum_gains(plottype,Mesh,room,Znobrat,refindex,Antpar,Pol,Nr,Nre,Ns)
      else:
        Gt=DSM.optimum_gains(plottype,Mesh,room,Znobrat,refindex,Antpar,Pol,Nr,Nre,Ns,LOS,PerfRef,ind)
      if not os.path.exists('./Mesh'):
        os.makedirs('./Mesh')
        os.makedirs('./Mesh/'+plottype)
      if not os.path.exists('./Mesh/'+plottype):
        os.makedirs('./Mesh/'+plottype)
      OptiStr='./Mesh/'+plottype+'/'+Box+Obstr+'OptimalGains%03dRefs%03dm%03d_tx%03d'%(Nr,Nre,index,job)
      np.save(OptiStr+'.npy',Gt)
      text_file = open(OptiStr+'.txt', 'w')
      n = text_file.write('Optimal antenna gains are')
      n = text_file.write(Gt)
      text_file.close()
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Job number',job)
  print('Optimum gains from DSM complete')
  print('Time taken',timemat)
  print('-------------------------------')
  return TransGains,timemat


def Residual(plottype=str(),box=str(),Roomnum=0,job=0):
  ''' Compute the residual between the computed mesh and the true mesh summed over x,y,z and averaged
  '''
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  Nrs        =np.load('Parameters/Nrs.npy')
  Nsur       =np.load('Parameters/Nsur.npy')
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
    refindex=np.load('Parameters/refindex%03d.npy'%index)
    Obstr=''
    if Nrs<Nsur:
      obnumbers=np.zeros((Nrs,1))
      k=0
      for ob, refin in enumerate(refindex):
        if abs(refin)>epsilon:
          obnumbers[k]=ob
          k+=1
          ObStr=Obstr+'Ob%02d'%ob
    for j in range(0,nra):
      Nr=int(Nra[j])
      Nre=int(Nre)
      pstr       ='./Mesh/'+plottype+'/'+box+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
      truestr    ='./Mesh/True/'+plottype+'/'+box+Obstr+'True_tx%03d.npy'%job
      if os.path.isfile(truestr) and os.path.isfile(pstr):
        Pt   =np.load(truestr)
        Pthat=DSM.db_to_Watts(Pt)
        P    =np.load(pstr)
        Phat =DSM.db_to_Watts(P)
        rat  =abs(Phat-Pthat)
        err[j]+=DSM.Watts_to_db(np.sum(rat)/(P.shape[0]*P.shape[1]*P.shape[2]))
        pratstr='./Mesh/'+plottype+'/'+box+Obstr+'PowerRes_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
        np.save(pratstr,rat)
        if not os.path.exists('./Errors'):
          mkdir('./Errors/')
          mkdir('./Errors/'+plottype)
        if not os.path.exists('./Errors/'+plottype):
          mkdir('./Errors/'+plottype)
        ResStr  ='./Errors/'+plottype+'/'+box+Obstr+'Residual%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
        np.save(ResStr,err[j])
  return err

def Quality(SN,Room,repeat=0,plottype=str(),Roomnum=0,job=0):
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
  Nra              =np.load('Parameters/Nra.npy')
  InnerOb          =np.load('Parameters/InnerOb.npy')
  Nrs              =np.load('Parameters/Nrs.npy')
  Nsur             =np.load('Parameters/Nsur.npy')
  if InnerOb:
      Box='Box'
  else:
      Box='NoBox'
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  Qmat=np.zeros(nra)
  timemat=np.zeros(nra)
  Nre=int(Nre)
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)

  for j in range(0,nra):
    ##----Retrieve the Mesh--------------------------------------
    Nr=int(Nra[j])
    meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    meshname=meshfolder+'/DSM_tx%03d'%(job)
    Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)

    ##----Initialise Grid For Power------------------------------------------------------
    Ns=max(Nx,Ny,Nz)
    Grid=np.zeros((Nx,Ny,Nz),dtype=float)
    t0=t.time()
    for index in range(0,Roomnum):
      refindex=np.load('Parameters/refindex%03d.npy%index')
      Obstr=''
      if Nrs<Nsur:
        obnumbers=np.zeros((Nrs,1))
        k=0
        for ob, refin in enumerate(refindex):
          if abs(refin)>epsilon:
            obnumbers[k]=ob
            k+=1
            Obstr=Obstr+'Ob%02d'%ob
      pstr       =meshfolder+'/'+Box+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
      P=np.load(pstr)
      Q=DSM.QualityFromPower(P)
      Qmat[j]=Q
      if not os.path.exists('./Quality'):
        os.makedirs('./Quality')
      if not os.path.exists('./Quality/'+plottype):
        os.makedirs('./Quality/'+plottype)
      np.save('./Quality/'+plottype+'/'+Box+Obstr+'Quality%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),Q)
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Job number',job)
  print('Quality from DSM complete', Qmat)
  print('Time taken',timemat)
  print('-------------------------------')
  #truestr='Mesh/True/'+plottype+'/True.npy'
  #P3=np.load(truestr)
  #Q2=DSM.QualityFromPower(P3)
  return Qmat #, Q2

def MoveTx(job,Nx,Ny,Nz,h,L):
  Tx=np.array([(job//Ny)*h+h/2,(job%Ny)*h+h/2,(job//(Nx*Ny))*h+h/2])
  if Tx[0]>Nx*h: Tx[0]=((job%(Nx*Ny))//Ny)*h+h/2
  if Tx[1]>Ny*h: Tx[1]=h/2
  if Tx[2]>Nz*h: Tx[2]=h/2
  np.save('Parameters/Origin_job%03d.npy'%job,Tx)
  return Tx

def jobfromTx(Tx,h):
  H=(Tx[2]-0.5*h)//h
  t=(Tx[0]-0.5*h)//h
  u=(Tx[1]-0.5*h)//h
  return int(H*100+t*10+u)

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
  Sheetname='InputSheet.xlsx'
  #out=PI.DeclareParameters(Sheetname)
  #out=PI.ObstacleCoefficients(Sheetname)
  #out=RayTracer() # To compute just the rays with no storage uncomment this line.
  timetest   =np.load('Parameters/timetest.npy')
  testnum    =np.load('Parameters/testnum.npy')
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  ResOn      =np.load('Parameters/ResOn.npy')
  InnerOb    =np.load('Parameters/InnerOb.npy')
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  LOS        =np.load('Parameters/LOS.npy')
  PerfRef    =np.load('Parameters/PerfRef.npy')
  Nrs        =np.load('Parameters/Nrs.npy')
  Nsur        =np.load('Parameters/Nsur.npy')
  #Tx=np.load('Parameters/Origin.npy')
  ##----Retrieve the environment--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  Oblist     =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Ntri       =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  Room       =rom.room(Oblist,Ntri)
  # -------------Find the number of cells in the x, y and z axis.-------
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  Nra =np.load('Parameters/Nra.npy')
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
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  # Call another function which moves the transmitter using job.
  if scriptcall:
    Tx=MoveTx(job,Nx,Ny,Nz,h,L)
  else:
    np.save('Parameters/Origin_job%03d.npy'%job,Tx)
  if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
    loca='Centre'
  else:
    loca='OffCentre'
  plottype=LOSstr+boxstr+loca
  InBook     =rd.open_workbook(filename=Sheetname)#,data_only=True)
  SimParstr  ='SimulationParameters'
  SimPar     =InBook.sheet_by_name(SimParstr)
  #InBook.save(filename=Sheetname)
  Qmat   =np.zeros((testnum,nra))
  Qtruemat=np.zeros((testnum,nra))
  G_zeros =np.zeros((testnum,nra)) # Number of nonzero terms
  if ResOn:
    Reserr  =np.zeros((testnum,nra))
  Timemat =np.zeros((testnum,nra,8))
  repeat=1
  if logon:
    logname='RayTracer'+plottype+'%03d.log'%job
    logging.basicConfig(filename=logname,filemode='w',format="[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
      level=logging.INFO,
      datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(sys.version)
  if not os.path.exists('./Quality'):
    os.makedirs('./Quality')
  if not os.path.exists('./Quality/'+plottype):
    os.makedirs('./Quality/'+plottype)
  if not os.path.exists('./Errors'):
    os.makedirs('./Errors')
  if not os.path.exists('./Errors/'+plottype):
    os.makedirs('./Errors/'+plottype)
  if not os.path.exists('./Times'):
    os.makedirs('./Times')
  if not os.path.exists('./Times/'+plottype):
    os.makedirs('./Times/'+plottype)
  for j in range(0,timetest):
    Roomnum=(2*j+1)*roomnumstat
    #Timemat[0,0]=Roomnum
    for count in range(0,testnum):
      start=t.time()
      Mesh1,timemesh,Room=MeshProgram(Sheetname,repeat,plottype,job) # Shoot the rays and store the information
      mid=t.time()
      Grid,G_z,timep=power_grid(Sheetname,Room,Mesh1,repeat,plottype,Roomnum,job)  # Use the ray information to compute the power
      #Gtout,timeo=optimum_gains(Sheetname,Room,Mesh1,repeat,plottype,Roomnum,job)
      # repeat=1
      # #G_zeros[count,:]=G_z
      Q             =Quality(Sheetname,repeat,plottype,Roomnum,job)
      # #Qmat[count,:]=Q
      # #Qtruemat[count,:]=Q2
      # end=t.time()
      if ResOn:
        Reserr[count,:]+=Residual(plottype,box,Roomnum,job)/Roomnum
      #Timemat[count,:,0]+=Roomnum
      #Timemat[count,:,1]+=timemesh
      #Timemat[count,:,2]+=timep/(Roomnum)
      #if count !=0:
      #  Timemat[0,:,2]+=timep/(Roomnum)
      #Timemat[count,:,3]+=timep
      #Timemat[count,:,4]+=timep+timemesh
      #Timemat[count,:,5]+=(timep+timemesh)/(Roomnum)
      start2=t.time()
      #Pstd,timemesh2=StdProgram(Sheetname,Roomnum,repeat,plottype,job) # Shoot the rays and store the information
      repeat=1
      end2=t.time()
      Timemat[count,:,6]+=Roomnum
      #Timemat[count,:,7]+=timemesh2
  Timemat[:,:,2]/=(testnum)
  Timemat[:,:,5]/=(testnum)
  Timemat/=(timetest)
#Reserr/=(timetest)
  print('-------------------------------')
  print('Time to complete program') # Roomnum, ray time, average power time,total power time,
  #total time, total time averaged by room, std total time, std total time averaged per room.
  print(Timemat)
  print('-------------------------------')
  print('-------------------------------')
  #print('Residual to the True calculation')
#print(Reserr)
  print('-------------------------------')
  MaxInter =np.load('Parameters/MaxInter.npy')
  for j in range(testnum):
    timename='./Times/'+plottype+'/'+box+'TimesNra%03dRefs%03dRoomnum%dto%03dMaxInter%d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+j*2,MaxInter,job)
    np.save(timename,Timemat[j,:,4])
    qualityname='./Quality/'+plottype+'/'+box+'QualityNrays%03dRefs%03dRoomnum%03dto%03d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+j*2,job)
    np.save(qualityname,Qmat[j,:])
    if ResOn:
      errorname='./Errors/'+plottype+'/'+box+'ErrorsNrays%03dRefs%03dRoomnum%03dto%03d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+j*2,job)
      np.save(errorname,Reserr[j,:])
  #np.save(timename,Timemat)
  np.save('Parameters/roomnumstat.npy',roomnumstat)
  np.save('Parameters/Roomnum.npy',Roomnum)
  np.save('Parameters/Numjobs.npy',job)
  return 0

if __name__=='__main__':
  main(sys.argv)
  exit()


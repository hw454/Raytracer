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
  coefficients, and the function :py:func:`RefCombine` to \git st
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
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
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
  filename=str('RayPoints'+str(int(Nra))+'Refs'+str(int(Nre))+'n.npy')
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
  print('-------------------------------')
  print('Building Mesh')
  print('-------------------------------')
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
  Tx            =np.load('Parameters/Origin.npy')             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  Nx=int(Room.maxxleng()/h+1)
  Ny=int(Room.maxyleng()/h+1)
  Nz=int(Room.maxzleng()/h+1)
  Mesh=DSM.DS(Nx,Ny,Nz,Nob*Nre+1,Nra*(Nre+1))
  print('-------------------------------')
  print('Mesh built')
  print('-------------------------------')
  print('Starting the ray bouncing and information storage')
  print('-------------------------------')
  Rays, Mesh=Room.ray_mesh_bounce(Tx,Nre,Nra,Direc,Mesh)
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  np.save('./Mesh/RayMeshPoints'+str(Nra)+'Refs'+str(Nre)+'m.npy',Rays)
  meshname=str('./Mesh/DSM'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  Mesh.save_dict(meshname)
  print('-------------------------------')
  print('Ray-launching complete')
  print('Time taken',Room.time)
  print('-------------------------------')
  return Mesh

def power_grid():
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
  Nob            =np.load('Parameters/Nob.npy')

  PI.ObstacleCoefficients()
  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/TxGains.npy')
  freq          = np.load('Parameters/frequency.npy')
  Freespace     = np.load('Parameters/Freespace.npy')
  c             =Freespace[3]
  khat          =freq*L/c
  lam           =(2*np.pi*c)/freq
  Antpar        =np.array([khat,lam,L])

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat.npy')
  refindex     =np.load('Parameters/refindex.npy')

  ##----Retrieve the Mesh--------------------------------------
  meshname=str('./Mesh/DSM'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  Mesh= DSM.load_dict(meshname)

  ##----Initialise Grid------------------------------------------------------
  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  Grid=np.zeros((Nx,Ny,Nz),dtype=float)

  Grid=DSM.power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt)
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  np.save('./Mesh/Power_grid'+str(Nra)+'Refs'+str(Nre)+'m.npy',Grid)

  return Grid


def RefCoefComputation(Mesh):
  ''' Compute the mesh of reflection coefficients.

  :param Mesh: The DS mesh which contains the angles and distances rays \
  have travelled.

  Load the physical parameters using \
  :py:func:`ParameterInput.ObstacleCoefficients`
  * Znobrat - is the vector of characteristic impedances for obstacles \
  divided by the characteristic impedance of air.
  * refindex - if the vector of refractive indexes for the obstacles.

  Compute the Reflection coefficients (RefCoefper,Refcoefpar) using:
  :py:func:`DictionarySparseMatrix.ref_coef`

  :rtype: (:py:class:`DictionarySparseMatrix.DS'(Nx,Ny,Nz,na,nb)\
  ,:py:class:`DictionarySparseMatrix.DS'(Nx,Ny,Nz,na,nb))

  :returns: (RefCoefper,Refcoefpar)

  '''
  out=PI.ObstacleCoefficients()
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

  :rtype: (:py:class:`DictionarySparseMatrix.DS'(Nx,Ny,Nz,1,nb), \
  :py:class:`DictionarySparseMatrix.DS'(Nx,Ny,Nz,na,nb))

  :return: Combper, Combpar

  '''
  Combper=Rper.dict_col_mult()
  Combpar=Rpar.dict_col_mult()
  return Combper, Combpar

def plot_grid():
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  P=np.load('Power_grid.npy')
  n=P.shape[2]
  if not os.path.exists('./GeneralMethodPowerFigures'):
    os.makedirs('./GeneralMethodPowerFigures')
  for i in range(n):
    mp.figure(i)
    #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
    mp.imshow(P[:,:,i], cmap='viridis', interpolation='nearest')#,extent=extent)
    filename=str('GeneralMethodPowerFigures/PowerSlice'+str(int(i))+'Nra'+str(int(Nra))+'n'+str(int(n))+'Nref'+str(int(Nre))+'.eps')
    mp.savefig(filename)
    #mp.colourbar()
  #mp.show()
  return

if __name__=='__main__':
  np.set_printoptions(precision=3)
  print('Running  on python version')
  print(sys.version)
  #out=RayTracer()
  start=t.time()
  Mesh=MeshProgram()
  #RefCoefperp, RefCoefpar=RefCoefComputation(Mesh)
  #Combper,Combpar=RefCombine(RefCoefperp,RefCoefpar)
  Grid=power_grid()
  plot_grid()
  end=t.time()
  print('-------------------------------')
  print('Time to complete program')
  print(end-start)
  print('-------------------------------')
  exit()


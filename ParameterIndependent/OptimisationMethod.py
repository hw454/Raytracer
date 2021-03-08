#!/usr/bin/env python3
# H Wragg 26th January 2021
'''This code will profile the script 'scriptname' for calls and timing.
Initially developed for profiling the raytracer code in :py:mod:'RayTracerMainProgram.py'.
'scriptname'=:py:mod:'RayTracerMainProgram':py:func:'Main', which calls and varies
the inputs from :py:mod:'ParameterLoad' then runs :py:func:'MeshProgram'
'''
import io
import sys
import os
import matplotlib.pyplot as mp
import RayTracerMainProgram as RT
import Rays as Ra
import DictionarySparseMatrix as DSM
from scipy.optimize import minimize
import numpy as np
import Room as rom
import num2words as nw
import RayTracerPlottingFunctions as RTplot
import time as t
epsilon=sys.float_info.epsilon


def Quality_MoreInputs(Tx,Direc,programterms,RayPar,foldtype,Room,Znobrat,refindex,Antpar,Gt, Pol,LOS,PerfRef,Boxstr,Obstr,index=0):
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
  Nr,Nre,AngChan,split,splitinv,delth=programterms
  Nre=int(Nre)
  Nr=int(Nr)
  ##----Retrieve the Raytracing Parameters-----------------------------
  _,h,L,_                     =RayPar
  Nsur                        =Room.Nsur
  if not 0<=Tx[0]<Room.maxxleng() and not 0<=Tx[1]<Room.maxyleng() and not 0<=Tx[2]<Room.maxzleng():
    return -1000
  if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
    loca='Centre'
  else:
    loca='OffCentre'
  foldtype=foldtype+loca
  plottype=foldtype+loca
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  Ns=max(Nx,Ny,Nz)
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  meshname=meshfolder+'/DSM_tx%03dx%03dy%03dz'%(Tx[0],Tx[1],Tx[2])
  if os.path.isfile(meshname):
    Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
  else:
    ##----Retrieve the environment--------------------------------------
    ##----The lengths are non-dimensionalised---------------------------
    # -------------Find the number of cells in the x, y and z axis.-------
    #------------Initialise the Mesh------------------------------------
    Mesh=DSM.DS(Nx,Ny,Nz,Nsur*Nre+1,Nr*(Nre+1),np.complex128,split)
    if not Room.CheckTxInner(Tx):
      return 0
    #-----------The input directions changes for each ray number.-------
    Rays, Mesh=Room.ray_mesh_bounce(Tx,Direc,Mesh,programterms)
    Mesh.save_dict(meshname)
  # Initialise Grid For Power-------------------------------------
  Ns=max(Nx,Ny,Nz)
  Grid=np.zeros((Nx,Ny,Nz),dtype=float)
  #Gt=DSM.optimum_gains(plottype,Mesh,Room,Znobrat,refindex,Antpar, Pol,Nra,Nre,Ns,LOS,PerfRef)
  Q,_=DSM.quality_compute(foldtype,Mesh,Grid,Room,Znobrat,refindex,Antpar,Gt, Pol,Nr,Nre,Ns,LOS,PerfRef)
  if not os.path.exists('./Quality'):
   os.makedirs('./Quality')
  if not os.path.exists('./Quality/'+plottype):
    os.makedirs('./Quality/'+plottype)
  np.save('./Quality/'+plottype+'/'+Boxstr+Obstr+'Quality%03dRefs%03dm%03d_tx%03dx%03dy%03dz.npy'%(Nr,Nre,index,Tx[0],Tx[1],Tx[2]),Q)
  return -Q

def MoreInputs_Run(index=0):
  ''' Load the input variables then run Quality_MoreInputs()'''
  ##----Retrieve the Raytracing Parameters-----------------------------
  RayPar        =np.load('Parameters/Raytracing.npy')
  Nre,h,L,split =RayPar
  InnerOb       =np.load('Parameters/InnerOb%d.npy'%index)              # Whether the innerobjects should be included or not.
  LOS           =np.load('Parameters/LOS%d.npy'%index)
  Nrs           =np.load('Parameters/Nrs.npy')
  Nsur          =np.load('Parameters/Nsur.npy')
  PerfRef       =np.load('Parameters/PerfRef%d.npy'%index)
  LOS           =np.load('Parameters/LOS.npy')
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index).astype(float)      # The obstacles which are within the outerboundary
  splitinv=1.0/split
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)
  if InnerOb:
    Ntri=np.append(Ntri,NtriOb)
    # Room contains all the obstacles and walls.
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Room.__set_MaxInter__(MaxInter)
  Nsur=Room.Nsur
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  Ns=max(Nx,Ny,Nz)
  if InnerOb:
    DumbyMesh=DSM.DS(Nx,Ny,Nz)
    rom.FindInnerPoints(Room,DumbyMesh)
  if Nre>1:
    Refstr=nw.num2words(Nre)+''
  else:
    Refstr='NoRef'
  if InnerOb:
    Box='Box'
  else:
    Box='NoBox'
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
  foldtype=Refstr+Box
  plottype=LOSstr+Box
  ##----Retrieve the environment--------------------------------------
  ##----The lengths are non-dimensionalised---------------------------
  # These paramters are only needed if the Mesh is not already saved.
  deltheta      =np.load('Parameters/delangle.npy')             # Array of
  AngChan       =np.load('Parameters/AngChan.npy')              # switch for whether angles should be corrected for the received points on cones and voxel centre.

  ##----Retrieve the antenna parameters--------------------------------------
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
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  for i in range(nra):
    Nr=Nra[i]
    delth=deltheta[i]
    Tx0=np.array([0.4,0.4,0.4])
    #-----------The input directions changes for each ray number.-------
    directionname='Parameters/Directions%03d.npy'%Nr
    Direc=np.load(directionname)
    gainname      ='Parameters/Tx%dGains%03d.npy'%(Nr,index)
    Gt            = np.load(gainname)
    programterms=np.array([Nr,Nre,AngChan,split,splitinv,delth])
    pars=(Direc,programterms,RayPar,foldtype,Room,Znobrat,refindex,Antpar,Gt, Pol,LOS,PerfRef,Box,Obstr)
    TxB=np.array([(0,h*Nx),(0,h*Ny),(0,h*Nz)])
    Tx=Tx0
    optTol=1e-1
    spacetol=0.075
    t0=t.time()
    TxOut=minimize(Quality_MoreInputs, Tx, method='Powell',args=pars,options={'xtol':spacetol,'ftol':optTol},bounds=TxB)
    t1=t.time()
    print('Optimal Tx at ',TxOut.x)
    spacetol2=0.025
    optTol2=1e-3
    print('Reduce tolerance')
    t2=t.time()
    TxOutTolHigh=minimize(Quality_MoreInputs, Tx, method='Powell',args=pars,options={'xtol':spacetol2, 'ftol':optTol2},bounds=TxB)
    t3=t.time()
    print('Optimal Tx at ',TxOutTolHigh.x)
    ResultsFolder='./OptimisationResults'
    SpecResultsFolder=ResultsFolder+'/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    if not os.path.exists(ResultsFolder):
      os.makedirs(ResultsFolder)
      os.makedirs(ResultsFolder+'/'+plottype)
      os.makedirs(SpecResultsFolder)
    if not os.path.exists(ResultsFolder+'/'+plottype):
      os.makedirs(ResultsFolder+'/'+plottype)
      os.makedirs(SpecResultsFolder)
    if not os.path.exists(SpecResultsFolder):
      os.makedirs(SpecResultsFolder)
    np.save(SpecResultsFolder+'/'+Obstr+'OptimumOrigin.npy',TxOut.x)
    print('Full results',TxOut)
    text_file = open(SpecResultsFolder+'/'+Obstr+'OptimalOrigin.txt', 'w')
    n = text_file.write('\n Optimal location at ')
    n = text_file.write(str(TxOut.x))
    n = text_file.write('\n Number of iterations ')
    n = text_file.write(str(TxOut.nit))
    n = text_file.write('\n Time taken ')
    n = text_file.write(str(t2-t0))
    n = text_file.write('\n Function Tolerance %f'%optTol)
    n = text_file.write('\n Spacial Tolerance %f'%spaceTol)
    n = text_file.write('\n ---------\n ')
    n = text_file.write('\n Optimal location at ')
    n = text_file.write(str(TxOutTolHigh.x))
    n = text_file.write('\n Number of iterations ')
    n = text_file.write(str(TxOutTolHigh.nit))
    n = text_file.write('\n Time taken ')
    n = text_file.write(str(t3-t2))
    n = text_file.write('\n Function Tolerance %f'%optTol2)
    n = text_file.write('\n Spacial Tolerance %f'%spaceTol2)
    text_file.close()
    meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    meshname=meshfolder+'/DSM_tx%03dx%03dy%03dz'%(TxOut.x[0],TxOut.x[1],TxOut.x[2])
    if os.path.isfile(meshname):
      Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
    else:
      print('The optimal Tx does not have a corresponding Mesh')
      print(meshname)
      Quality_MoreInputs(TxOut.x,pars)
    P,_=DSM.power_compute(plottype,Mesh,Room,Znobrat,refindex,Antpar,Gt, Pol,Nr,Nre,Ns,LOS,PerfRef)
    np.save(meshfolder+'/'+Box+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03dx%03dy%03dz.npy'%(Nr,Nre,index,TxOut.x[0],TxOut.x[1],TxOut.x[2]),P)
    RadAstr=meshfolder+'/RadA_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
    if os.path.isfile(RadAstr):
      os.rename(r''+meshfolder+'/RadA_grid%dRefs%dm%d.npy'%(Nr,Nre,index),r''+meshfolder+'/'+Box+'RadA_grid%dRefs%dm%d_tx%03dx%03dy%03dz.npy'%(Nr,Nre,index,TxOut.x[0],TxOut.x[1],TxOut.x[2]))
    if not LOS:
      Angstr='./Mesh/'+plottype+'/AngNpy.npy'
      if os.path.isfile(Angstr):
        os.rename(r''+meshfolder+'/AngNpy.npy',r''+meshfolder+'/'+Box+'AngNpy%03dRefs%03dNs%03d_tx%03dx%03dy%03dz.npy'%(Nr,Nre,Ns,TxOut.x[0],TxOut.x[1],TxOut.x[2]))
      for su in range(0,Nsur):
        RadSstr=meshfolder+'/RadS%d_grid%dRefs%dm%d.npy'%(su,Nr,Nre,index)
        if os.path.isfile(RadSstr):
          os.rename(r''+meshfolder+'/RadS%d_grid%dRefs%dm%d.npy'%(su,Nr,Nre,index),r''+meshfolder+'/'+Box+'RadS%d_grid%dRefs%dm%d_tx%03dx%03dy%03dz.npy'%(su,Nr,Nre,index,TxOut.x[0],TxOut.x[1],TxOut.x[2]))
    myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
    plotfit= myfile.read()         # read the entire file into a string
    myfile.close()
    RTplot.plot_mesh(Mesh,Room,TxOut.x,plottype,boxstr,Obstr,Nr,Nre,Ns,plotfit,LOS,index)
    meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    meshname=meshfolder+'/DSM_tx%03dx%03dy%03dz'%(TxOutTolHigh.x[0],TxOutTolHigh.x[1],TxOutTolHigh.x[2])
    if os.path.isfile(meshname):
      Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
    else:
      print('The optimal Tx does not have a corresponding Mesh')
      print(meshname)
    P,_=DSM.power_compute(plottype,Mesh,Room,Znobrat,refindex,Antpar,Gt, Pol,Nr,Nre,Ns,LOS,PerfRef)
    np.save(meshfolder+'/'+Box+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03dx%03dy%03dz.npy'%(Nr,Nre,index,TxOutTolHigh.x[0],TxOutTolHigh.x[1],TxOutTolHigh.x[2]),P)
    RadAstr=meshfolder+'/RadA_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
    if os.path.isfile(RadAstr):
      os.rename(r''+meshfolder+'/RadA_grid%dRefs%dm%d.npy'%(Nr,Nre,index),r''+meshfolder+'/'+Box+'RadA_grid%dRefs%dm%d_tx%03dx%03dy%03dz.npy'%(Nr,Nre,index,TxOut.x[0],TxOut.x[1],TxOut.x[2]))
    if not LOS:
      Angstr='./Mesh/'+plottype+'/AngNpy.npy'
      if os.path.isfile(Angstr):
        os.rename(r''+meshfolder+'/AngNpy.npy',r''+meshfolder+'/'+Box+'AngNpy%03dRefs%03dNs%03d_tx%03dx%03dy%03dz.npy'%(Nr,Nre,Ns,TxOutTolHigh.x[0],TxOutTolHigh.x[1],TxOutTolHigh.x[2]))
      for su in range(0,Nsur):
        RadSstr=meshfolder+'/RadS%d_grid%dRefs%dm%d.npy'%(su,Nr,Nre,index)
        if os.path.isfile(RadSstr):
          os.rename(r''+meshfolder+'/RadS%d_grid%dRefs%dm%d.npy'%(su,Nr,Nre,index),r''+meshfolder+'/'+Box+'RadS%d_grid%dRefs%dm%d_tx%03dx%03dy%03dz.npy'%(su,Nr,Nre,index,TxOutTolHigh.x[0],TxOutTolHigh.x[1],TxOutTolHigh.x[2]))
    myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
    plotfit= myfile.read()         # read the entire file into a string
    myfile.close()
    RTplot.plot_mesh(Mesh,Room,TxOutTolHigh.x,plottype,boxstr,Obstr,Nr,Nre,Ns,plotfit,LOS,index)
  return 0


def main():
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  _,h,_,_    =np.load('Parameters/Raytracing.npy')
  Room=rom.room(Oblist,Ntri)
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  #Ra.centre_dist_test()
  #RT.main(sys.argv)
  Tx0=np.array([0,0,0])
  Quality_Tx(Tx0)
  TxB=((0,h*Nx),(0,h*Ny),(0,h*Nz))
  TxOut=minimize(Quality_Tx, Tx0, method='SLSQP', tol=1e-6,bounds=TxB)
  print('main',TxOut)

  return 0

if __name__=='__main__':
  #main
  index=1
  MoreInputs_Run(index)
  exit()


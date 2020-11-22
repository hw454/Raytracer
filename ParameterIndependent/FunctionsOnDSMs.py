#!/usr/bin/env python3
# Hayley 6th November 2020
'''----------------------------------------------------------------------
 NOTATION
 ----------------------------------------------------------------------
 dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix
 DS or DSM is a DS object which is a dictionary of sparse matrices
 ----------------------------------------------------------------------

 Code for the dictionary of sparse matrices class :py:class:`DS` which\
 indexes like a multidimensional array but the array is sparse. \
 To exploit :py:mod:`scipy.sparse.dok_matrix`=SM the `DS` uses a key for \
 each x,y, z position and associates a SM.

 This module also contains functions which load a dictionary sparse matrix then calculate different functions.
 '''


import numpy as np
from scipy.sparse import dok_matrix as SM
import scipy.sparse.linalg
from six.moves import cPickle as pkl
import DictionarySparseMatrix as DSM
import sys
import time as t
import os
import time as t
import Room as rom

epsilon=sys.float_info.epsilon
#----------------------------------------------------------------------
# NOTATION IN COMMENTS
#----------------------------------------------------------------------
# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix
# DS or DSM is a DS object which is a dictionary of sparse matrices.
def PowerFromDSM(plottype,Nra,numjobs,testnum,timetest,roomnumstat):
  ##----The lengths are non-dimensionalised---------------------------
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Ntri       =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  InnerOb       =np.load('Parameters/InnerOb.npy')              # Whether the innerobjects should be included or not.
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.

  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Nsur=Room.Nsur
  ##----Retrieve the Raytracing Parameters-----------------------------
  PerfRef    =np.load('Parameters/PerfRef.npy')
  LOS        =np.load('Parameters/LOS.npy')
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  timemat=np.zeros(nra)
  Nre=int(Nre)
  Nob            =np.load('Parameters/Nob.npy')
  Nsur           =np.load('Parameters/Nsur.npy')
  #-------Run the power calculations for each ray number----------------
  repeat=0
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  meshname='./Mesh/'+plottype+'/DSM%03dRefs%03d_tx%03d.npy'%(Nra[0],Nre,0)
  Mesh= DSM.load_dict(meshname)
  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  Ns=min(Nx,Ny,Nz)
  PGrid   =np.zeros((testnum,nra,numjobs,Nx,Ny,Nz))
  jobarray=np.array([15,20,25])
  for i in range(0,timetest):
    Roomnum=(2*i+1)*roomnumstat
    for j in range(0,nra):
      Nr=int(Nra[j])
      for job in jobarray:
        ##----Retrieve the Mesh--------------------------------------
        meshname='./Mesh/'+plottype+'/DSM%03dRefs%03d_tx%03d.npy'%(Nr,Nre,job)
        Mesh= DSM.load_dict(meshname)
        #print(LOS)
        #print(Mesh.nonzero())
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
          # Make the refindex, impedance and gains vectors the right length to
          # match the matrices.
          Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1. Repeat Znobrat to match Mesh dimensions
          Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
          refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nsur*Nre+1. Repeat refindex to match Mesh dimensions
          refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
          Gt=np.tile(Gt,(Nre+1,1))

          # Calculate the necessry parameters for the power calculation.
          c             =Freespace[3]            # Speed of Light
          Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
          khat          =Antpar[0]
          lam           =Antpar[1]
          L             =Antpar[2]
          PGrid[i,j,job,:,:,:],ind=DSM.power_compute(plottype,Mesh,Room,Znobrat,refindex,Antpar,Gt, Pol,Nra,Nre,Ns,LOS=0,PerfRef=0,ind=-1)
          if not os.path.exists('./Mesh'):
            os.makedirs('./Mesh')
            os.makedirs('./Mesh/'+plottype)
          if not os.path.exists('./Mesh/'+plottype):
            os.makedirs('./Mesh/'+plottype)
          np.save('./Mesh/'+plottype+'/Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),PGrid[i,j,job,:,:,:])
  return 0


def QualityPercentileFromPower(SN,repeat=0,plottype=str(),Roomnum=0):
  ''' Calculate the field on a grid using enviroment parameters and the \
  ray Mesh.

  Loads:

  * (*Nra*\ = number of rays, *Nre*\ = number of reflections, \
  *h*\ = meshwidth, *L*\ = room length scale, *split*\ =number of steps through a mesh element)\
  =`Paramters/Raytracing.npy`
  * *P*\ = power grid.
  * *numjobs*\ = number of transmitter locations

  Method:
  * Use the function :py:func:`DictionarySparseMatrix.QualityPercentileFromPower(P)`
  to compute the power.

  :rtype: A numpy array of floats with shape (nra,numjobs)

  :returns: Q

  '''

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra              =np.load('Parameters/Nra.npy')
  numjobs          =np.load('Parameters/Numjobs.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  Qmat=np.zeros((nra,numjobs+1))
  timemat=np.zeros(nra)
  Nre=int(Nre)
  for j in range(0,nra):
    ##----Retrieve the Mesh--------------------------------------
    Nr=int(Nra[j])
    t0=t.time()
    for index in range(0,Roomnum):
     for job in range(0,numjobs+1):
      if job<numjobs+1:#!=15 and job!=20 and job!=25:
        pstr       ='./Mesh/'+plottype+'/Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
        P=np.load(pstr)
        Q=DSM.QualityPercentileFromPower(P)
        Qmat[j,job]=Q
        if not os.path.exists('./Quality'):
          os.makedirs('./Quality')
        if not os.path.exists('./Quality/'+plottype):
          os.makedirs('./Quality/'+plottype)
        np.save('./Quality/'+plottype+'/QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),Q)
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Quality Percentile from Power Grid complete', Qmat)
  print('Time taken',timemat)
  print('-------------------------------')
  return Q

def QualityPercentileAndMinQFromPower(SN,repeat=0,plottype=str(),Roomnum=0):
  ''' Calculate the field on a grid using enviroment parameters and the \
  ray Mesh.

  Loads:

  * (*Nra*\ = number of rays, *Nre*\ = number of reflections, \
  *h*\ = meshwidth, *L*\ = room length scale, *split*\ =number of steps through a mesh element)\
  =`Paramters/Raytracing.npy`
  * *P*\ = power grid.
  * *numjobs*\ = number of transmitter locations

  Method:
  * Use the function :py:func:`DictionarySparseMatrix.QualityPercentileFromPower(P)`
  to compute the power.

  :rtype: A numpy array of floats with shape (nra,numjobs)

  :returns: Q

  '''

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra              =np.load('Parameters/Nra.npy')
  numjobs          =np.load('Parameters/Numjobs.npy')
  InnerOb          =np.load('Parameters/InnerOb.npy')
  Orig             =np.load('Parameters/Origin.npy')
  Oblist           =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Ntri             =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  MaxInter         =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.

  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  if InnerOb:
      Box='Box'
  else:
      Box='NoBox'
  Qmat=np.zeros((nra,numjobs))
  Qmat2=np.zeros((nra,numjobs))
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Room.__set_MaxInter__(MaxInter)
  Nsur=Room.Nsur
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  timemat=np.zeros(nra)
  Nre=int(Nre)
  for j in range(0,nra):
    ##----Retrieve the Mesh--------------------------------------
    Nr=int(Nra[j])
    t0=t.time()
    Mesh=DSM.DS(Nx,Ny,Nz,Nsur*Nre+1,Nr*(Nre+1),np.complex128,split)
    rom.FindInnerPoints(Room,Mesh,Orig)
    for index in range(0,Roomnum):
      for job in range(0,numjobs+1):
       if not os.path.exists('./Quality'):
         os.makedirs('./Quality')
       if not os.path.exists('./Quality/'+plottype):
         os.makedirs('./Quality/'+plottype)
       Tx=np.load('Parameters/Origin_job%03d.npy'%job)
       Txind=Room.position(Tx,h)
       if not Room.CheckTxInner(Tx):
         print(Tx)
         np.save('./Quality/'+plottype+'/'+Box+'QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),0.0)
         np.save('./Quality/'+plottype+'/'+Box+'QualityMin%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),0.0)
         continue
       pstr       ='./Mesh/'+plottype+'/'+Box+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
       P=np.load(pstr)
       Q=DSM.QualityPercentileFromPower(P)
       Q2=DSM.QualityMinFromPower(P)
       Qmat[j,job]=Q
       Qmat2[j,job]=Q2
       np.save('./Quality/'+plottype+'/'+Box+'QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),Q)
       np.save('./Quality/'+plottype+'/'+Box+'QualityMin%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job),Q2)
    t1=t.time()
    timemat[j]=t1-t0
  print('-------------------------------')
  print('Q percentile range',np.amin(Qmat),np.amax(Qmat))
  print('Q min range',np.amin(Qmat2),np.amax(Qmat2))
  #print('Quality Percentile from Power Grid complete', Qmat)
  #print('Quality Percentile from Power Grid complete', Qmat2)
  #print('Time taken',timemat)
  print('-------------------------------')
  return Qmat,Qmat2


def main(argv,verbose=False):
  SN='InputSheet.xlsx'
  timetest   =np.load('Parameters/timetest.npy')
  testnum    =np.load('Parameters/testnum.npy')
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  Nra        =np.load('Parameters/Nra.npy')
  numjobs    =np.load('Parameters/Numjobs.npy')
  repeat=0
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  Qmat   =np.zeros((testnum,nra,numjobs))
  Qmat2  =np.zeros((testnum,nra,numjobs))
  #P=PowerFromDSM(plottype,Nra,numjobs,testnum,timetest,roomnumstat)
  for j in range(0,timetest):
    Roomnum=(2*j+1)*roomnumstat
    Qmat[j,:,:],Qmat2[j,:,:]=QualityPercentileAndMinQFromPower(SN,repeat,plottype,Roomnum)
  return 0

if __name__=='__main__':
  main(sys.argv)
  exit()

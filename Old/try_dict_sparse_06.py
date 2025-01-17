#!/usr/bin/env python3
# Hayley 2019-02-14

import numpy as np
from scipy.sparse import lil_matrix as SM
from itertools import product
import math
import sys
import time

# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix
class DS:
  def __init__(s,nx=1,ny=1,nz=1,na=1,nb=1):
    '''nx,ny,nz are the maximums for the key for the dictionary and na
    and nb are the dimensions of the sparse matrix associated with each key.'''
    s.shape=(na,nb)
    s.d={}
    for x, y, z in product(range(nx),range(ny),range(nz)):
      s.d[x,y,z]=SM(s.shape,dtype=np.complex128)
    s.nx=nx
    s.ny=ny
    s.nz=nz
  def __getitem__(s,i):
    if len(i)==3:
      dk=i[:3]
      return s.d[dk]                # return a SM
    elif len(i)==4:
      dk,smk=i[:3],i[3]
      return s.d[dk][smk,:]         # return a SM row
    elif len(i)==5:
      dk,smk=i[:3],i[3:]
      return s.d[dk][smk[0],smk[1]] # return a SM element if smk=[init,int], column if smk=[:,int], row if smk=[int,:], full SM if smk=[:,:]
    # elif i.shape[1]>1:
      # dk1=i[0][:]
      # dk2=i[1][:]
      # dk3=i[2][:]
      # smk1=i[3][:]
      # smk2=i[4][:]
      # for x, y, z in product(dk1,dk2,dk3):
      # #return 0
    else:
      # ...
      raise Exception('Error getting the (%s) part of the sparse matrix. Invalid index (%s). A 3-tuple is required to return a sparse matrix(SM), 4-tuple for the row of a SM or 5-tuple for the element in the SM.' %(i,x,i))
      pass
  def __setitem__(s,i,x):
    if len(i)==3:
      dk,smk=i[:3],i[3:]            # set a SM to an input SM
      if dk not in s.d:
        s.d[dk]=SM(0.0,shape=s.shape,dtype=np.complex128)
      s.d[dk]=x
    elif len(i)==4:                 # set a SM row
      dk,smk=i[:3],i[3:]
      if dk not in s.d:
        s.d[dk]=SM(0.0,shape=s.shape,dtype=np.complex128)
      s.d[dk][smk[0],:]=x
    elif len(i)==5:                # set a SM element
      dk,smk=i[:3],i[3:]
      if dk not in s.d:
        s.d[dk]=SM(0.0,shape=s.shape,dtype=np.complex128)
      s.d[dk][smk[0],smk[1]]=x
    else:
      # ...
      raise Exception('Error setting the (%s) part of the sparse matrix to (%s). Invalid index (%s). A 3-tuple is required to return a sparse matrix(SM), 4-tuple for the row of a SM or 5-tuple for the element in the SM.' %(i,x,i))
      pass
  def __str__(s):
    return str(s.d)
  def __repr__(s):
    return str(s.d) # TODO
  def nonzero(s):
    for x,y,z in product(range(s.nx),range(s.ny),range(s.nz)):
      indicesM=s.d[x,y,z].nonzero()
      NI=len(indicesM[0])
      for j in range(0,NI):
        if x==0 and y==0 and z==0 and j==0:
          indices=np.array([[0,0,0,indicesM[0][j],indicesM[1][j]]])
        else:
          indices=np.append(indices,[[x,y,z,indicesM[0][j],indicesM[1][j]]],axis=0)
    return indices
  def dense(s):
    (na,nb)=s.d[0,0,0].shape
    nx=s.nx
    ny=s.ny
    nz=s.nz
    den=np.zeros((nx,ny,nz,na,nb),dtype=np.complex128)
    for x,y,z in product(range(s.nx),range(s.ny),range(s.nz)):
      den[x,y,z]=s.d[x,y,z].todense()
    return den

def sparse_angles(M):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  AngM=SM(M.shape,dtype=float)
  AngM2=SM(M.shape,dtype=float)
  indices=M.nonzero()
  AngM2[indices]=np.angle(M.todense()[indices])
  t1=time.time()
  AngM2[indices]=np.angle(M.todense()[indices])
  t2=time.time()
  AngM[indices]=np.angle(M[indices].todense())
  t3=time.time()
  print('first time', t2-t1)
  print('second time',t3-t2)
  print('Difference in times', t2-t1-t3+t2)
  return AngM

def dict_sparse_angles(DSM):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  nx,ny,nz=DSM.nx, DSM.ny, DSM.nz
  na,nb=DSM[0,0,0].shape
  AngDSM=DS(nx,ny,nz,na,nb)
  indices=DSM.nonzero()
  #DSM2=DSM.dense()
  #AngDSM[indices]=np.angle(DSM.dense()[indices])
  return 0

def test_00():
  ds=DS()
  ds[1,2,3,0,0]=2+3j
  print(ds[1,2,3])
  print(ds[1,2,3][0,0])

def test_01(nx=3,ny=2,nz=1,na=5,nb=6):
  '''testing creation of dictionary containing sparse matrices'''
  ds=DS(nx,ny,nz,na,nb)

def test_02(nx=7,ny=6,nz=1,na=5,nb=6):
  '''testing creation of matrix and adding on element'''
  ds=DS(nx,ny,nz,na,nb)
  ds[0,3,0,2,0]=2+3j
  print(ds[0,3,0])

def test_03(nx,ny,nz,na,nb):
  '''testing creation of diagonal sparse matrices contained in every position'''
  ds=DS(nx,ny,nz,na,nb)
  for x,y,z,a in product(range(nx),range(ny),range(nz),range(na)):
    if a<nb:
      ds[x,y,z,a-1,a-1]=complex(a,a)
  return ds

def test_03b(nx,ny,nz,na,nb):
  '''testing creation of first column sparse matrices contained in every position'''
  ds=DS(nx,ny,nz,na,nb)
  for x,y,z,a in product(range(nx),range(ny),range(nz),range(na)):
    if a<nb:
      ds[x,y,z,a-1,2]=complex(a,a)
  return ds

def test_03c(nx,ny,nz,na,nb):
  '''testing creation of lower triangular sparse matrices contained in every position'''
  ds=DS(nx,ny,nz,na,nb)
  for x,y,z,a in product(range(nx),range(ny),range(nz),range(na)):
    if a<nb:
      for ai in range(a-1,na):
        ds[x,y,z,ai,a-1]=complex(a,a)
  return ds

def test_04():
  '''testing matrix addition operation'''
  ds=test_03(7,6,1,5,6)
  M=ds[2,0,0]+ds[0,1,0]
  ds[0,0,0]=M
  print(ds[0,0,0])

def test_05():
  '''testing get column'''
  ds=test_03b(7,6,1,6,5)
  print(ds[0,0,0,:,2])
  print(ds[0,0,0,0,:])
  print(ds[0,0,0,1:6:2,:])

def test_06():
  '''testing matrix multiplication'''
  ds1=test_03(7,6,1,5,5)
  ds2=test_03b(7,6,1,5,5)
  M0=ds1[0,0,0]*ds2[0,0,0]
  M1=ds1[5,5,0]*ds2[6,5,0]
  print(M0)
  print(M1)

def test_07():
  '''testing getting angle from complex entries in matrix'''
  ds=test_03(3,3,1,3,3)
  M0=ds[0,0,0]
  indices=zip(*M0.nonzero())
  M1= SM(M0.shape,dtype=np.float)
  for i,j in indices:
    M1[i,j]=np.angle(M0[i,j])
  print(M0,M1)

def test_08():
  '''testing getting angle from complex entries in matrix then taking the cosine of every nonzero entry'''
  ds=test_03(3,3,1,3,3)
  M0=ds[0,0,0]
  indices=zip(*M0.nonzero())
  M1= SM(M0.shape,dtype=np.float)
  for i,j in indices:
    M1[i,j]=np.cos(np.angle(M0[i,j]))
  print(M0,M1)

def test_09():
  '''testing operation close to Fresnel reflection formula'''
  obs=np.array([1.0,2.0,3.0])
  obs=obs*np.eye(3)
  ds=test_03(3,3,1,3,3)
  M0=ds[0,0,0]
  indices=zip(*M0.nonzero())
  M1= SM(M0.shape,dtype=np.float)
  M2= SM(M0.shape,dtype=np.float)
  for i,j in indices:
    M1[i,j]=np.cos(np.angle(M0[i,j]))
    M2[i,j]=np.cos(0.7*np.angle(M0[i,j]))
  N1=M1[:,0].T*obs-M2[:,0].T
  N2=M1[:,0].T*obs+M2[:,0].T
  return (N1) #[N1.nonzero()]) #/(N2[N2.nonzero()]))

def test_10():
  '''Multiply by coefficient and sum the nonzero terms in the columns'''
  refcoef=test_09()
  ds=test_03b(3,3,1,3,3)
  M0=ds[0,0,0]
  M0[:,2]=(M0[:,2].T*refcoef.T).T
  indices=M0.nonzero()
  field=1.0 #np.zeros(M0.shape[1],1)
  for j in indices[1]:
    for i in indices[0]:
      field*=abs(M0[i,j])
  return 0

def test_11():
  '''Extract reflection angles from matrix'''
  nx=1
  ny=1
  nz=1
  na=1000
  nb=1000
  DSM=test_03c(nx,ny,nz,na,nb)
  for i,j,k in product(range(nx),range(ny),range(nz)):
      M=DSM[i,j,k]
      AngM=sparse_angles(M)
  #print(AngM)
  return AngM

def test_12():
  '''Extract the cos of the reflection angles of the matrix'''
  nx=3
  ny=3
  nz=1
  na=3
  nb=3
  DSM=test_03c(nx,ny,nz,na,nb)
  for i,j,k in product(range(nx),range(ny),range(nz)):
      M=DSM[i,j,k]
      indices=M.nonzero()
      CosAngM=SM(M.shape,dtype=float)
      CosAngM[indices]=np.cos(sparse_angles(M)[indices].todense())
  print(CosAngM)
  return CosAngM

def test_13():
  '''Attempt to create an array of the indices nonzero element of SM inside dictionary'''
  nx=3
  ny=3
  nz=1
  na=3
  nb=3
  DSM=test_03c(nx,ny,nz,na,nb)
  print(DSM[0,0,0][DSM[0,0,0].nonzero()])
  #print(DSM.nonzero())


def test_14():
  '''Attempt to find angle of nonzero element of SM inside dictionary'''
  nx=3
  ny=3
  nz=1
  na=3
  nb=3
  DSM=test_03c(nx,ny,nz,na,nb)
  ang=dict_sparse_angles(DSM)
  return 1


if __name__=='__main__':
  test_11()
  print('Running  on python version')
  print(sys.version)

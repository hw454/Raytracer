#!/usr/bin/env python3
# Hayley 2019-02-06

import numpy as np
from scipy.sparse import lil_matrix as SM
from itertools import product

# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix
class DS:
  def __init__(s,nx=1,ny=1,nz=1,na=1,nb=1):
    '''nx,ny,nz are the maximums for the key for the dictionary and na and nb are the dimensions of the sparse matrix associated with each key.'''
    s.shape=(na,nb)
    s.d={}
    for x, y, z in product(range(nx),range(ny),range(nz)):
      s.d[x,y,z]=SM(s.shape,dtype=np.complex128)
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

     
if __name__=='__main__':
  test_05()

#!/usr/bin/env python3
# Hayley 2019-02-06

import numpy as np
from scipy.sparse import lil_matrix as SM

# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix
class DS:
  def __init__(s,nx=1,ny=1,nz=1,na=1,nb=1):
    '''nx,ny,nz are the maximums for the key for the dictionary and na and nb are the dimensions of the sparse matrix associated with each key.'''
    s.shape=(na,nb)
    s.d={}
    for x in range(nx):
        for y in range(ny):
          for z in range(nz):
              s.d[x,y,z]=SM(s.shape,dtype=np.complex128)
  def __getitem__(s,i):
    if len(i)==3:
      dk=i[:3]
      return s.d[dk]                # return a SM
    elif len(i)==4:
      dk,smk=i[:3],i[3]
      return s.d[dk][smk[0],:]      # return a SM row
    elif len(i)==5:
      dk,smk=i[:3],i[3:]
      return s.d[dk][smk[0],smk[1]] # return a SM element
    else:
      # ...
      print('invalid position. a 3-tuple is required to return a sparse matrix(SM), 4-tuple for the row of a SM or 5-tuple for the element in the SM.')
      pass 
  def __setitem__(s,i,x):
    dk,smk=i[:3],i[3:] 
    if dk not in s.d:
      s.d[dk]=SM(0.0,shape=s.shape,dtype=np.complex128)
    s.d[dk][smk[0],smk[1]]=x
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
  ds=DS(nx,ny,nz,na,nb)

def test_02(nx=7,ny=6,nz=1,na=5,nb=6):
  ds=DS(nx,ny,nz,na,nb)
  ds[0,3,0,2,0]=2+3j 
  print(ds[0,3,0])

def test_03(nx=7,ny=6,nz=1,na=5,nb=6):
  ds=DS(nx,ny,nz,na,nb)
  for x in range(nx):
    for y in range(ny):
      for z in range(nz):
        for a in range(na):
          if a<nb:
            ds[x,y,z,a,a]=complex(a,a) 
        print(ds[x,y,z])   
     
if __name__=='__main__':
  test_03() 

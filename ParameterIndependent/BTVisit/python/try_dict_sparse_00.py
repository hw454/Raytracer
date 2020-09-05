#!/usr/bin/env python3
# Keith Briggs 2019-02-06

import numpy as np
from scipy.sparse import lil_matrix as SM

class DS:
  def __init__(s,shape=(1,1)):
    s.d={}
    s.shape=shape
  def __getitem__(s,i):
    if len(i)==3:
      dk=i[:3]
      return s.d[dk] # return a SM
    elif len(i)==5:
      dk,smk=i[:3],i[3:]
      return s.d[dk][smk[0],smk[1]] # return a SM element
    else:
      # ...
      print('!!!')
      pass # FIXME
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

def test_01(nx=3,ny=2,nz=1):
  ds=DS()
  for x in range(nx):
    for y in range(ny):
      for z in range(nz):
         ds[x,y,z,0,0]=1j
       
if __name__=='__main__':
  test_01() 

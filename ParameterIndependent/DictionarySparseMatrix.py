#!/usr/bin/env python3
# Hayley 2019-05-01

import numpy as np
from scipy.sparse import lil_matrix as SM
from itertools import product
import math
import sys
import time as t
import matplotlib.pyplot as mp

# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix

class DS:
  def __init__(s,nx=1,ny=1,nz=1,na=1,nb=1):
    '''nx,ny,nz are the maximums for the key for the dictionary and na
    and nb are the dimensions of the sparse matrix associated with each key.'''
    s.shape=(na,nb)
    Keys=product(range(nx),range(ny),range(nz))
    default_value=SM(s.shape,dtype=np.complex128)
    s.d=dict.fromkeys(Keys,default_value)
    s.nx=nx
    s.ny=ny
    s.nz=nz
    s.time=np.array([t.time()])
  def __getitem__(s,i):
    dk,smk=i[:3],i[3:]
    if isinstance(dk[0],(float,int,np.int64, np.complex128 )): n=1
    else:
      n=len(dk)
    if n==1:
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
    else:
      if isinstance(smk,(float,int,np.complex128,np.int64)):
        # If smk is just a value then all rows smk in every corresponding
        # Key element must be returned
        n2=1
      else:
        n2=len(smk)
      if n2>2:
        out=s.d[dk[0][0],dk[1][0],dk[2][0]][smk,:]
      elif n2==2:
        out=s.d[dk[0][0],dk[1][0],dk[2][0]][ smk[0][0],smk[1][0]]
      elif n2==0:
        out=s.d[dk[0][0],dk[1][0],dk[2][0]]
      else:
        raise Exception('Error, not a valid SM dimension')
      for j in range(1,len(dk[0])):
        if n2>2:
          out=np.vstack((out,s.d[dk[0][j],dk[1][j],dk[2][j]][smk,:]))
        elif n2==2:
          out=np.vstack((out,s.d[dk[0][j],dk[1][j],dk[2][j]][ smk[0][j],smk[1][j]]))
        elif n2==0:
          out=np.vstack((out,s.d[dk[0][j],dk[1][j],dk[2][j]]))
        else:
          raise Exception('Error, not a valid SM dimension')
      return out
  def __setitem__(s,i,x):
    dk,smk=i[:3],i[3:]
    if isinstance(dk[0],(float,int,np.int64, np.complex128 )): n=1
    else:
        n=len(dk)
    if n==1:
        if len(i)==3:
          # set a SM to an input SM
         if dk not in s.d:
           s.d[dk]=SM(s.shape,dtype=np.complex128)
         s.d[dk]=x
        elif len(i)==4:                 # set a SM row
          if isinstance(smk[0],(float,int,np.int64, np.complex128)): n2=1
          else:
            n2=len(smk[0]) 
          if dk not in s.d:
            s.d[dk]=SM(s.shape,dtype=np.complex128)
          if n2==1:
            s.d[dk][smk[0],:]=x
          else:
            p=0
            for j in smk[0]:
              s.d[dk][smk,:]=x[p]
              p+=1
        elif len(i)==5:                # set a SM element
          if isinstance(smk[0],(float,int,np.int64, np.complex128)): n2=1
          else:
            n2=len(smk[0])
          if dk not in s.d:
            s.d[dk]=SM(s.shape,dtype=np.complex128)
          if n2==1:
            s.d[dk][smk[0],smk[1]]=x
          else:
            end=len(smk[0])
            for c in range(0,end):
              s.d[dk][smk[0][c],smk[1][c]]=x[c]
        else:
          # ...
          raise Exception('Error setting the (%s) part of the sparse matrix to (%s). Invalid index (%s). A 3-tuple is required to return a sparse matrix(SM), 4-tuple for the row of a SM or 5-tuple for the element in the SM.' %(i,x,i))
          pass
    else:
      if isinstance(x,(float,int,np.complex128,np.int64)):
        n1=0
      elif x.shape==s.shape: n1=0
      elif x.shape==s.shape[0]: n1=0
      else: n1=len(x)
      if isinstance(smk,(float,int,np.complex128,np.int64)):
        # If smk is just a value then all rows in every corresponding
        # Key element must be set to the same x
        value=SM(s.shape,dtype=np.complex128)
        if n1==0:
          value[ smk, : ] = x
        n2=1
      else:
        value=SM(s.shape,dtype=np.complex128)
        n2=len(smk)
      for j in range(len(dk[0])):
        if n2>2:
          if n1==0:
            value[ smk[j], : ]          = np.asscalar(x)         # Setting a row j to be x
          else:
            value[ smk[j], : ]          = np.asscalar(x[j])      # Setting a row j to be the j'th x
        elif n2==1 and n1>0:
          value[ smk , : ]=x[j]                     # Setting the fixed row to be the j'th x
        elif n2==2:
          if n1==0:
            value[ smk[0][j],smk[1][j]] = np.asscalar(x)         # Setting a point to be x
          else:
            value[ smk[0][j],smk[1][j]] = np.asscalar(x[j])      # Setting a point to be j'th x
        elif n2==0:
          if n1==0:
            value=np.asscalar(x)                                 # Setting the whole sparse matrix to be x
          else:
            value=np.asscalar(x[j])                              # Setting the sparse matrix to be the j'th in x
        else:
          raise Exception('Error, not a valid SM dimension')
        s.d[dk[0][j],dk[1][j],dk[2][j]]=value
      return
  def __str__(s):
    return str(s.d)
  def __repr__(s):
    return str(s.d) # TODO
  def nonzero(s):
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      indicesM=s.d[x,y,z].nonzero()
      NI=len(indicesM[0])
      for j in range(0,NI):
        if x==0 and y==0 and z==0 and j==0:
          indices=np.array([0,0,0,indicesM[0][j],indicesM[1][j]])
        else:
          indices=np.vstack((indices,[x,y,z,indicesM[0][j],indicesM[1][j]]))
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
  def stopcheck(s,i,j,k,p1,h):
    #if i>=p1[0] and j>=p1[1] and k>=p1[2]:
    #  return 0
    if i>s.nx or j>s.ny or k>s.nz or i<0 or j<0 or k<0:
      return 0
    else: return 1
  def stopchecklist(s,ps,p1,h):
    start=0
    newps=np.array([])
    for k in ps:
      check=s.stopcheck(k[0],k[1],k[2],p1,h)
      if check==1:
        if start==0:
          newps=np.array([[k[0]],[k[1]],[k[2]]])
          start=1
        else:
          newps=np.hstack((newps,np.array([[k[0]],[k[1]],[k[2]]])))
      else:
        pass
    return start, newps



def sparse_angles(M):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  AngM=SM(M.shape,dtype=float)
  AngM2=SM(M.shape,dtype=float)
  indices=M.nonzero()
  AngM2[indices]=np.angle(M.todense()[indices])
  t1=t.time()
  AngM2[indices]=np.angle(M.todense()[indices])
  t2=t.time()
  AngM[indices]=np.angle(M[indices].todense())
  t3=t.time()
  print('first time', t2-t1)
  print('second time',t3-t2)
  print('Difference in times', t2-t1-t3+t2)
  return AngM

def dict_sparse_angles(DSM):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  nx,ny,nz=DSM.nx, DSM.ny, DSM.nz
  na,nb=DSM[0,0,0].shape
  AngDSM=DS(nx,ny,nz,na,nb)
  indices=np.transpose(DSM.nonzero())
  AngDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.angle(DSM[indices[0],indices[1],indices[2],indices[3],indices[4]])
  return AngDSM
  
def dict_cos(DSM):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  nx,ny,nz=DSM.nx, DSM.ny, DSM.nz
  na,nb=DSM[0,0,0].shape
  CosDSM=DS(nx,ny,nz,na,nb)
  indices=np.transpose(DSM.nonzero())
  CosDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.cos(DSM[indices[0],indices[1],indices[2],indices[3],indices[4]])
  return CosDSM
  
def dict_sin(DSM):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  nx,ny,nz=DSM.nx, DSM.ny, DSM.nz
  na,nb=DSM[0,0,0].shape
  SinDSM=DS(nx,ny,nz,na,nb)
  indices=np.transpose(DSM.nonzero())
  SinDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.sin(DSM[indices[0],indices[1],indices[2],indices[3],indices[4]])
  return SinDSM
  
def dict_asin(DSM):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  nx,ny,nz=DSM.nx, DSM.ny, DSM.nz
  na,nb=DSM[0,0,0].shape
  asinDSM=DS(nx,ny,nz,na,nb)
  indices=np.transpose(DSM.nonzero())
  asinDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.asin(DSM[indices[0],indices[1],indices[2],indices[3],indices[4]])
  return asinDSM
  
def dict_vec_multiply(vec,DSM):
  '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
  nx,ny,nz=DSM.nx, DSM.ny, DSM.nz
  na,nb=DSM[0,0,0].shape
  outDSM=DS(nx,ny,nz,na,nb)
  indices=np.transpose(DSM.nonzero())
  for i in indices[0]:
    for j in indices[1]:
      for k in indices[2]:
        out=np.multiply(vec[indices[3],indices[4]],DSM[i,j,k,indices[3],indices[4]])
        print(i,j,k)
        outDSM[i,j,k,indices[3],indices[4]]=np.multiply(vec[indices[3],indices[4]],DSM[i,j,k,indices[3],indices[4]])
  return outDSM 

def ref_coef(DSM,roomcoefs):
  return 0

def test_00():
  ds=DS()
  ds[1,2,3,0,0]=2+3j
  print(ds[1,2,3][0,0])

def test_01(nx=3,ny=2,nz=1,na=5,nb=6):
  '''testing creation of dictionary containing sparse matrices'''
  ds=DS(nx,ny,nz,na,nb)

def test_02(nx=7,ny=6,nz=1,na=5,nb=6):
  '''testing creation of matrix and adding on element'''
  ds=DS(nx,ny,nz,na,nb)
  ds[0,3,0,:,0]=2+3j
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
  nx=2
  ny=2
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
  nx=2
  ny=2
  nz=1
  na=3
  nb=3
  DSM=test_03c(nx,ny,nz,na,nb)
  print(DSM[0,0,0][DSM[0,0,0].nonzero()])
  #print(DSM.nonzero())


def test_14():
  '''Attempt to find angle of nonzero element of SM inside dictionary'''
  nx=2
  ny=2
  nz=1
  na=3
  nb=3
  DSM=test_03c(nx,ny,nz,na,nb)
  ang=dict_sparse_angles(DSM)
  return 1

def test_15():
  ds=test_03(8,6,1,11,6)
  Nb=ds.shape[1]
  # Number of obstacles and the corresponding coefficients
  Nre=1
  Nob=int((ds.shape[0]-1)/(Nre+1))
  mur=np.full((Nob,1), complex(1.0,0))
  epsr=np.full((Nob,1),complex(3.6305,7.41E-2))
  sigma=np.full((Nob,1),1.0E-2)

  # PHYSICAL CONSTANTS
  mu0=4*np.pi*1E-6
  c=2.99792458E+8
  eps0=1/(mu0*c**2)#8.854187817E-12
  Z1=(mu0/eps0)**0.5 #120*np.pi

  # CALCULATE PARAMETERS
  frequency=2*np.pi*2.43E+9                       # 2.43 GHz
  gamma=np.sqrt(np.divide(complex(0,frequency*mu0)*mur,np.multiply(sigma,eps0*frequency*complex(0,1)*epsr)))
  Znob=Z1*np.divide((1+gamma),(1-gamma)   )    # Characteristic impedance of the obstacles
  Znob=np.tile(Znob,Nre+1)
  Znob=np.insert(Znob,0,complex(0.0,0.0))
  Znob=np.transpose(np.tile(Znob,(Nb,1)))
  refindex=np.sqrt(np.multiply(mur,epsr))     # Refractive index of the obstacles

  AngDSM=dict_sparse_angles(ds)
  indices=AngDSM.nonzero()
  S=dict_vec_multiply(Znob,dict_cos(AngDSM))#-Z1*np.cos(np.asin(np.divide(np.sin(AngDSM[indices]),refindex[indices])))
  # np.multiply(Znob[indices],np.cos(AngDSM[indices])
  # Divide
  # Znob[nonzero]*Cos(AngDSM[nonzero])+Z1cos(asin(sin(AngDSM[nonzero])/ref[nonzero]))
  # Z1*Cos(AngDSM[nonzero])-Znob[nonzero]cos(asin(sin(AngDSM[nonzero])/ref[nonzero]))
  # Divide
  # Z1*Cos(AngDSM[nonzero])+Znob[nonzero]cos(asin(sin(AngDSM[nonzero])/ref[nonzero]))
  return


if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  test_15()

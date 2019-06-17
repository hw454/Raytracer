#!/usr/bin/env python3
# Hayley 2019-05-01

import numpy as np
from scipy.sparse import lil_matrix as SM
from itertools import product
import math
import sys
import time as t
import matplotlib.pyplot as mp
from six.moves import cPickle as pkl

epsilon=sys.float_info.epsilon
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
          if isinstance(smk[0],(float,int,np.int64, np.complex128,slice)): n2=1
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
    keys=s.d.keys()
    out=str()
    for k in keys:
      new= str(k) + str(s.d[k])
      out= (""" {0}
               {1}""").format(out,new)
    return out
  def __repr__(s):
    return str(s.d) # TODO #FIXME
  def __add__(s, DSM):
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      out[x,y,z]=s[x,y,z]+DSM[x,y,z]
    return out
  def __sub__(s, DSM):
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      out[x,y,z]=s[x,y,z]-DSM[x,y,z]
    return out
  def __mul__(s, DSM):
    t0=t.time()
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      out[x,y,z]=np.multiply(s[x,y,z],DSM[x,y,z])
    t1=t.time()-t0
    print('time multiplying DSMs',t1)
    return out
  def __truediv__(s, DSM):
    t0=t.time()
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      ind=DSM[x,y,z].nonzero()
      out[x,y,z][ind]=np.true_divide(s[x,y,z][ind],DSM[x,y,z][ind])
    t1=t.time()-t0
    print('time dividing DSMs',t1)
    return out
  def asin(s):
    '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
    t0=t.time()
    na,nb=s.shape
    asinDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=np.transpose(s.nonzero())
    asinDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.asin(DSM[indices[0],indices[1],indices[2],indices[3],indices[4]])
    t1=t.time()-t0
    print('time finding arcsin(theta)',t1)
    return asinDSM
  def cos(s):
    '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
    t0=t.time()
    na,nb=s.shape
    CosDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=np.transpose(s.nonzero())
    CosDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.cos(s[indices[0],indices[1],indices[2],indices[3],indices[4]])
    t1=t.time()-t0
    print('time finding cos(theta)',t1)
    return CosDSM
  def sin(s):
    '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
    na,nb=s.shape
    SinDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=np.transpose(s.nonzero())
    SinDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.sin(s[indices[0],indices[1],indices[2],indices[3],indices[4]])
    return SinDSM
  def sparse_angles(s):
    '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
    print('start angles')
    t0=t.time()
    na,nb=s.shape
    AngDSM=DS(s.nx,s.ny,s.nz,na,nb)
    t1=t.time()
    print('before transpose',t1-t0)
    indices=s.nonzero()
    print('indices shape', indices.shape)
    indices=indices.T
    t2=t.time()
    print('before angles',t2-t0)
    AngDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.angle(s[indices[0],indices[1],indices[2],indices[3],indices[4]])
    t3=t.time()-t0
    print('time finding angles',t3)
    return AngDSM
  def vec_multiply(s,vec):
    '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
    na,nb=s.shape
    outDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=s.nonzero()
    Ni=len(indices)
    for l in range(0,Ni):
      out=np.multiply(vec[indices[l][3]],s[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]])
      outDSM[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]]=out
    return outDSM
  def dict_DSM_divideby_vec(s,vec):
    '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
    na,nb=s.shape
    outDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=np.transpose(vec.nonzero())
    for x,y,z,a,b in product(range(0,s.nx),range(0,s.ny),range(0,s.nz),indices,range(0,nb)):
      a=a[0]
      if abs(s[x,y,z,a,b])<epsilon:
        pass
      else:
        out=np.divide(s[x,y,z,a,b],vec[a])
        outDSM[x,y,z,a,b]=out
    return outDSM
  def dict_vec_divideby_DSM(s,vec):
    '''Takes in a complex sparse matrix M and outputs the arguments of the nonzero() entries'''
    na,nb=s.shape
    outDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=s.nonzero()
    Ni=len(indices[0])
    for l in range(0,Ni):
      out=np.divide(vec[indices[l][3]],s[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]])
      outDSM[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]]=out
    return outDSM
  def save_dict(s, filename_):
    with open(filename_, 'wb') as f:
        pkl.dump(s.d, f)
    return
  def nonzero(s):
    # FIXME this is too slow and needs parallelising / speeding up.
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      indicesM=s.d[x,y,z].nonzero()
      NI=len(indicesM[0])
      for j in range(0,NI):
        if x==0 and y==0 and z==0 and j==0:
          indices=np.array([0,0,0,indicesM[0][j],indicesM[1][j]])
        else:
          indices=np.vstack((indices,[x,y,z,indicesM[0][j],indicesM[1][j]]))
      #if x==0 and y==0 and z==0:
      #  indices=np.array([0,0,0,indicesM[0][0],indicesM[1][0]])
      #  indicesSec=np.c_[np.tile(np.array([x,y,z]),(1,NI-1)),IndicesM[1:-1]]
      #  indices=np.vstack((indices,indicesSec))
      #else:
      #  indicesSec=np.c_[np.tile(np.array([x,y,z]),(1,NI)),IndicesM[0:-1]]
      #  indices=np.vstack((indices,indicesSec))
    return indices
  def parnonzero(s):
    # FIXME this is too slow and needs parallelising / speeding up.
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
  def stopchecklist(s,ps,p1,h,p3,n):
    start=0
    newps=np.array([])
    newp3=np.array([])
    newn =np.array([])
    j=0
    for k in ps:
      check=s.stopcheck(k[0],k[1],k[2],p1,h)
      if check==1:
        if start==0:
          newps=np.array([[k[0]],[k[1]],[k[2]]])
          newp3=np.array([[p3[j][0]],[p3[j][1]],[p3[j][2]]])
          newn =np.array([[n[j][0]], [n[j][1]], [n[j][2]]])
          start=1
        else:
          newps=np.hstack((newps,np.array([[k[0]],[k[1]],[k[2]]])))
          newp3=np.hstack((newp3,np.array([[p3[j][0]],[p3[j][1]],[p3[j][2]]])))
          newn =np.hstack((newn, np.array([[n[j][0]], [n[j][1]], [ n[j][2]]])))
      else:
        pass
      j+=1
    return start, newps, newp3, newn

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pkl.load(f)
    return ret_di

def ref_coef(Mesh,FreeSpace,freq,Znob,refindex):
  print('-------------------------------')
  print('Retrieving the angles of reflection')
  print('-------------------------------')
  AngDSM=Mesh.sparse_angles()                       # Get the angles of incidence from the mesh.
  ind=AngDSM.nonzero()                              # Return the indices for the non-zero terms in the mesh.
  ind=np.transpose(ind)
  SIN=DS(Mesh.nx,Mesh.ny,Mesh.nz,Mesh.shape[0],Mesh.shape[1])   # Initialise a DSM which will be sin(theta)
  cthi=DS(Mesh.nx,Mesh.ny,Mesh.nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta)
  ctht=DS(Mesh.nx,Mesh.ny,Mesh.nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta_t)
  print('-------------------------------')
  print('Computing cos(theta_i) on all reflection terms')
  print('-------------------------------')
  cthi=AngDSM.cos()                                   # Compute cos(theta_i)
  print('-------------------------------')
  print('Computing cos(theta_t) on all reflection terms')
  print('-------------------------------')
  SIN[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.sin(AngDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]) # Compute sin(theta)
  Div=SIN.dict_DSM_divideby_vec(refindex)             # Divide each column in DSM with refindex elementwise. Set any 0 term to 0.
  ctht[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.cos(np.arcsin(Div[ind[0],ind[1],ind[2],ind[3],ind[4]]))
  print('-------------------------------')
  print('Multiplying by the impedances')
  print('-------------------------------')
  S1=(cthi).vec_multiply(Znob)                # Compute S1=Znob*cos(theta_i)
  S2=Z0*ctht                                  # Compute S2=Z0*cos(theta_t)
  S3=Z0*cthi                                  # Compute S3=Z0*cthi
  S4=(ctht).vec_multiply(Znob)                # Compute S4=Znob*cos(theta_t)
  print('-------------------------------')
  print('Computing the reflection coefficients.')
  print('-------------------------------')
  Sper=(S1-S2)/(S1+S2)                        # Compute the Reflection coeficient perpendicular
                                              # to the polarisiation S=(S1-S2)/(S1+S2)  S1=(cthi).vec_multiply(Znob)                # Compute S1=Znob*cos(theta_i)

  Spar=(S3-S4)/(S3+S4)                        # Compute the Reflection coeficient parallel
                                              # to the polarisiation S=(S3-S4)/(S3+S4)
  return Sper, Spar

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
  ds=test_03(8,6,1,11,6)                       # test_03() initialises a
                                               # DSM with values on the
                                               # diagonal of each mesh element
  Nb=ds.shape[1]                               # Number of columns on each Mesh element
  Nre=1                                        # Number of reflections
  Nob=int((ds.shape[0]-1)/(Nre+1))             # The Number of obstacle.
  mur=np.full((Nob,1), complex(1.0,0))         # For this test mur is
                                               # the same for every obstacle.
                                               # Array created to get functions correct.
  epsr=np.full((Nob,1),complex(3.6305,7.41E-2))# For this test epsr is the
                                               # same for every obstacle
  sigma=np.full((Nob,1),1.0E-2)                # For this test sigma is the
                                               # same for every obstacle

  # PHYSICAL CONSTANTS
  mu0=4*np.pi*1E-6
  c=2.99792458E+8
  eps0=1/(mu0*c**2)#8.854187817E-12
  Z0=(mu0/eps0)**0.5 #120*np.pi

  # CALCULATE PARAMETERS
  frequency=2*np.pi*2.43E+9                       # 2.43 GHz
  gamma=np.sqrt(np.divide(complex(0,frequency*mu0)*mur,np.multiply(sigma,eps0*frequency*complex(0,1)*epsr)))
  Znob=Z0*np.divide((1+gamma),(1-gamma)   )   # Characteristic impedance of the obstacles
  Znob=np.tile(Znob,Nre)                      # The number of rows is Nob*Nre+1. Repeat Nob
  Znob=np.insert(Znob,0,complex(0.0,0.0))     # Use a zero for placement in the LOS row
  #Znob=np.transpose(np.tile(Znob,(Nb,1)))    # Tile the obstacle coefficient number to be the same size as a mesh array.
  refindex=np.sqrt(np.multiply(mur,epsr))     # Refractive index of the obstacles
  refindex=np.tile(refindex,Nre)
  refindex=np.insert(refindex,0,complex(0,0))

  AngDSM=ds.sparse_angles()
  ind=AngDSM.nonzero()
  ind=np.transpose(ind)
  SIN=DS(ds.nx,ds.ny,ds.nz,ds.shape[0],ds.shape[1])
  S1=DS(ds.nx,ds.ny,ds.nz,ds.shape[0],ds.shape[1])
  S2=DS(ds.nx,ds.ny,ds.nz,ds.shape[0],ds.shape[1])
  S1=(AngDSM.cos()).vec_multiply(Znob)
  SIN[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.sin(AngDSM[ind[0],ind[1],ind[2],ind[3],ind[4]])
  Div=SIN.dict_DSM_divideby_vec(refindex)
  S2[ind[0],ind[1],ind[2],ind[3],ind[4]]=Z0*np.cos(np.arcsin(Div[ind[0],ind[1],ind[2],ind[3],ind[4]]))
  S=(S1-S2)/(S1+S2)
  # Znob[nonzero]*Cos(AngDSM[nonzero])+Z1cos(asin(sin(AngDSM[nonzero])/ref[nonzero]))
  # Z1*Cos(AngDSM[nonzero])-Znob[nonzero]cos(asin(sin(AngDSM[nonzero])/ref[nonzero]))
  # Divide
  # Z1*Cos(AngDSM[nonzero])+Znob[nonzero]cos(asin(sin(AngDSM[nonzero])/ref[nonzero]))
  return

def test_16():
  Nob=24
  Nre=3
  Nra=20
  ktries=20
  narray=np.zeros((ktries,1),dtype=np.float)
  n=2
  timevec=np.zeros((ktries,1),dtype=np.float)
  for k in range(0,ktries):
    narray[k]=n
    DS=test_03(n,n,n,int(Nob*Nre+1),int((Nre)*(Nra)+1))
    t0=t.time()
    indices=DS.nonzero()
    timevec[k]=t.time()-t0
    print(timevec[k])
    n=n*2
  mp.plot(narray,timevec)
  mp.show
  return timevec

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  test_16()

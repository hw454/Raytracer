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
from multiprocessing import Pool
#import pp

epsilon=sys.float_info.epsilon
# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix

## The DS class is a dictionary of sparse matrices.
# The keys for the dictionary are (i,j,k) such that i is in [0,Nx],
# j is in [0, Ny], and k is in [0,Nz].

class DS:
  ''' The DS class is a dictionary of sparse matrices.
  The keys for the dictionary are (i,j,k) such that i is in [0,Nx],
  j is in [0, Ny], and k is in [0,Nz]. '''
  ## The constructor.
  # \par
  # DSM is a dictionary of sparse matrices with keys (x,y,z) such that x is in [0,Nx],
  # y is in [0, Ny], and z is in [0,Nz].
  # @param Nx number of terms in the first index
  # @param Ny number of terms in the second index
  # @param Nz number of terms in the third index
  # \par
  # SM=DS[x,y,z] is a sparse matrix with dictionary key [x,y,z]
  # @param na number of rows in each sparse matrix
  # @param nb number of columns in each sparse matrix
  def __init__(s,Nx=1,Ny=1,Nz=1,na=1,nb=1):
    s.shape=(na,nb)
    Keys=product(range(Nx),range(Ny),range(Nz))
    default_value=SM(s.shape,dtype=np.complex128)
    s.d=dict.fromkeys(Keys,default_value)
    s.Nx=Nx
    s.Ny=Ny
    s.Nz=Nz
    s.time=np.array([t.time()])
  ## @param i
  # - If 'i' has length 5, i=[x,y,z,k,j] then this is the position of a
  # single term out=A[k,j] in a matrix A=DSM[x,y,z], return out.
  # - 'i' has length 4 then i=[x,y,z,k], this is the position of a
  # row out=A[k] (array of length nb) of a sparse matrix corresponding
  # to the dictionary key A=DSM[x,y,z].
  # - If 'i' has length 3 then i is the position of the whole sparse
  # matrix out=A for the sparse matrix at location [x,y,z].
  # .
  # \par
  # The 'k' and 'j' indices can be replaced with : to return rows or
  # columns of the SM A=DSM[x,y,z].
  # @return the 'i' term of the DSM
  def __getitem__(s,i):
    dk,smk=i[:3],i[3:]
    # If dk is a number then one position (x,y,z) is being refered to.
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
        return s.d[dk][smk[0],smk[1]] # If smk=[int,int] return element,
                                      # If smk=[:,int] return column,
                                      # If smk=[int,:] return row,
                                      # If smk=[:,:] return full SM.
      else:
        # Invalid 'i' the length does not match a possible position.
        errmsg=str('''Error getting the (%s) part of the sparse matrix.
        Invalid index (%s). A 3-tuple is required to return a sparse
        matrix(SM), 4-tuple for the row of a SM or 5-tuple for the
        element in the SM.''' %(i,i))
        raise IndexError(errmsg)
        pass
    # If dk[0] is not a number then multiple (x,y,z) terms are being called.
    else:
      # If smk is just a value then all rows smk in every dk element is returned.
      if isinstance(smk,(float,int,np.complex128,np.int64)):
        n2=1
      else:
        n2=len(smk)
      # Return the SMK rows for the [x,y,z] matrices in dk.
      if n2>2:
        out=s.d[dk[0][0],dk[1][0],dk[2][0]][smk,:]
      # Return the SM[k,j] term for all [x,y,z] matrices in dk.
      elif n2==2:
        out=s.d[dk[0][0],dk[1][0],dk[2][0]][ smk[0][0],smk[1][0]]
      # Return the entire SM matrix for all [x,y,z] matrices in dk
      elif n2==0:
        out=s.d[dk[0][0],dk[1][0],dk[2][0]]
      # If smk is not a number or length 2 then it is invalid and can
      # not be found in the sparse matrix
      else:
        raise IndexError('Error, not a valid SM dimension')
      ## Iterate through the list of (x,y,z) keys and return the
      # corresponding element in the SM at that key.
      for j in range(1,len(dk[0])):
        if n2>2:
          # Different rows
          out=np.vstack((out,s.d[dk[0][j],dk[1][j],dk[2][j]][smk,:]))
        elif n2==2:
          out=np.vstack((out,s.d[dk[0][j],dk[1][j],dk[2][j]][ smk[0][j],smk[1][j]]))
        elif n2==0:
          out=np.vstack((out,s.d[dk[0][j],dk[1][j],dk[2][j]]))
        else:
          raise IndexError('Error, not a valid SM dimension')
      return out
  ## Set a new value to all or part of a DSM.
  # param i
  # - If 'i' has length 5, i=[x,y,z,k,j] then this is the position of a
  # single term out=A[k,j] in a matrix A=DSM[x,y,z], return out.
  # - 'i' has length 4 then i=[x,y,z,k], this is the position of a
  # row out=A[k] (array of length nb) of a sparse matrix corresponding
  # to the dictionary key A=DSM[x,y,z].
  # - If 'i' has length 3 then i is the position of the whole sparse
  # matrix out=A for the sparse matrix at location [x,y,z].
  # .
  # \par
  # The 'k' and 'j' indices can be replaced with : to return rows or
  # columns of the SM A=DSM[x,y,z].
  # @ param x the new value to be assigned to DSM[i]
  # @return the 'i' term of the DSM
  def __setitem__(s,i,x):
    dk,smk=i[:3],i[3:]
    # If dk is a number then one position (x,y,z) is being refered to.
    if isinstance(dk[0],(float,int,np.int64, np.complex128 )): n=1
    else:
        n=len(dk)
    if n==1:
        # set a SM to an input SM
        if len(i)==3:
          # If dk is not already one of the dictionary keys add the new
          # key to the DSM with an initialise SM.
          if dk not in s.d:
            s.d[dk]=SM(s.shape,dtype=np.complex128)
          # Assign 'x' to the SM with key dk.
          s.d[dk]=x
        # set a SM row or multiple row.
        elif len(i)==4:
          # Set one row.
          if isinstance(smk[0],(float,int,np.int64, np.complex128)): n2=1
          # Set multiple rows.
          else:
            n2=len(smk[0])
          # If the key isn't in the DSM add it with the initialised SM.
          if dk not in s.d:
            s.d[dk]=SM(s.shape,dtype=np.complex128)
          # Set a row to the value 'x'
          if n2==1:
            s.d[dk][smk[0],:]=x
          # Set multiple rows to the rows of 'x'.
          else:
            p=0
            for j in smk[0]:
              s.d[dk][smk,:]=x[p]
              p+=1
        # set a SM element or column if smk[0]=: (slice) or multiple elements or columns.
        elif len(i)==5:
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
          # Invalid 'i' the length does not match a possible position.
          errmsg=str('''Error setting the (%s) part of the sparse matr
          ix to (%s). Invalid index (%s). A 3-tuple is required to
          return a sparse matrix(SM), 4-tuple for the row of a SM or
          5-tuple for the element in the SM.''' %(i,x,i))
          raise IndexError(errmsg)
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
  ## String representation of the DSM s.
  # constructs a string of the keys with their corresponding values
  # (sparse matrices with the nonzero positions and values).
  # @return the string of the DSM s.
  def __str__(s):
    keys=s.d.keys()
    out=str()
    for k in keys:
      new= str(k) + str(s.d[k])
      out= (""" {0}
               {1}""").format(out,new)
    return out
  ## Add two DSM's together.
  # Add sparse matrices from DSM and DSM2 elementwise if they have the
  # same dictionary key (x,y,z).
  # Return a new DSM with the same dimensions
  def __add__(s, DSM2):
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      out[x,y,z]=s[x,y,z]+DSM[x,y,z]
    return out
  ## Subtract DSM2 from DSM
  # Subtract sparse matrices in DSM2 from DSM elementwise if they have the
  # same dictionary key (x,y,z).
  # Return a new DSM with the same dimensions
  def __sub__(s, DSM):
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.nx),range(0,s.ny),range(0,s.nz)):
      out[x,y,z]=s[x,y,z]-DSM[x,y,z]
    return out
  ## Multiply DSM2 with s
  # Perform matrix multiplication AB for all sparse matrices A in s
  # and B in DSM2 with the same key (x,y,z)
  # @param DSM2 is a DSM with the same dimensions as s.
  # @return a new DSM with the same dimensions
  def __mul__(s, DSM2):
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      out[x,y,z]=np.multiply(s[x,y,z],DSM2[x,y,z])
    return out
  ## Divide elementwise s with DSM2
  # Perform elementwise division A/B for all sparse matrices A in DSM
  # and B in DSM2 with the same key (x,y,z).
  # @param DSM2 is a DSM with the same dimensions as s.
  # @return a new DSM with the same dimensions
  def __truediv__(s, DSM2):
    out=DS(s.nx,s.ny,s.nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      #ind=DSM[x,y,z].nonzero()
      out=[x,y,z]=np.true_divide(s[x,y,z],DSM2[x,y,z])
      #out[x,y,z][ind]=np.true_divide(s[x,y,z][ind],DSM[x,y,z][ind])
    return out
  ## Finds arcsin(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with arcsin(theta) in the
  # same position as the corresponding theta terms.
  def asin(s):
    """ Finds arcsin(s) for all terms theta \!= 0 in the DS s. Since \
    all angles are in [0,pi/2] arcsin is not a problem.

    :returns: DSM with the same dimensions as s, with arcsin(s)=theta in \
     the same positions as the corresponding theta terms.
    """
    t0=t.time()
    na,nb=s.shape
    asinDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=np.transpose(s.nonzero())
    asinDSM[indices[0],indices[1],indices[2],indices[3],indices[4]]=np.asin(DSM[indices[0],indices[1],indices[2],indices[3],indices[4]])
    return asinDSM
  ## Finds cos(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with cos(theta) in the
  # same position as the corresponding theta terms.
  def cos(s):
    """ Finds cos(theta) for all terms theta \!= 0 in the DS s.

    :returns: A DSM with the same dimensions with cos(theta) in the \
     same position as the corresponding theta terms.
    """
    na,nb=s.shape
    CosDSM=DS(s.nx,s.ny,s.nz,na,nb)
    ind=np.transpose(s.nonzero())
    CosDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.cos(s[ind[0],ind[1],ind[2],ind[3],ind[4]])
    return CosDSM
  ## Finds sin(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with sin(theta) in the
  # same position as the corresponding theta terms.
  def sin(s):
    """ Finds sin(theta) for all terms theta \!= 0 in the DS s.

    :return: A DSM with the same dimensions with sin(theta) in the \
     same position as the corresponding theta terms.
    """
    na,nb=s.shape
    SinDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=np.transpose(s.nonzero())
    SinDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.sin(s[ind[0],ind[1],ind[2],ind[3],ind[4]])
    return SinDSM
  ## Finds the angles theta which are the arguments of the nonzero
  # complex terms in the DSM s.
  # @return a DSM with the same dimensions with theta in the
  # same position as the corresponding complex terms.
  def sparse_angles(s):
    """ Finds the angles theta which are the arguments of the nonzero \
     complex terms in the DSM s.

    :return: A DSM with the same dimensions with theta in the same \
     position as the corresponding complex terms.
    """
    print('start angles')
    t0=t.time()
    na,nb=s.shape
    AngDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    t1=t.time()
    print('before transpose',t1-t0)
    indices=s.nonzero()
    print('indices shape', indices.shape)
    ind=indices.T
    t2=t.time()
    print('before angles',t2-t0)
    AngDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.angle(s[ind[0],ind[1],ind[2],ind[3],ind[4]])
    t3=t.time()-t0
    print('time finding angles',t3)
    return AngDSM
  ## Multiply every column of the DSM s elementwise with the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=vec[k]*DSM[x,y,z,k,j]
  def vec_multiply(s,vec):
    """ Multiply every column of the DSM s elementwise with the
    vector vec.

    :param a: vec: a row vector with length na.

    :rtype: A DSM 'out' with the same dimensions as s.

    :returns: out[x,y,z,k,j]=vec[k]*DSM[x,y,z,k,j]
     """
    na,nb=s.shape
    outDSM=DS(s.nx,s.ny,s.nz,na,nb)
    ind=s.nonzero()
    Ni=len(np.transpose(ind)[4]) #FIXME find the nonzero columns without repeat column index for each term
    for l in range(0,Ni):
      #out=np.multiply(vec,s[ind[l][3]],s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]])
      out=np.multiply(vec[ind[l][3]],s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]])
      outDSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=out
    return outDSM
  ## Divide every column of the DSM s elementwise with the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=DSM[x,y,z,k,j]/vec[k]
  def dict_DSM_divideby_vec(s,vec):
    """ Divide every column of the DSM s elementwise with the vector vec.

    :param vec: a row vector with length na.

    :rtype: a DSM 'out' with the same dimensions as s.

    :return: out[x,y,z,k,j]=DSM[x,y,z,k,j]/vec[k] """
    na,nb=s.shape
    outDSM=DS(s.nx,s.ny,s.nz,na,nb)
    #ind=np.transpose(vec.nonzero())
    for x,y,z,a,b in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz),indices,range(0,nb)):
      a=a[0]
      if abs(s[x,y,z,a,b])<epsilon:
        pass
      else:
        out=np.divide(s[x,y,z,a,b],vec[a])
        outDSM[x,y,z,a,b]=out
    return outDSM
  ## Every column of the DSM s divides elementwise the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=vec[k]/DSM[x,y,z,k,j]
  def dict_vec_divideby_DSM(s,vec):
    """ Every column of the DSM s divides elementwise the vector vec.

    :param vec: a row vector with length na.

    :rtype: a DSM 'out' with the same dimensions as s.

    :return: out[x,y,z,k,j]=vec[k]/DSM[x,y,z,k,j]
    """
    na,nb=s.shape
    outDSM=DS(s.nx,s.ny,s.nz,na,nb)
    indices=s.nonzero()
    Ni=len(indices[0])
    for l in range(0,Ni):
      out=np.divide(vec[indices[l][3]],s[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]])
      outDSM[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]]=out
    return outDSM
  ## Save the DSM s.
  # @param filename_ the name of the file to save to.
  # @return nothing
  def save_dict(s, filename_):
    """ Save the DSM s.

    :param filename_: the name of the file to save to.

    :return: nothing
    """
    with open(filename_, 'wb') as f:
        pkl.dump(s.d, f)
    return
  ## Find the indices of the nonzero terms in the DSM s.
  # The indices are found by iterating through all keys (x,y,z) for the
  # DSM s and finding the nonzero indices of the corresponding sparse
  # matrix. These indices are then combinded with the x,y,z key and
  # stacked to create an 5xN array of all the nonzero terms in the DSM,
  # where N is the number of nonzero terms.
  # @return indices=[ [x1,y1,z1,k1,j1],...,[xn,yn,zn,kn,jn]]
  def nonzero(s):
    """ Find the indices of the nonzero terms in the DSM s.

      .. note::

          The indices are found by iterating through all keys (x,y,z) \
          for the DSM s and finding the nonzero indices of the \
          corresponding sparse matrix. These indices are then combinded \
          with the x,y,z key and stacked to create an 5xN array of all \
          the nonzero terms in the DSM, where N is the number of nonzero \
          terms.

    :return: indices=[ [x1,y1,z1,k1,j1],...,[xn,yn,zn,kn,jn]]
    """
    # FIXME this is too slow and needs parallelising / speeding up.
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      indicesM=s.d[x,y,z].nonzero()
      NI=len(indicesM[0])
      if x==0 and y==0 and z==0:
        indices=np.array([0,0,0,indicesM[0][0],indicesM[1][0]])
        indicesSec=np.c_[np.tile(np.array([x,y,z]),(NI-1,1)),indicesM[0][1:],indicesM[1][1:]]
        indices=np.vstack((indices,indicesSec))
      else:
        indicesSec=np.c_[np.tile(np.array([x,y,z]),(NI,1)),indicesM[0][0:],indicesM[1][0:]]
        indices=np.vstack((indices,indicesSec))
    return indices
  ## Find the indices of the nonzero terms for part of the DSM s.
  # @param s the part of s that you want the nonzero indices for.
  # .
  # \par
  # The indices are found by using the nonzero() function on s[cor]
  # @return indices=[ [x1,y1,z1,k1,j1],...,[xn,yn,zn,kn,jn]]
  def nonzeroMat(s,cor):
    """ Find the indices of the nonzero terms for part of the DSM s.

    :param cor: the part of s that you want the nonzero indices for.

    The indices are found by using the :func: nonzero() function on s[cor]

    :return: indices=[ [x1,y1,z1,k1,j1],...,[xn,yn,zn,kn,jn]]
    """
    if isinstance(cor,(float,int,np.int64, np.complex128)):
      ns=1
    else:
      ns=len(cor)
    for j in range(0,ns):
      indM=s[cor[0],cor[1],cor[2]].nonzero()
      NI=len(indM[0])
      if j==0:
        ind=np.r_[cor,indM[0][0],indM[1][0]]
        ind=np.c_[np.tile(cor,(NI-1,1)),indM[0][1:],indM[1][1:]]
      else:
        indt=np.c_[np.tile(cor,(NI,1)),indM[0][0:],indM[1][0:]]
        ind=np.vstack((ind,indt))
    return ind
  ## Fills the DSM s.
  # @return a dense Nx*Ny*Nz*na*nb array with matching nonzero terms to
  # the sparse matrix s and zeroes elsewhere.
  def dense(s):
    """Fills the DSM s.

    :returns: A dense Nx*Ny*Nz*na*nb array with matching nonzero terms to \
     the sparse matrix s and zeroes elsewhere.
    """
    (na,nb)=s.d[0,0,0].shape
    nx=s.nx
    ny=s.ny
    nz=s.nz
    den=np.zeros((nx,ny,nz,na,nb),dtype=np.complex128)
    for x,y,z in product(range(s.nx),range(s.ny),range(s.nz)):
      den[x,y,z]=s.d[x,y,z].todense()
    return
  ## Check if the index [i,j,k] is valid.
  # @param i is the index for the x axis.
  # @param j is the index for the y axis.
  # @param k is the index for the z axis.
  # @param p1 is the point at the end of the ray.
  # @param h is the mesh width
  # @return 0 if valid, 1 if not.
  def stopcheck(s,i,j,k,p1,h):
    """ Check if the index [i,j,k] is valid.

    :param i: is the index for the x axis.

    :param j: is the index for the y axis.

    :param k: is the index for the z axis.

    :param p1: is the point at the end of the ray.

    :param h: is the mesh width

    :return: 0 if valid, 1 if not.

    .. todo:: add the inside check to this function

    .. todo:: add the check for the end of the ray.
    """
    #FIXME add the inside check to this function
    #FIXME add the check for the end of the ray.
    #if i>=p1[0] and j>=p1[1] and k>=p1[2]:
    #  return 0
    if i>s.Nx or j>s.Ny or k>s.Nz or i<0 or j<0 or k<0:
      return 0
    else: return 1
  ## Check if the list of points is valid.
  # @param ps the indices for the points in the list
  # @param p1 the end of the ray
  # @param h the meshwidth
  # @param p3 the points on the cone vectors
  # @param the normal vectors forming the cone.
  # @return start=0 if no points were valid if at least 1 point was
  # valid, ps=[[i1,j1,k1],...,[in,jn,kn]] the indices of the
  # valid points, p3=[[x1,y1,z1],...,[xn,yn,zn]] co-ordinates of
  # the valid points., N=[n0,...,Nn] the normal vectors corresponding to
  # the valid points.
  def stopchecklist(s,ps,p1,h,p3,n):
    """ Check if the list of points is valid.

    :param ps: the indices for the points in the list

    :param p1: the end of the ray

    :param h: the meshwidth

    :param p3: the points on the cone vectors

    :param n: the normal vectors forming the cone.

    start=0 if no points were valid if at least 1 point was valid,
    ps=[[i1,j1,k1],...,[in,jn,kn]] the indices of the valid points,
    p3=[[x1,y1,z1],...,[xn,yn,zn]] co-ordinates of the valid points,
    N=[n0,...,Nn] the normal vectors corresponding to the valid points.

    :return: start, ps, p3, N
    """
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
  ctht=DS(Mesh.nx,Mesh.ny,Mesh.nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta_t) #FIXME
  print('-------------------------------')
  print('Computing cos(theta_i) on all reflection terms')
  print('-------------------------------')
  cthi=AngDSM.cos()                                   # Compute cos(theta_i)
  print('-------------------------------')
  print('Computing cos(theta_t) on all reflection terms')
  print('-------------------------------')
  SIN[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.sin(AngDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]) # Compute sin(theta)
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



def parnonzero():
  Nob=3
  Nre=3
  Nra=5
  n=10
  nj=4
  DS=test_03(n,n,n,int(Nob*Nre+1),int((Nre)*(Nra)+1))
  #p = Process(target=nonzeroMat, args=(cor,DS))
  x=np.arange(0,DS.nx,1)
  y=np.arange(0,DS.ny,1)
  z=np.arange(0,DS.nz,1)
  coords=np.transpose(np.meshgrid(x,y,z))
  with Pool(processes=nj) as pool:         # start 4 worker processes
    ind=pool.map(DS.nonzeroMat, product(range(DS.nx),range(DS.ny),range(DS.nz)))      # prints "[0, 1, 4,..., 81]"
    #it = pool.imap(f, range(10))
    #print(next(it))                     # prints "0"
    #print(next(it))                     # prints "1"
    #print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow
    #result = pool.apply_async(time.sleep, (10,))
    #print(result.get(timeout=1))
  #p.start()
  #p.join
  #FIXME
  print(ind)
  return 0

def test_00():
  ds=DS()
  ds[1,2,3,0,0]=2+3j
  print(ds[1,2,3][0,0])

## Test creation of dictionary containing sparse matrices
def test_01(nx=3,ny=2,nz=1,na=5,nb=6):
  ds=DS(nx,ny,nz,na,nb)

##  test creation of matrix and adding on element
def test_02(nx=7,ny=6,nz=1,na=5,nb=6):
  ds=DS(nx,ny,nz,na,nb)
  ds[0,3,0,:,0]=2+3j
  print(ds[0,3,0])

## Test creation of diagonal sparse matrices contained in every position
def test_03(nx,ny,nz,na,nb):
  ds=DS(nx,ny,nz,na,nb)
  for x,y,z,a in product(range(nx),range(ny),range(nz),range(na)):
    if a<nb:
      ds[x,y,z,a-1,a-1]=complex(a,a)
  return ds

## Test creation of first column sparse matrices contained in every position
def test_03b(nx,ny,nz,na,nb):
  ds=DS(nx,ny,nz,na,nb)
  for x,y,z,a in product(range(nx),range(ny),range(nz),range(na)):
    if a<nb:
      ds[x,y,z,a-1,2]=complex(a,a)
  return ds

## Test creation of lower triangular sparse matrices contained in every position
def test_03c(nx,ny,nz,na,nb):
  ds=DS(nx,ny,nz,na,nb)
  for x,y,z,a in product(range(nx),range(ny),range(nz),range(na)):
    if a<nb:
      for ai in range(a-1,na):
        ds[x,y,z,ai,a-1]=complex(a,a)
  return ds

# Test matrix addition operation
def test_04():
  ds=test_03(7,6,1,5,6)
  M=ds[2,0,0]+ds[0,1,0]
  ds[0,0,0]=M
  print(ds[0,0,0])
  return

##  Test get column
def test_05():
  ds=test_03b(7,6,1,6,5)
  print(ds[0,0,0,:,2])
  print(ds[0,0,0,0,:])
  print(ds[0,0,0,1:6:2,:])
  return

## Test matrix multiplication
def test_06():
  ds1=test_03(7,6,1,5,5)
  ds2=test_03b(7,6,1,5,5)
  M0=ds1[0,0,0]*ds2[0,0,0]
  M1=ds1[5,5,0]*ds2[6,5,0]
  print(M0)
  print(M1)
  return

## Test getting angle from complex entries in matrix
def test_07():
  ds=test_03(3,3,1,3,3)
  M0=ds[0,0,0]
  indices=zip(*M0.nonzero())
  M1= SM(M0.shape,dtype=np.float)
  for i,j in indices:
    M1[i,j]=np.angle(M0[i,j])
  print(M0,M1)
  return

## Test getting angle from complex entries in matrix then taking the cosine of every nonzero entry
def test_08():
  ds=test_03(3,3,1,3,3)
  M0=ds[0,0,0]
  indices=zip(*M0.nonzero())
  M1= SM(M0.shape,dtype=np.float)
  for i,j in indices:
    M1[i,j]=np.cos(np.angle(M0[i,j]))
  print(M0,M1)
  return

## test operation close to Fresnel reflection formula
# On the [0,0,0] matrix in the DS
# N1=Z1*cos(thetai)-Z2*cos(thetat)
# N2=Z1*cos(thetai)+Z2*cos(thetat)
# \todo N1/N2
def test_09():
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
  return (N1) # next step [N1.nonzero()]) #/(N2[N2.nonzero()]))

##  Multiply by coefficient and sum the nonzero terms in the columns
# On the [0,0,0] matrix of the DS
def test_10():
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

## Extract reflection angles from DS
def test_11():
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

## Extract the cos of the reflection angles of the DS
def test_12():
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

## Attempt to find angle of nonzero element of SM inside dictionary
def test_13():
  nx=2
  ny=2
  nz=1
  na=3
  nb=3
  DSM=test_03c(nx,ny,nz,na,nb)
  ang=dict_sparse_angles(DSM)
  return 1

## Attempt to compute Reflection Coefficents on DS
def test_14():
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

## Timing nonzero indexing
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
  mp.plot(timevec,narray)
  mp.title('Time against n for nonzero() function')
  mp.savefig('timenonzero.png')
  return timevec

## Attempting to parallelise nonzero function
def test_17():
  out=parnonzero()
  return

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  #job_server = pp.Server()
  test_14()

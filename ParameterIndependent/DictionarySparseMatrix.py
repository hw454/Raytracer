#!/usr/bin/env python3
# Hayley 2019-05-01
'''Code for the dictionary of sparse matrices class :py:class:`DS` which\
 indexes like a multidimensional array but the array is sparse. \
 To exploit :py:mod:`scipy.sparse.dok_matrix`=SM the `DS` uses a key for \
 each x,y, z position and associates a SM.

 This module also contains functions which are not part of the class \
 but act on it.

 '''


import numpy as np
from scipy.sparse import dok_matrix as SM
import scipy.sparse.linalg
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
  j is in [0, Ny], and k is in [0,Nz].
  SM=DS[x,y,z] is a na*nb sparse matrix, initialised with complex128 data type.
  :math:`na=(Nob*Nre+1)`
  :math:`nb=((Nre)*(Nra)+1)`
  '''  ## The constructor.
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
  def __init__(s,Nx=1,Ny=1,Nz=1,na=1,nb=1,dt=np.complex128):
    s.shape=(na,nb)
    Keys=product(range(Nx),range(Ny),range(Nz))
    default_value=SM(s.shape,dtype=dt)
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
    ''' Get item i from s

    :param i:
     * If 'i' has length 5, :math:`i=[x,y,z,k,j]` then this is the \
     position of a single term out=A[k,j] in a matrix \
     :math:`A=DSM[x,y,z]`, return out.
     * 'i' has length 4 then :math:`i=[x,y,z,k]`, this is the \
     position of a row out=A[k] (array of length nb) of a sparse \
      matrix corresponding to the dictionary key A=DSM[x,y,z].
     * If 'i' has length 3 then i is the position of the whole sparse
     matrix out=A for the sparse matrix at location [x,y,z].

    The 'k' and 'j' indices can be replaced with : to return rows or \
    columns of the SM A=DSM[x,y,z].

    :return: the 'i' term of the DSM

    '''
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
    ''' Set a new value to all or part of a DSM.

    :param i:

      * If 'i' has length 5, i=[x,y,z,k,j] then this is the position of \
      a single term out=A[k,j] in a matrix A=DSM[x,y,z], return out.

      * 'i' has length 4 then i=[x,y,z,k], this is the position of a \
      row out=A[k] (array of length nb) of a sparse matrix corresponding \
      to the dictionary key A=DSM[x,y,z].

      * If 'i' has length 3 then i is the position of the whole sparse \
      matrix out=A for the sparse matrix at location [x,y,z].

      The 'k' and 'j' indices can be replaced with : to return rows or \
      columns of the SM A=DSM[x,y,z].

    Method:

    * .. code::

         dk,smk=i[:3],i[3:]

    * If dk[0],dk[1], and dk[2] are all numbers and smk[0] and smk[1] \
      are not numbers of slices then :math:`k=1`. This means that a \
      whole sparse matrix is being set at the (x,y,z) position.

    * k indicates which (x,y,z) terms are being set.

      * If k==-1: Set only one grid position (x,y,z).
      * If k==0. Set all x positions with (y,z) co-ordinates.
      * If k==1. Set all y positions with (x,z) co-ordinates.
      * If k==2. Set all z positions with (x,y) co-ordinates.
      * If k==3. Set all x and y positions with (z) co-ordinates.
      * If k==4. Set all x and z  positions with (y) co-ordinates.
      * If k==5. Set all y and z  positions with (x) co-ordinates
      * If k==6. Set all x, y and z positions.

    * n indicates whether a whole SM is set, a row or a column.

      * If n==0 a whole SM.
      * If n==1 a row or rows.
        * n2 is the number of rows.
      * If n==2 a column or columns.
        * n2 is the number of columns.

    :param x: the new value to be assigned to DSM[i]

    :return: the 'i' term of the DSM

    '''
    dk,smk=i[:3],i[3:]
    # If dk[0], dk[1], and dk[2] are numbers then one position (x,y,z)
    # is being refered to.
    if isinstance(dk[0],(float,int,np.int64, np.complex128 )):
      if isinstance(dk[1],(float,int,np.int64, np.complex128 )):
        if isinstance(dk[2],(float,int,np.int64, np.complex128 )):
          k=-1
        else:
          k=2
      else:
        if isinstance(dk[2],(float,int,np.int64, np.complex128 )):
          k=1
        else:
          k==3
    else:
      if isinstance(dk[1],(float,int,np.int64, np.complex128 )):
        if isinstance(dk[2],(float,int,np.int64, np.complex128 )):
          k=0
        else:
          k=4
      else:
        if isinstance(dk[2],(float,int,np.int64, np.complex128 )):
          k=5
        else:
          k==6
    n=len(i)-3
    if k==-1:
        if n==0:
          # If dk is not already one of the dictionary keys add the new
          # key to the DSM with an initialise SM.
          if dk not in s.d:
            s.d[dk]=SM(s.shape,dtype=np.complex128)
          # Assign 'x' to the SM with key dk.
          s.d[dk]=x
        elif n==1:
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
            s.d[dk][smk,:]=x
          # Set multiple rows to the rows of 'x'.
          else:
            p=0
            for j in smk:
              s.d[dk][j,:]=x[p]
              p+=1
        # set a SM element or column if smk[0]=: (slice) or multiple elements or columns.
        elif n==2:
          if isinstance(smk[0],(float,int,np.int64, np.complex128,slice)): n2=1
          else:
            n2=len(smk[0])
          if dk not in s.d:
            s.d[dk]=SM(s.shape,dtype=np.complex128)
          if n2==1:
            s.d[dk][smk[0],smk[1]]=x
            SMtemp=s.d[dk]
            SMtemp[smk[0],smk[1]]=x
            s.d[dk]=SMtemp
            #DEBUG All terms in the dictionary point to the same value.
            # When reassigning the value this is reassigning the term
            # they all point to. Not sure how to correct this.
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
    else: #FIXME what case is this for?
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
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      out[x,y,z]=s[x,y,z]+DSM2[x,y,z]
    return out
  ## Subtract DSM2 from DSM
  # Subtract sparse matrices in DSM2 from DSM elementwise if they have the
  # same dictionary key (x,y,z).
  # Return a new DSM with the same dimensions
  def __sub__(s, DSM):
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      out[x,y,z]=s[x,y,z]-DSM[x,y,z]
    return out
  ## Multiply DSM2 with s
  # Perform matrix multiplication AB for all sparse matrices A in s
  # and B in DSM2 with the same key (x,y,z)
  # @param DSM2 is a DSM with the same dimensions as s.
  # @return a new DSM with the same dimensions
  def __mul__(s, DSM2):
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      out[x,y,z]=s[x,y,z].multiply(DSM2[x,y,z])
    return out
  ## Divide elementwise s with DSM2
  # Perform elementwise division A/B for all sparse matrices A in DSM
  # and B in DSM2 with the same key (x,y,z).
  # @param DSM2 is a DSM with the same dimensions as s.
  # @return a new DSM with the same dimensions
  def __truediv__(s, DSM2):
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      ind=DSM2[x,y,z].nonzero()
      #out=[x,y,z]=np.true_divide(s[x,y,z],DSM2[x,y,z])
      n=len(ind[0])
      for i in range(0,n):
        out[x,y,z][ind[0][i],ind[1][i]]=np.true_divide(s[x,y,z][ind[0][i],ind[1][i]],DSM2[x,y,z][ind[0][i],ind[1][i]])
    return out
  def __eq__(s,DSM2):
    ''' If all non-zero terms aren't equal then matrices aren't equal.
    :return: True if equal, False if not.
    '''
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      if scipy.sparse.linalg.norm(s[x,y,z]-DSM2[x,y,z],1) != 0:
        if scipy.sparse.linalg.norm(s[x,y,z]-DSM2[x,y,z],1)>epsilon:
          return False
        else: pass
      else: pass
    return True
  def __self_eq__(s):
    ''' Check that the SMs at each grid point aren't equal.

    Method:
    Check each grid point against a different one. If o
    :return: True if equal, False if not.
    '''
    for x1,y1,z1 in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      for x2,y2,z2 in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
        if x1==x1 and y1==y2 and z1==z2: pass
        else:
          ind=s[x1,y1,z1].nonzero()
          n=len(ind[0])
          for i in range(n):
            if s[x1,y1,z1][ind[0][n-1],ind[1][n-1]]!=s[x2,y2,z2][ind[0][n-1],ind[1][n-1]]: return False
            else: pass
    return True
  ## Return the distance which is stored in each column of s
  # @rtype a new DS with sparse vectors rather than matrices in each grid element
  # @return DS(Nx,Ny,Nz,1,nb)
  def __get_rad__(s):
    ''' Return a DS corresponding to the distances stored in the mesh.

      * Initialise out=DS(Nx,Ny,Nz,1,s.shape[1])

      * Go through all the nonzero x,y,z grid points.

      * Go through the nonzero columns and put the absolute value of the \
      first term in the corresponding column in out[x,y,z]

      * Pass until the next nonzero index is for a new column and repeat.

    :rtype: DS(Nx,Ny,Nz,1,s.shape[1]) of real values.

    :return: out

    '''
    out=DS(s.Nx,s.Ny,s.Nz,1,s.shape[1],float)
    # Find the nonzero indices. There's no need to retrieve distances on 0 terms.
    ind=s.nonzero()
    ind=ind.transpose()
    n=len(ind[0])
    p=-1
    l=-1
    m=-1
    for i in range(0,n):
      # No need to go through the same grid element again.
      if ind[0][i]==p and ind[1][i]==l and ind[2][i]==m:
        pass
      else:
        ind2=nonzero_bycol(s[ind[0][i],ind[1][i],ind[2][i]])
        p=ind[0][i]
        l=ind[1][i]
        m=ind[2][i]
        count=0
        chk=-1
        for j in ind2[1]:
          if chk==j:
            pass
          else:
            out[p,l,m,0,j]=abs(s[p,l,m,ind2[0][count],j])
            chk=j
          count+=1
    return out
  ## Finds arcsin(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with arcsin(theta) in the
  # same position as the corresponding theta terms.
  def togrid(s):
    ''' Computethe matrix norm at each grid point and return a \
    3d numpy array.

    :rtype: Nx x Ny x Nz numpy array

    :return: Grid

    '''
    Nx=s.Nx
    Ny=s.Ny
    Nz=s.Nz
    na=s.shape[0]
    nb=s.shape[1]
    Grid=np.zeros((Nx,Ny,Nz),dtype=float)
    for x, y, z in product(range(0,Nx), range(0, Ny), range(0,Nz)):
      Grid[x,y,z]=scipy.sparse.linalg.norm(s[x,y,z],1)
    return Grid

  def asin(s):
    """ Finds

    :math:`\\theta=\\arcsin(x)` for all terms :math:`x != 0` in \
    the DS s. Since all angles \
    :math:`\\theta` are in :math:`[0,\pi /2]`, \
    :math:`\\arcsin(x)` is not a problem.

    :returns: DSM with the same dimensions as s, with \
    :math:`\\arcsin(s)=\\theta` in \
     the same positions as the corresponding theta terms.

    """
    t0=t.time()
    na,nb=s.shape
    asinDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    ind=np.transpose(s.nonzero())
    asinDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.arcsin(s[ind[0],ind[1],ind[2],ind[3],ind[4]])
    return asinDSM
  ## Finds cos(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with cos(theta) in the
  # same position as the corresponding theta terms.
  def cos(s):
    """ Finds :math:`\\cos(\\theta)` for all terms \
    :math:`\\theta != 0` in the DS s.

    :returns: A DSM with the same dimensions with \
    :math:`\\cos(\\theta)` in the \
     same position as the corresponding theta terms.

    """
    na,nb=s.shape
    CosDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    ind=np.transpose(s.nonzero())
    CosDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.cos(s[ind[0],ind[1],ind[2],ind[3],ind[4]])
    return CosDSM
  ## Finds sin(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with sin(theta) in the
  # same position as the corresponding theta terms.
  def sin(s):
    """ Finds :math:`\\sin(\\theta)` for all terms \
    :math:`\\theta != 0` in the DS s.

    :return: A DSM with the same dimensions with \
    :math:`\\sin(\\theta)` in the \
     same position as the corresponding theta terms.

    """
    na,nb=s.shape
    SinDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    ind=np.transpose(s.nonzero())
    SinDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.sin(s[ind[0],ind[1],ind[2],ind[3],ind[4]])
    return SinDSM
  ## Finds the angles theta which are the arguments of the nonzero
  # complex terms in the DSM s.
  # @return a DSM with the same dimensions with theta in the
  # same position as the corresponding complex terms.
  def sparse_angles(s):
    """ Finds the angles :math:`\\theta` which are the arguments \
    of the nonzero complex terms in the DSM s.

    :return: A DSM with the same dimensions with \
    :math:`\\theta` in the same \
     position as the corresponding complex terms.

    """
    print('start angles')
    t0=t.time()
    na,nb=s.shape
    AngDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    t1=t.time()
    indices=s.nonzero()
    ind=indices.T
    AngDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.angle(s[ind[0],ind[1],ind[2],ind[3],ind[4]])
    t3=t.time()-t0
    print('time finding angles',t3)
    return AngDSM
  ## Multiply every column of the DSM s elementwise with the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=vec[k]*DSM[x,y,z,k,j]
  def dict_scal_mult(s,scal):
    '''Multiply every term of the DSM s by scal.

    :param scal: scalar variable

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out[x,y,z,k,j]=scal*DSM[x,y,z,k,j]

    :rtype: DS(Nx,Ny,Nz,na,nb)

    :return: out

    '''
    out=DS(s.Nx,s.Ny,s.Nz,1,s.shape[1])
    ind=s.nonzero()
    ind=ind.transpose()
    n=len(ind[0])
    p=-1
    l=-1
    m=-1
    for i in range(0,n):
      # No need to go through the same grid element again.
      if ind[0][i]==p and ind[1][i]==l and ind[2][i]==m:
        pass
      else:
        p=ind[0][i]
        l=ind[1][i]
        m=ind[2][i]
        out[p,l,m]=scal*s[p,l,m]
    return out

  def dict_vec_multiply(s,vec):
    """ Multiply every column of the DSM s elementwise with the \
    vector vec.

    :param vec: a row vector with length na.

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out[x,y,z,k,j]=vec[k]*DSM[x,y,z,k,j]

    Multiplication is done using \
    :py:class:`DS`. :py:func:`dict_vec_multiply(vec)`

    :rtype: A DSM 'out' with the same dimensions as s.

    :returns: out

     """
    na,nb=s.shape
    outDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
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

  def dict_row_vec_multiply(s,vec):
    """ Multiply every row of the DSM s elementwise with the
    vector vec.

    :param vec: a row vector with length na.

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out[x,y,z,k,j]=vec[j]*DSM[x,y,z,k,j]

    Multiplication is done using \
    :py:class:`DS`. :py:func:`dict_vec_multiply(vec)`

    :rtype: A DSM 'out' with the same dimensions as s.

    :returns: out

     """
    na,nb=s.shape
    outDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    ind=s.nonzero()
    Ni=len(np.transpose(ind)[4]) #FIXME find the nonzero columns without repeat column index for each term
    for l in range(0,Ni):
      #out=np.multiply(vec,s[ind[l][3]],s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]])
      out=np.multiply(vec[ind[l][4]],s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]])
      outDSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=out
    return outDSM
  ## Divide every column of the DSM s elementwise with the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=DSM[x,y,z,k,j]/vec[k]

  def dict_DSM_divideby_vec(s,vec):
    """ Divide every column of the DSM s elementwise with the vector vec.

    :param vec: a row vector with length na.

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out[x,y,z,k,j]=DSM[x,y,z,k,j]/vec[k]

    :rtype: a DSM 'out' with the same dimensions as s.

    :return:  out

    """
    na,nb=s.shape
    outDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    ind=np.transpose(vec.nonzero())
    for x,y,z,a,b in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz),ind,range(0,nb)):
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
  def dict_col_mult(s):
    ''' Multiply all nonzero terms in a column.

    In every grid point x,y,z of s there is a sparse matrix SM. \
    Take the product of all nonzero terms in each column and \
    keep these in a vector v.
    Construct a new DS of size Nx x Ny x Nz x 1 x nb=s.shape[1].\
    Call this out. out[x,y,z] should be the v corresponding \
    to the SM in s at x,y,z.

    Method:
      * Find the :py:class:`DS`. :py:func:`nonzero()` indices of s.
      * For each nonzero x,y,z grid point find the \
      nonzero() indices of the SM. Do this by column so \
      that the output has pairs going through \
      each nonzero column and matching the nonzero row \
      number. Use function :py:func:`nonzero_bycol()`.
      * Go through each of these indice pairs for the SM. \
      Check if the column index is new. If so assign \
      the column in out to the matching \
      value in the SM. If the column number is not \
      new then multiply the value in the column in \
      out by the corresponding value in the SM.

    .. code::

       out=[
       [prod(nonzero terms in column 0 in s[0,0,0]),
       prod(nonzero terms in column 1 in s[0,0,0]),
       ...,
       prod(nonzero terms in column nb in s[0,0,0]
       ],
       ...,
       [prod(nonzero terms in column 0 in s[Nx-1,Ny-1,Nz-1]),
       prod(nonzero terms in column 1 in s[Nx-1,Ny-1,Nz-1]),
       ...,
       prod(nonzero terms in column nb in s[Nx-1,Ny-1,Nz-1]
       ]
       ]

    :rtype: DS of size Nx x Ny x Nz x 1 x nb

    :return: out

    '''
    out=DS(s.Nx,s.Ny,s.Nz,1,s.shape[1])
    ind=s.nonzero()
    ind=ind.transpose()
    n=len(ind[0])
    p=-1
    l=-1
    m=-1
    for i in range(0,n):
      # No need to go through the same grid element again.
      if ind[0][i]==p and ind[1][i]==l and ind[2][i]==m:
        pass
      else:
        p=ind[0][i]
        l=ind[1][i]
        m=ind[2][i]
        ind2=nonzero_bycol(s[p,l,m])
        count=0
        chk=-1
        for j in ind2[1]:
          if chk==j:
            col=col*s[p,l,m,ind2[0][count],j]
            out[p,l,m,0,ind2[1][count]]=col
          else:
            col=s[p,l,m,ind2[0][count],j]
            out[p,l,m,0,ind2[1][count]]=col
            chk=j
          count+=1
    return out
  def dict_vec_divideby_DSM(s,vec):
    """ Every column of the DSM s divides elementwise the vector vec.

    :param vec: a row vector with length na.

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out[x,y,z,k,j]=vec[k]/DSM[x,y,z,k,j]

    :rtype: a DSM 'out' with the same dimensions as s.

    :return: out

    """
    na,nb=s.shape
    outDSM=DS(s.Nx,s.ny,s.nz,na,nb)
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

          The indices are found by iterating through all \
          keys (x,y,z) for the DSM s and finding the nonzero \
          indices of the corresponding sparse matrix. \
          These indices are then combinded \
          with the x,y,z key and stacked to create a 5xN \
          array of all the nonzero terms in the DSM, \
          where N is the number of nonzero \
          terms.

    :return: indices=[ [x1,y1,z1,k1,j1],...,[xn,yn,zn,kn,jn]]

    """
    # FIXME this is too slow and needs parallelising / speeding up.
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      indicesM=s.d[x,y,z].nonzero()
      NI=len(indicesM[0])
      if abs(NI)<epsilon:
        pass
      else:
        if x==0 and y==0 and z==0:
          indices=np.array([0,0,0,indicesM[0][0],indicesM[1][0]])
          indicesSec=np.c_[np.tile(np.array([x,y,z]),(NI-1,1)),indicesM[0][1:],indicesM[1][1:]]
          indices=np.vstack((indices,indicesSec))
        else:
          indicesSec=np.c_[np.tile(np.array([x,y,z]),(NI,1)),indicesM[0][0:],indicesM[1][0:]]
          indices=np.vstack((indices,indicesSec))
    return indices
  def row_sum(s):
    ''' Sum all nonzero terms in a row.

    In every grid point x,y,z of s there is a sparse matrix SM.
    Construct a new DS of size Nx x Ny x Nz x na=s.shape[0] x 1.
    Call this out.
    out[x,y,z] should be the corresponding na x1 SM to the SM in s at x,y,z.

    Method:
      * Find the :py:class:`DS`. :py:func:`nonzero()` indices of s`
      * Go through each of these indice. Check if the \
      row index is new. If so assign the row in out to the matching \
      value in the SM. If the row number is not new then sum the \
      value in the column in out by the corresponding value in the SM.

    .. code::

       out=[
       [sum(nonzero terms in row 0 in s[0,0,0]),
       sum(nonzero terms in row 1 in s[0,0,0]),
       ...,
       sum(nonzero terms in row na in s[0,0,0]
       ],
       ...,
       [sum(nonzero terms in row 0 in s[Nx-1,Ny-1,Nz-1]),
       sum(nonzero terms in row 1 in s[Nx-1,Ny-1,Nz-1]),
       ...,
       sum(nonzero terms in row na in s[Nx-1,Ny-1,Nz-1]
       ]
       ]

    :rtype: DS of size Nx x Ny x Nz x na x  1

    :return: out

    '''
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],1)
    ind=s.nonzero()
    ind=ind.transpose()
    n=len(ind[0])
    for i in range(0,n):
      out[ind[0][i],ind[1][i],ind[2][i],ind[3][i],0]+=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
    return out
  ## Find the indices of the nonzero terms for part of the DSM s.
  # @param s the part of s that you want the nonzero indices for.
  # .
  # \par
  # The indices are found by using the nonzero() function on s[cor]
  # @return indices=[ [x1,y1,z1,k1,j1],...,[xn,yn,zn,kn,jn]]
  def nonzeroMat(s,cor):
    """ Find the indices of the nonzero terms for part of the DSM s.

    :param cor: the part of s that you want the nonzero indices for.

    The indices are found by using the :py:func:`nonzero()` function on s[cor]

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

    :returns: A dense Nx x Ny x Nz x na x nb array with matching nonzero terms to \
     the sparse matrix s and zeroes elsewhere.

    """
    (na,nb)=s.d[0,0,0].shape
    Nx=s.Nx
    Ny=s.Ny
    Nz=s.Nz
    den=np.zeros((Nx,Ny,Nz,na,nb),dtype=np.complex128)
    for x,y,z in product(range(s.Nx),range(s.Ny),range(s.Nz)):
      den[x,y,z]=s.d[x,y,z].todense()
    return den
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

    :return: 1 if valid, 0 if not.

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
    N=[n0,...,nN] the normal vectors corresponding to the valid points.

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

def phase_calc(RadMesh,khat,L):
  ''' Compute :math:`\\exp(i\frac{\hat{k}\hat{r}}{L^2})` \
  for a Mesh of :math:`r`

  The phase is usually expressed at :math:`exp(ikr)`.
  Since :math:`\hat{k}` and :math:`\hat{r}` are nondimensional lengths \
  scaled by the room length L the power of :math:`L^{-2}` must be used.

  Exponentials are not defined on DS, instead use \
  :math:`\\exp(i \theta)=\\cos(\theta)+i\\sin(\theta)`.

  .. code::

     S1=RadMesh.dict_scal_mult(khat)
     S2=S1.dict_scal_mult(1.0/(L**2))
     out=S1.cos()+S1.sin().dict_scal_mult(1j)

  :rtype: DS of size Nx x Ny x Nz x na x 1

  :return: out

  '''

  S1=RadMesh.dict_scal_mult(khat)
  S2=S1.dict_scal_mult(1.0/(L**2))
  out=S1.cos()+S1.sin().dict_scal_mult(1j)
  return out

def power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt):
  ''' Compute the field from a Mesh of ray information and the physical \
  parameters.

  :param Mesh: The :py:class:`DS` mesh of ray information.
  :param Znobrat: An Nob x Nre+1 array containing tiles of the impedance \
    of obstacles divided by the impedance of air.
  :param refindex: An Nob x Nre+1 array containing tiles of the refractive\
    index of obstacles.

  Method:

    * First compute the reflection coefficients using \
    :py:func:`ref_coef(Mesh,Znobrat,refindex)`
    * Combine the reflection coefficients that correspond to the same \
    ray using :py:class:`DS`. :py:func:`dict_col_mult()`. This \
    multiplies reflection coefficients in the same column.
    * Extract the distance each ray had travelled using \
    :py:class:`DS`. :py:func:`__get_rad__()`
    * Multiply by the gains for the corresponding ray.
    * Multiply terms by the phases \
    :math:`\\exp(i\hat{k} \hat{r})^{L^{-2}}`. With :math:`L` being \
    the room length scale. :math:`\hat{r}` being the relative distance \
    travelled which is the actual distance divided by the room length \
    scale, and :math:`\hat{k}` is the relative wavenumber which is the \
    actual wavenumber times the room length scale.
    * Multiply by the gains corresponding to each ray.
    * Divide by the distance corresponding to each ray segment.
    * Sum all the ray segments in a grid point.
    * Multiply the grid by the transmitted field times the wavelngth \
    divided by the room length scale. :math:`\\frac{\\lambda}{L 4 \pi}`
    * Multiply by initial polarisation vectors and combine.
    * Ignore dividing by initial phi as when converting to power in db \
    this disappears.
    * Take the amplitude and square.
    * Take :math:`10log10()` to get the db Power.

  :rtype: Nx x Ny x Nz numpy array of real values.

  :return: Grid

  '''
  t0=t.time()
  print('----------------------------------------------------------')
  print('Start computing the power from the Mesh')
  print('----------------------------------------------------------')
  # Retrieve the parameters
  khat,lam,L = Antpar

  # Compute the reflection coefficients
  print(Mesh[0,0,0],Mesh[1,1,1])
  print('Mesh equal check', Mesh.__self_eq__()) #DEBUG
  Rper, Rpar=ref_coef(Mesh,Znobrat,refindex)
  print('Ref equal check', Rper.__self_eq__()) #DEBUG

  # Combine the reflection coefficients to get the reflection loss on each ray.
  t1=t.time()
  print('----------------------------------------------------------')
  print('Multiplying reflection coefficients ')
  print('----------------------------------------------------------')
  Comper=Rper.dict_col_mult()
  print('Ref after mult equal check',Comper.__self_eq__()) #DEBUG
  Compar=Rpar.dict_col_mult()
  t2=t.time()
  print('----------------------------------------------------------')
  print('Reflection coefficients multiplied, time taken ', t2-t1)
  print('----------------------------------------------------------')
  # Get the distances for each ray segment from the Mesh
  RadMesh=Mesh.__get_rad__()
  print('Radius equal check',RadMesh.__self_eq__()) #DEBUG
  # Compute the mesh of phases
  pha=phase_calc(RadMesh,khat,L)
  print('Phase equal check',pha.__self_eq__()) #DEBUG
  # Divide by the rads
  pharad=pha/RadMesh
  # Multiply by the gains.
  Gtpha=pharad.dict_row_vec_multiply(np.sqrt(Gt))
  # Combine Gains, phase and reflection
  GtphaRpe=Gtpha*Comper
  print(GtphaRpe.__self_eq__())
  GtphaRpa=Gtpha*Compar
  # At this stage the terms are still different.
  # Sum cols
  Grid0pe=GtphaRpe.row_sum()
  Grid0pa=GtphaRpa.row_sum()
  # Multiply by the lambda\L
  Gridpe=Grid0pe.dict_scal_mult(lam*L/(4*np.pi))
  Gridpa=Grid0pa.dict_scal_mult(lam*L/(4*np.pi))
  # Turn into numpy array
  Gridpe=Gridpe.togrid()
  Gridpa=Gridpa.togrid()
  # FIXME polarisation dummy
  aper=np.array([1,1,1])
  apar=np.array([0,0,0])
  # Power
  P=np.power(np.absolute(Gridpe),2)
  P=10*np.log10(P)
  return P

def nonzero_bycol(SM):
  ''' Find the index pairs for the nonzero terms in a sparse matrix.
  Go through each column and find the nonzero rows.

  :param SM: sparse matrix.

  :return: [[i(0j0),i(1j0),...,i(nj0),...,i(njn)],\
    [j0,...,j0,...,jn,...,jn]]
  '''
  S2=SM.transpose()
  inddum=S2.nonzero()
  ind=[inddum[1],inddum[0]]
  return ind

#=======================================================================
# FUNCTIONS CONNECTED TO DS BUT AREN'T PART OF THE OBJECT
#=======================================================================


def load_dict(filename_):
  ''' Load a DS as a dictionary and construct the DS again.

  :param filename_: the name of the DS saved

  .. code::

     Nx=max(Keys[0])-min(Keys[0])
     Ny=max(Keys[1])-min(Keys[1])
     Nz=max(Keys[2])-min(Keys[2])

  :returns: nothing

  '''
  with open(filename_, 'rb') as f:
    ret_di = pkl.load(f)
  Keys=ret_di.keys()
  Nx=max(Keys)[0]-min(Keys)[0]
  Ny=max(Keys)[1]-min(Keys)[1]
  Nz=max(Keys)[2]-min(Keys)[2]
  na=ret_di[0,0,0].shape[0]
  nb=ret_di[0,0,0].shape[1]
  ret_ds=DS(Nx,Ny,Nz,na,nb)
  default_value=SM((na,nb),dtype=np.complex128)
  for k in Keys:
    ret_ds.__setitem__(k,ret_di[k])
  return ret_ds

def ref_coef(Mesh,Znobrat,refindex):
  ''' Find the reflection coefficients.

  :param Mesh: The DS mesh which contains terms re^(itheta) with theta \
  the reflection angle of incidence.

  Method:

    * Gets the mesh of angles using :py:class:`DS`. :py:func:`sparse_angles()`
    * Gets the indices of the nonzero terms using :py:class:`DS`. :py:func:`nonzero()`
    * Initialise sin(thetai), cos(thetai) and cos(thetat) meshes.
    * Compute cos(thetai),sin(thetai), cos(thetat)

    .. code::

      cthi=AngDSM.cos()
      SIN=AngDSM.sin()
      Div=SIN.dict_DSM_divideby_vec(refindex)
      ctht=Div.asin().cos()

    * Compute the reflection coefficients.

    .. code::

       S1=(cthi).dict_vec_multiply(Znobrat)
       S2=(ctht).dict_vec_multiply(Znobrat)
       Rper=(S1-ctht)/(S1+ctht)
       Rpar=(cthi-S2)/(cthi+S2)

  :rtype: Rper=DS(Nx,Ny,Nz,na,nb),Rpar=DS(Nx,Ny,Nz,na,nb)

  :returns: Rper, Rpar

  '''
  t0=t.time()
  print('----------------------------------------------------------')
  print('Retrieving the angles of reflection')
  print('----------------------------------------------------------')
  AngDSM=Mesh.sparse_angles()                       # Get the angles of incidence from the mesh.
  print('Angles equal check', AngDSM.__self_eq__()) #DEBUG
  ind=AngDSM.nonzero()                              # Return the indices for the non-zero terms in the mesh.
  ind=np.transpose(ind)
  SIN =DS(Mesh.Nx,Mesh.Ny,Mesh.Nz,Mesh.shape[0],Mesh.shape[1])   # Initialise a DSM which will be sin(theta)
  cthi=DS(Mesh.Nx,Mesh.Ny,Mesh.Nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta)
  ctht=DS(Mesh.Nx,Mesh.Ny,Mesh.Nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta_t) #FIXME
  print('----------------------------------------------------------')
  print('Computing cos(theta_i) on all reflection terms')
  print('----------------------------------------------------------')
  cthi=AngDSM.cos()                                   # Compute cos(theta_i)
  t1=t.time()
  print('----------------------------------------------------------')
  print('cos(theta_i) found time taken ', t1-t0)
  print('----------------------------------------------------------')
  print('----------------------------------------------------------')
  print('Computing cos(theta_t) on all reflection terms')
  print('----------------------------------------------------------')
  #SIN[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.sin(AngDSM[ind[0],ind[1],ind[2],ind[3],ind[4]]) # Compute sin(theta)
  #ctht[ind[0],ind[1],ind[2],ind[3],ind[4]]=np.cos(np.arcsin(Div[ind[0],ind[1],ind[2],ind[3],ind[4]]))
  SIN=AngDSM.sin()
  Div=SIN.dict_DSM_divideby_vec(refindex)             # Divide each column in DSM with refindex elementwise. Set any 0 term to 0.
  ctht=(Div.asin()).cos()
  t2=t.time()
  print('----------------------------------------------------------')
  print('cos(theta_t) found time taken ', t2-t1)
  print('----------------------------------------------------------')
  print('----------------------------------------------------------')
  print('Multiplying by the impedances')
  print('----------------------------------------------------------')
  S1=(cthi).dict_vec_multiply(Znobrat)                # Compute S1=Znob*cos(theta_i)
  S2=(ctht).dict_vec_multiply(Znobrat)                # Compute S2=Znob*cos(theta_t)
  t3=t.time()
  print('----------------------------------------------------------')
  print('Multiplication done time taken ', t3-t2)
  print('----------------------------------------------------------')
  print('----------------------------------------------------------')
  print('Computing the reflection coefficients.')
  print('----------------------------------------------------------')
  Rper=(S1-ctht)/(S1+ctht)                        # Compute the Reflection coeficient perpendicular
                                                  # to the polarisiation Rper=(Zm/Z0cos(theta_i)-cos(theta_t))/(Zm/Z0cos(theta_i)+cos(theta_t))

  Rpar=(cthi-S2)/(cthi+S2)                        # Compute the Reflection coeficient parallel
                                                  # to the polarisiation Rpar=(cos(theta_i)-Zm/Z0cos(theta_t))/(cos(theta_i)+Zm/Z0cos(theta_t))
  t4=t.time()
  print('----------------------------------------------------------')
  print('Reflection coefficients found, time taken ', t4-t3)
  print('----------------------------------------------------------')
  print('----------------------------------------------------------')
  print('Total time to compute reflection coefficients from Mesh ', t4-t0)
  print('----------------------------------------------------------')
  return Rper, Rpar

def parnonzero(nj,DS):
  ''' Parallel version of a program with a dummy DS and a function for \
  finding the indices of the nonzero terms in a mesh.

  :param nj: number of processes.
  :param DS: the mesh

  Pool the nj processes
  Specify what needs to be done.
  Combine the information.

  :return: 5xn array which n is the number of nonzero terms.

  '''
  x=np.arange(0,DS.Nx,1)
  y=np.arange(0,DS.Ny,1)
  z=np.arange(0,DS.Nz,1)
  coords=np.transpose(np.meshgrid(x,y,z))
  with Pool(processes=nj) as pool:         # start nj worker processes
    # prints "[0, 1, 4,..., 81]"
    ind=pool.map(DS.nonzeroMat, product(range(DS.nx),range(DS.ny),range(DS.nz)))
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

#=======================================================================

# TEST FUNCTIONS

#=======================================================================

def test_00():
  ds=DS()
  ds[1,2,3,0,0]=2+3j
  print(ds[1,2,3][0,0])

## Test creation of dictionary containing sparse matrices
def test_01(Nx=3,Ny=2,Nz=1,na=5,nb=6):
  ds=DS(Nx,Ny,Nz,na,nb)

##  test creation of matrix and adding on element
def test_02(Nx=7,Ny=6,Nz=1,na=5,nb=6):
  ds=DS(Nx,Ny,Nz,na,nb)
  ds[0,3,0,:,0]=2+3j
  print(ds[0,3,0])

## Test creation of diagonal sparse matrices contained in every position
def test_03(Nx,Ny,Nz,na,nb):
  ds=DS(Nx,Ny,Nz,na,nb)
  for x,y,z,a in product(range(Nx),range(Ny),range(Nz),range(na)):
    if a<nb:
      ds[x,y,z,a-1,a-1]=complex(a,a)
  return ds

## Test creation of first column sparse matrices contained in every position
def test_03b(Nx,Ny,Nz,na,nb):
  ds=DS(nx,ny,nz,na,nb)
  for x,y,z,a in product(range(Nx),range(Ny),range(Nz),range(na)):
    if a<nb:
      ds[x,y,z,a-1,2]=complex(a,a)
  return ds

## Test creation of lower triangular sparse matrices contained in every position
def test_03c(Nx,Ny,Nz,na,nb):
  ds=DS(Nx,Ny,Nz,na,nb)
  for x,y,z,a in product(range(Nx),range(Ny),range(Nz),range(na)):
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
  Nx=1
  Ny=1
  Nz=1
  na=1000
  nb=1000
  DSM=test_03c(Nx,Ny,Nz,na,nb)
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      M=DSM[i,j,k]
      AngM=sparse_angles(M)
  #print(AngM)
  return AngM

## Extract the cos of the reflection angles of the DS
def test_12():
  Nx=2
  Ny=2
  Nz=1
  na=3
  nb=3
  DSM=test_03c(Nx,Ny,Nz,na,nb)
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      M=DSM[i,j,k]
      indices=M.nonzero()
      CosAngM=SM(M.shape,dtype=float)
      CosAngM[indices]=np.cos(sparse_angles(M)[indices].todense())
  print(CosAngM)
  return CosAngM

## Attempt to find angle of nonzero element of SM inside dictionary
def test_13():
  Nx=2
  Ny=2
  Nz=1
  na=3
  nb=3
  DSM=test_03c(Nx,Ny,Nz,na,nb)
  ang=dict_sparse_angles(DSM)
  return 1

## Attempt to compute Reflection Coefficents on DS
def test_14():
  ''' This is a test of the reflection coefficient function.
  It sets test versions for the input parameters required and fills a DS \
  with dummy values.
  It then computes the reflection coefficients associated with those \
  dummy parameters and values.
  '''
  Nre=2                                        # Number of reflections
  Nob=12                                       # The Number of obstacle.
  Nra=20                                     # Number of rays
  na=Nob*Nre+1                                 # Number of rows in each SM in the DS
  nb=Nre*Nra+1                                 # Number of columns in each SM in the DS
  Nx=5                                         # Number of x spaces
  Ny=5
  Nz=5
  ds=test_03c( Nx , Ny , Nz , na , nb )         # test_03() initialises a
                                               # DSM with values on the
                                               # diagonal of each mesh element
  mur=np.full((Nob,1), complex(3.0,0))         # For this test mur is
                                               # the same for every obstacle.
                                               # Array created to get functions correct.
  epsr=np.full((Nob,1),complex(2.9493, 0.1065))         # For this test epsr is the
                                               # same for every obstacle
  sigma=np.full((Nob,1),0)                # For this test sigma is the
                                               # same for every obstacle

  # PHYSICAL CONSTANTS
  mu0=4*np.pi*1E-6
  c=2.99792458E+8
  eps0=1/(mu0*c**2)#8.854187817E-12
  Z0=(mu0/eps0)**0.5 #120*np.pi Characteristic impedance of free space.

  # CALCULATE PARAMETERS
  frequency=2*np.pi*2.79E+08                   # 2.43 GHz
  top=complex(0,frequency*mu0)*mur
  bottom=sigma+complex(0,eps0*frequency)*epsr
  Znob =np.sqrt(top/bottom)                    # Wave impedance of the obstacles
  del top, bottom
  Znob=np.tile(Znob,Nre)                      # The number of rows is Nob*Nre+1. Repeat Nob
  Znob=np.insert(Znob,0,complex(0.0,0.0))     # Use a zero for placement in the LOS row
  #Znob=np.transpose(np.tile(Znob,(Nb,1)))    # Tile the obstacle coefficient number to be the same size as a mesh array.
  Znobrat=Znob/Z0
  refindex=np.sqrt(np.multiply(mur,epsr))     # Refractive index of the obstacles
  refindex=np.tile(refindex,Nre)
  refindex=np.insert(refindex,0,complex(0,0))

  Rper,Rpar=ref_coef(ds,Znobrat,refindex)
  print(Rper,Rpar)
  return Rper

def test_15():
  ''' Testing multiplying nonzero terms in columns '''
  DS=test_14()
  print(DS)
  out=DS.dict_col_mult()
  print(out)
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
  ''' Test the :py:func:`parnonzero()` function which should find \
  nonzero() indices in parallel.
  '''
  Nob=3
  Nre=3
  Nra=5
  n=10
  nj=4
  DS=test_03(n,n,n,int(Nob*Nre+1),int((Nre)*(Nra)+1))
  #p = Process(target=nonzeroMat, args=(cor,DS))
  out=parnonzero(nj,DS)
  return

def test_18():
  '''Testing the save and load pickle functions. '''
  Nre=7                                        # Number of reflections
  Nob=12                                       # The Number of obstacle.
  Nra=100                                      # Number of rays
  na=Nob*Nre+1                                 # Number of rows in each SM in the DS
  nb=Nre*Nra+1                                 # Number of columns in each SM in the DS
  Nx=5                                         # Number of x spaces
  Ny=5
  Nz=10
  ds=test_03( Nx , Ny , Nz , na , nb )         # test_03() initialises a
                                               # DSM with values on the
                                               # diagonal of each mesh element
  filename=str('testDS')
  ds.save_dict(filename)
  ds=load_dict(filename)
  return

def test_19():
  ''' Test the :py:func:`nonzero_bycol(SM)` function.
  Initialise a dummy sparse matrix SM.

  In :py:func:`nonzero_bycol(SM)`:
    * Transpose the matrix.
    * Find the nonzero indices.
    * Swap the rows and columns in the indices.
    * Return the indices

  Check these match the nonzero terms in SM.

  :return: 0 if successful 1 if not.
  '''
  na=5
  nb=5
  S=SM((na,nb),dtype=np.complex128)
  for i in range(0,na):
    for j in range(0,nb):
      if j >= i :
        S[i,j]=i+1
  ind=nonzero_bycol(S)
  n=len(ind[0])
  for count in range(0,n):
    if S[ind[0][count],ind[1][count]] == 0:
      return 1
    else: pass
  return 0

def test_20():
  ''' Test the dict_col_mult() function.
  Use a dummy DS with each matrix upper triangular with the number in \
  every position the row.
  Check that the col_mult that comes out is the column number +1 \
  factorial.
  '''
  Nx=2
  Ny=2
  Nz=2
  na=5
  nb=5
  D1=DS(Nx,Ny,Nz,na,nb)
  for x,y,z in product(range(0,Nx),range(0,Ny),range(0,Nz)):
    for i in range(0,na):
      for j in range(0,nb):
        if j >= i :
          D1[x,y,z,i,j]=i+1
  D2=D1.dict_col_mult()
  for x,y,z in product(range(0,Nx),range(0,Ny),range(0,Nz)):
    for j in range(0,nb):
      if D2[x,y,x,0,j] != math.factorial(j+1):
        return 1
      else : pass
  return 0

def test_21():
  '''Test if the __get_rad__() function works.'''
  Nx=2
  Ny=2
  Nz=2
  na=5
  nb=5
  D=DS(Nx,Ny,Nz,na,nb)
  for x,y,z in product(range(0,Nx),range(0,Ny),range(0,Nz)):
    for i in range(0,na):
      for k in range(0,nb):
        if k >= i :
          D[x,y,z,i,k]=np.exp(1j*k)
  R=D.__get_rad__()
  for x,y,z in product(range(0,Nx),range(0,Ny),range(0,Nz)):
    for j in range(0,nb):
      if abs(R[x,y,x,0,j] -1)>epsilon :
        return 1
      else : pass
  return 0

def test_22():
  ''' Test the set_item() function for setting columns in a DSM '''
  Nx=3
  Ny=3
  Nz=3
  na=5
  nb=5
  Mesh=DS(Nx,Ny,Nz,na,nb)
  count=0
  vec=np.ones((nb,1))
  for x,y,z in product(range(Nx),range(Ny),range(Nz)):
    col=int(nb/2)
    for j in range(na):
      Mesh[x,y,z,j,col]=count*vec[j]
      count+=1
  if Mesh.__self_eq__():
    print(Mesh)
    return 1
  count=0
  Mesh=DS(Nx,Ny,Nz,na,nb)
  for x,y,z in product(range(Nx),range(Ny),range(Nz)):
    col=int(nb/2)
    Mesh[x,y,z,:,col]=count*vec
    count+=1
  if Mesh.__self_eq__():
    #print(Mesh)
    return 1
  else: return 0

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  #job_server = pp.Server()
  t=test_22()
  print(t)

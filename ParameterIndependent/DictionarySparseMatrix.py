#!/usr/bin/env python3
# Hayley 15th July 2020
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

 This module also contains functions which act on the class. Including: \

 * :py:func:`QualityFromPower` which computes the quality from an array on power values.
 * :py:func:`RadGrid`
 * :py:func:`load_dict` which loads a dictionary sparse matriz when given a string.
 * :py:func:`non_fromrow` which returns the obstacle number corresponding\
  to a DS row given the row number and total number of obstacles.
  * :py:func:`nre_fromrow` which returns the reflection number corresponding\
  to a DS row given the row number and total number of obstacles.
  * :py:func:`parnonzero`
  * :py:func:`phase_calc`
  * :py:func:`power_compute` this functions takes in obstacle parameters\
   and a DS and outputs an array of power values.
  * :py:func:`ref_coef` this functions computes the reflection \
  coefficents of nonzero terms in a DS. With inputs of the DS and the obstacle coefficients.
  * :py:func:`singletype` this function checks if an input term is a set\
   of terms in the form of an array, list or tuple, or a single term, \
   such as a float, complex number, or integer.
  * :py:func:`stopcheck`
  * :py:func:`stopchecklist`
 '''


import numpy as np
from numpy import array_equal as arr_eq
from numpy import cos, sin, sqrt
from numpy.linalg import norm as leng
from scipy.sparse import dok_matrix as SM
import scipy.sparse.linalg
from numpy.linalg import inv,pinv
from scipy.sparse import save_npz, load_npz
from itertools import product
import sys
import time as t
from six.moves import cPickle as pkl
from pathlib import Path
import timeit
import os
import logging
import pdb
import RayTracerMainProgram as RTM
#from collections import defaultdict

epsilon=sys.float_info.epsilon
#----------------------------------------------------------------------
# NOTATION IN COMMENTS
#----------------------------------------------------------------------
# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix
# DS or DSM is a DS object which is a dictionary of sparse matrices.
dbg=0
xcheck=2
ycheck=5
zcheck=9
newvar=1
if dbg:
  logon=1
else:
  logon=np.load('Parameters/logon.npy')

class DS:
  ''' The DS class is a dictionary of sparse matrices.
  The keys for the dictionary are (i,j,k) such that i is in [0,Nx],
  j is in [0, Ny], and k is in [0,Nz].
  SM=DS[x,y,z] is a na*nb sparse matrix, initialised with complex128 data type.
  :math:`na=(Nob*Nre+1)`
  :math:`nb=((Nre)*(Nra)+1)`
  The DS is initialised with keys Nx, Ny, and Nz to a dictionary with \
  keys,
  :math:`\{ (x,y,z) \\forall x \in [0,Nx), y \in [0,Ny), z \in [0,Nz)\}.`

  With the value at each key being an na*nb SM.

  '''
  def __init__(s,Nx=1,Ny=1,Nz=1,na=1,nb=1,dt=np.complex128,split=1):
    s.shape=(na,nb)
    Keys=product(range(Nx),range(Ny),range(Nz))
    #default_value=SM(s.shape,dtype=dt)
    #s.d=dict.fromkeys(Keys,SM(s.shape,dtype=dt))
    s.d={}
    #s.d=defaultdict(default_value)
    for k in Keys:
      s.d[k]=SM(s.shape,dtype=dt)
    s.Nx=Nx
    s.Ny=Ny
    s.Nz=Nz
    s.time=np.array([t.time()])
    s.split=int(split)
  def __get_SM__(s,smk,dk,n):
    ''' Get a SM at the position dk=[x,y,z].
    :param dk:  The x,y,z key to retrieve the sparse matrix.
    :param smk: The keys to the sparse matrix at the dk positions.
    :param n:   The types wants from the sparse matrix.

    * n indicates whether a whole SM is set, a row or a column.
    * If n==0 a whole SM.
    * If n==1 a row or rows.
      * n2 is the number of rows.
    * If n==2 a column or columns.
      * n2 is the number of columns.

    :rtype: DS

    :returns: The terms in the SM[smk] at s[dk].
    '''
    dt=type(s.d[0,0,0][0,0])
    if n==0:
      out=s.d[dk]
    elif n==1:
      # Get one row.
      if singletype(smk[0]): n2=1
      # Get multiple rows.
      else:
        n2=len(smk[0])
      # Get a row
      if n2==1:
        out=s.d[dk][smk,:]
      # Get multiple rows
      else:
        out=np.zeros((len(smk),s.shape[1]),dtype=dt)
        for j in smk:
          out[j,:]=s.d[dk][j,:]
    # Get a  column or columns.
    elif n==2:
      if singletype(smk[1]): n2=1
      elif isinstance(smk[1],slice): n2=1
      else:
        n2=len(smk[1])
      if n2==1:
        out=s.d[dk][smk[0],smk[1]]
      else:
        na=s.shape[0]
        out=np.zeros((na,n2),dtype=dt)
        for c in range(0,n2):
          out[:,c]=s.d[dk][smk[0][c],smk[1][c]]
    else:
      # Invalid 'i' the length does not match a possible position.
      errmsg=str('''Error setting the (%s) part of the sparse matr
      ix to (%s). Invalid index (%s). A 3-tuple is required to
      return a sparse matrix(SM), 4-tuple for the row of a SM or
      5-tuple for the element in the SM.''' %(i,x,i))
      raise IndexError(errmsg)
      pass
    return out
  def __getitem__(s,i):
    ''' Get all or part of a DSM.

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
      * If k==7. Set all [x1,...,xn] , [y1,...,y1], [z1,...,zn] terms.
        In this case a numpy array is returned.

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
    if singletype(dk[0]):
      if singletype(dk[1]):
        if singletype(dk[2]):
          k=-1
        else:
          k=2
      else:
        if singletype(dk[2]):
          k=1
        else:
          k=3
    elif len(dk[0])>1 and  len(dk[1])>1 and len(dk[2])>1:
      k=7
    else:
      if singletype(dk[1]):
        if singletype(dk[2]):
          k=0
        else:
          k=4
      else:
        if singletype(dk[2]):
          k=5
        else:
          k=6
    n=len(i)-3
    if n==1 and isinstance(i[3], type(slice)):
      n=0
    elif n==2 and isinstance(i[3],type(slice)):
      if isinstance(i[4],type(slice)):
        n=0
      else:
        pass
    elif n==2 and isinstance(i[4],type(slice)):
      n=1
    else:
      pass
    dt=type(s.d[0,0,0][0,0])
    if k==-1:
    ## Returning a scalar variable or sparse matrix at an exact \
    # position (dk0,dk1,dk2,smk0,smk1). Only return a sparse matrix is
    # smk is empty of slices.
      out=s.__get_SM__(smk,dk,n)
    elif k==0:
    ## Return a DS for all x in (0,Nx) and exact y=dk1, z=dk2, \
    # Each x,y,z term is either a SM or a scalar variable depending on
    # smk.
      out=DS(s.Nx,1,1,s.shape[0],s.shape[1])
      for x in range(0,s.Nx):
        dk=[x,dk[1],dk[2]]
        out[dk]=s.__get_SM__(smk,dk,n)
    elif k==1:
    ## Return a DS for all y in (0,Ny) and exact x=dk0, z=dk2, \
    # Each x,y,z term is either a SM or a scalar variable depending on
    # smk.
      out=DS(1,s.Ny,1,s.shape[0],s.shape[1])
      for y in range(0,s.Ny):
        dk=[dk[0],y,dk[2]]
        out[dk]=s.__get_SM__(smk,dk,n)
    elif k==2:
    ## Return a DS for all z in (0,Nz) and exact x=dk0, y=dk1 \
    # Each x,y,z term is either a SM or a scalar variable depending on
    # smk.
      out=DS(1,1,s.Nz,s.shape[0],s.shape[1])
      for z in range(0,s.Nz):
        dk=[dk[0],y,dk[2]]
        out[dk]=s.__get_SM__(smk,dk,n)
    elif k==3:
    ## Return a DS for all x in (0,Nx) and y in (0,Ny), and exact z=dk2, \
    # Each x,y,z term is either a SM or a scalar variable depending on
    # smk.
      out=DS(s.Nx,s.Ny,1,s.shape[0],s.shape[1])
      for x,y in product(range(0,s.Nx),range(0,s.Ny)):
        dk=[x,y,dk[2]]
        out[dk]=s.__get_SM__(smk,dk,n)
    elif k==4:
    ## Return a DS for all x in (0,Nx) and z in (0,Nz) and exact y=dk1 \
    # Each x,y,z term is either a SM or a scalar variable depending on
    # smk.
      out=DS(s.Nx,1,s.Nz,s.shape[0],s.shape[1])
      for x,z in product(range(0,s.Nx),range(0,s.Nz)):
        dk=[x,dk[1],z]
        out[dk]=s.__get_SM__(smk,dk,n)
    elif k==5:
    ## Return a DS for all i in (0,Ny), z in (0,Nz) and exact x=dk0 \
    # Each x,y,z term is either a SM or a scalar variable depending on
    # smk.
      out=DS(1,s.Ny,s.Nz,s.shape[0],s.shape[1])
      for y,z in product(range(0,s.Ny),range(0,s.Nz)):
        dk=[dk[0],y,z]
        out[dk]=s.__get_SM__(smk,dk,n)
    elif k==6:
    ## Return a DS for all x in (0,Nx), y in (0,Ny) and z in (0,Nz) \
    # Each x,y,z term is either a SM or a scalar variable depending on
    # smk.
      out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
      for dk in s.d.keys():
        out[dk]=s.__get_SM__(smk,dk,n)
    elif k==7:
      ## In the case when an 5xn co-ordinates are input output a nx1
      # array containing the terms in the respective positions.
      nkey=len(dk[0])
      out=np.zeros((nkey,1),dtype=dt)
      for count in range(0,nkey):
        sm=[smk[0][count],smk[1][count]]
        kd=(dk[0][count],dk[1][count],dk[2][count])
        out[count]=s.d[kd][sm[0],sm[1]] #s.__get_SM__(sm,kd,n)
    else: raise ValueError('no k has been assigned')
    return out
  def __set_SM__(s,smk,dk,x,n):
    ''' Set a SM at the position dk=[x,y,z].

    :param dk:  The dictionary key
    :param smk: The position within the sparse matrix at the dictionary key.
    :param x:   The term to set the sm[smk] terms to.
    :param n:   The type of terms being called.

    * n indicates whether a whole SM is set, a row or a column.
    * If n==0 a whole SM.
    * If n==1 a row or rows.
      * n2 is the number of rows.
    * If n==2 a column or columns.
      * n2 is the number of columns.
    '''
    if n==0:
      # If dk is not already one of the dictionary keys add the new
      # key to the DSM with an initialise SM.
      if dk not in s.d:
        s.d[dk]=SM(s.shape,dtype=np.complex128)
      # Assign 'x' to the SM with key dk.
      s.d[dk]=x
    elif n==1:
      # Set one row.
      if singletype(smk[0]): n2=1
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
        for j in smk:
          s.d[dk][j,:]=x[p]
          p+=1
    # set a SM element or column if smk[0]=: (slice) or multiple elements or columns.
    elif n==2:
      if singletype(smk[1]) or isinstance(smk[1],(slice)): n2=1
      else:
        n2=len(smk[1])
      if dk not in s.d:
        s.d[dk]=SM(s.shape,dtype=np.complex128)
      if n2==1:
        #print(x) #DEBUG
        if isinstance(smk[0],(slice)):
          n3=s.shape[0]
          if singletype(x):
            for i in range(n3):
              s.d[dk][i,smk[1]]=x
          else:
            if singletype(x[0]):
              s.d[dk][:,smk[1]]=x
            else:
              s.d[dk][:,smk[1]]=x[:][0]
        else:
          if isinstance(smk[1],slice):
            n3=s.shape[1]
            if singletype(x):
              for i in range(n3):
                s.d[dk][smk[0],i]=x
            else:
              if singletype(x[0]):
                for i in range(n3):
                  s.d[dk][smk[0],i]=x[i]
              else:
                 for i in range(n3):
                  s.d[dk][smk[0],i]=x[0,i]
          else:
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
      * If k==7. Set all [x1,...,xn] , [y1,...,y1], [z1,...,zn] terms.

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
    elif len(dk[0])>1 and  len(dk[1])>1 and len(dk[2])>1:
      k=7
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
          k=6
    n=len(i)-3
    if n==1 and isinstance(i[3], type(slice)):
      n=0
    elif n==2 and isinstance(i[3],type(slice)):
      if isinstance(i[4],type(slice)):
        n=0
      else:
        pass
    elif n==2 and isinstance(i[4],type(slice)):
      n=1
    if k==-1:
      s.__set_SM__(smk,dk,x,n)
    elif k==0:
      for x in range(0,s.Nx):
        dk=[x,dk[1],dk[2]]
        s.__set_SM__(smk,dk,x,n)
    elif k==1:
      for y in range(0,s.Ny):
        dk=[dk[0],y,dk[2]]
        s.__set_SM__(smk,dk,x,n)
    elif k==2:
      for z in range(0,s.Nz):
        dk=[dk[0],y,dk[2]]
        s.__set_SM__(smk,dk,x,n)
    elif k==3:
      for x,y in product(range(0,s.Nx),range(0,s.Ny)):
        dk=[x,y,dk[2]]
        s.__set_SM__(smk,dk,x,n)
    elif k==4:
       for x,z in product(range(0,s.Nx),range(0,s.Nz)):
        dk=[x,dk[1],z]
        s.__set_SM__(smk,dk,x,n)
    elif k==5:
      for y,z in product(range(0,s.Ny),range(0,s.Nz)):
        dk=[dk[0],y,z]
        s.__set_SM__(smk,dk,x,n)
    elif k==6:
      if isinstance(x,DS):
        for k in s.d.keys():
          s.d[k]=x[k]
      else:
        for dk in s.d.keys():
          s.__set_SM__(smk,dk,x,n)
    elif k==7:
      nkey=len(dk[0])
      for count in range(0,nkey):
        k=(dk[0][count],dk[1][count],dk[2][count])
        sm=[smk[0][count],smk[1][count]]
        s.__set_SM__(sm,k,x[count],n)
    else: raise ValueError('no k has been assigned')
    return
  def __str__(s):
    '''String representation of the DSM s.
    constructs a string of the keys with their corresponding values
    (sparse matrices with the nonzero positions and values).
    :return: the string of the DSM s.
    :rtype: string
    '''
    keys=s.d.keys()
    out=str()
    for k in keys:
      new= str(k) + str(s.d[k])
      out= (""" {0}
               {1}""").format(out,new)
    return out
  def __add__(s, DSM,ind=-1):
    '''
    Add sparse matrices from DSM and DSM2 elementwise if they have the /
    same dictionary key (x,y,z).
    :param DSM2: The Dictionary of sparse matrices which will be /
    elementwise added to s.

    :return: DSM=s+DSM2 for each point in s and DSM2.
    :rtype: Dictionary of sparse matrices.
    '''
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    if isinstance(ind, type(-1)):
      for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
        out[x,y,z]=s[x,y,z]+DSM[x,y,z]
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
      #x,y,z=ind[0][0:3]
      n=len(ind[0])
      for l in range(n):
        out[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]+DSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]
        # if x==ind[l][0] and y==ind[l][1] and z==ind[l][2]:
          # pass
        # else:
          # out[ind[l][0],ind[l][1],ind[l][2]]=s.d[ind[l][0],ind[l][1],ind[l][2]]+DSM2[ind[l][0],ind[l][1],ind[l][2]]
          # x,y,z=ind[l][0:3]
    return out
  def __sub__(s, DSM,ind=-1):
    '''
    Subtract DSM2 from DSM elementwise if they have the
    :rtype: a new DSM with the same dimensions
    '''
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    if isinstance(ind, type(-1)):
      for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
        out[x,y,z]=s[x,y,z]-DSM[x,y,z]
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
      #x,y,z=ind[0][0:3]
      n=len(ind[0])
      for l in range(n):
        out[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]-DSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]
        # if x==ind[l][0] and y==ind[l][1] and z==ind[l][2]:
          # pass
        # else:
          # out[ind[l][0],ind[l][1],ind[l][2]]=s.d[ind[l][0],ind[l][1],ind[l][2]]-DSM.d[ind[l][0],ind[l][1],ind[l][2]]
          # x,y,z=ind[l][0:3]
    return out
  def __mul__(s, M2,DSM2=0,ind=-1):
    ''' Multiply M2 with s.

    :param M2: is a DSM with the same dimensions as s, or a matrix with dimensions a,b or a scalar.
    :param DSM2: optional input, if used then the multiplication applied to s is also applied to DSM2.

    * If M2 is a scalar multiply all non-zero terms in s by M2.
    * If M2 is a matrix, array, list or tuple, with same dimensions as each SM in s\
     multiply every SM in s by M2.
    * If M2 is a matrix, array, list or tuple, with the dimensions[s.shape[0],1], or [1,s.shape[0]],\
    then multiply every column in each SM in s by M2.
    * If M2 is a DSM:

      * Perform matrix multiplication AB for all sparse matrices A in s and \
      B in DSM2 with the same key (x,y,z)

    outDSM=s[x,y,z]*M2[x,y,z] :math: `\\forall x, y, z, \in [0,Nx], [0,Ny], [0,Nz]`,\
    or s[x,y,z]*M2 :math: `\\forall x, y, z, \in [0,Nx], [0,Ny], [0,Nz]`,

    :returns: outDSM or (outDSM,outDSM2)

    :rtype: DSM with the dimensions[s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1]] \
    or (DSM with the dimensions[s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1]],DSM with the dimensions[s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1]])

    '''
    outDSM=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    if isinstance(DSM2,type(0)):
      double=0
    else:
      outDSM2=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
      double=1
    if isinstance(ind, type(-1)):
      if singletype(M2) and double==0:
        for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
          outDSM[x,y,z]=M2*s[x,y,z]
        return outDSM
      elif singletype(M2) and double==1:
        for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
          outDSM[x,y,z]=M2*s[x,y,z]
          outDSM2[x,y,z]=M2*DSM2[x,y,z]
        return outDSM,outDSM2
      elif isinstance(M2, (list, tuple, np.ndarray,SM)) and double==0:
        if M2.shape==s[x,y,z].shape:
          for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
            outDSM[x,y,z]=s[x,y,z].multiply(M2)
        else:
          for x,y,z,a,b in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz),range(0,s.shape[0]),range(0,s.shape[1])):
            outDSM[x,y,z,a,b]=s[x,y,z,a,b]*M2[a]
        return outDSM
      elif isinstance(M2, (list, tuple, np.ndarray,SM)) and double==1:
        if M2.shape==s[x,y,z].shape:
          for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
            outDSM[x,y,z]=s[x,y,z].multiply(M2)
            outDSM2[x,y,z]=DSM2[x,y,z].multiply(M2)
        else:
          for x,y,z,a,b in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz),range(0,s.shape[0]),range(0,s.shape[1])):
            outDSM[x,y,z,a,b]=s[x,y,z,a,b]*M2[a]
            outDSM2[x,y,z,a,b]=s[x,y,z,a,b]*M2[a]
        return outDSM,outDSM2
      else:
        if double==1:
          for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
            outDSM[x,y,z]=s[x,y,z].multiply(M2[x,y,z])
            outDSM2[x,y,z]=DSM2[x,y,z].multiply(M2[x,y,z])
          return outDSM,outDSM2
        else:
          for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
            outDSM[x,y,z]=s[x,y,z].multiply(M2[x,y,z])
          return outDSM
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
      n=len(ind[0])
      p=-1
      l=-1
      m=-1
      if singletype(M2) and double==0:
        for i in range(0,n):
          # No need to go through the same grid element again.
          if ind[0][i]==x and ind[1][i]==y and ind[2][i]==z:
            pass
          else:
            x=ind[0][i]
            y=ind[1][i]
            z=ind[2][i]
            outDSM[x,y,z]=M2*s[x,y,z]
        return outDSM
      elif singletype(M2) and double==1:
        for i in range(0,n):
          # No need to go through the same grid element again.
          if ind[0][i]==x and ind[1][i]==y and ind[2][i]==z:
            pass
          else:
            x=ind[0][i]
            y=ind[1][i]
            z=ind[2][i]
            outDSM[x,y,z]=M2*s[x,y,z]
            outDSM2[x,y,z]=M2*DSM2[x,y,z]
        return outDSM,outDSM2
      elif isinstance(M2, (list, tuple, np.ndarray,SM)) and double==0:
        if M2.shape==s.shape:
          for i in range(0,n):
            # No need to go through the same grid element again.
            if ind[0][i]==x and ind[1][i]==y and ind[2][i]==z:
              pass
            else:
              x=ind[0][i]
              y=ind[1][i]
              z=ind[2][i]
              outDSM[x,y,z]=s[x,y,z].multipy(M2)
        else:
          for i in range(0,n):
            # No need to go through the same grid element again.
            x=ind[0][i]
            y=ind[1][i]
            z=ind[2][i]
            a=ind[3][i]
            b=ind[4][i]
            outDSM[x,y,z,a,b]=s[x,y,z,a,b]*M2[a]
        return outDSM
      elif isinstance(M2, (list, tuple, np.ndarray,SM)) and double==1:
        if M2.shape==s.shape:
          for i in range(0,n):
            # No need to go through the same grid element again.
            if ind[0][i]==x and ind[1][i]==y and ind[2][i]==z:
              pass
            else:
              x=ind[0][i]
              y=ind[1][i]
              z=ind[2][i]
              outDSM[x,y,z]=s[x,y,z].multipy(M2)
              outDSM2[x,y,z]=DSM2[x,y,z].multipy(M2)
        else:
          for i in range(0,n):
            # No need to go through the same grid element again.
            x=ind[0][i]
            y=ind[1][i]
            z=ind[2][i]
            a=ind[3][i]
            b=ind[4][i]
            outDSM[x,y,z,a,b]=s[x,y,z,a,b]*M2[a]
            outDSM2[x,y,z,a,b]=DSM2[x,y,z,a,b]*M2[a]
        return outDSM,outDSM2
      else:
        if double==0:
          for i in range(0,n):
            # No need to go through the same grid element again.
            if ind[0][i]==x and ind[1][i]==y and ind[2][i]==z:
              pass
            else:
              x=ind[0][i]
              y=ind[1][i]
              z=ind[2][i]
              outDSM[x,y,z]=s[x,y,z].multipy(M2[x,y,z])
          return outDSM
        else:
          for i in range(0,n):
            # No need to go through the same grid element again.
            if ind[0][i]==x and ind[1][i]==y and ind[2][i]==z:
              pass
            else:
              x=ind[0][i]
              y=ind[1][i]
              z=ind[2][i]
              outDSM[x,y,z]=s[x,y,z].multipy(M2[x,y,z])
              outDSM2[x,y,z]=DSM2[x,y,z].multipy(M2[x,y,z])
        return outDSM,outDSM2
  ## Divide elementwise s with DSM2
  # Perform elementwise division A/B for all sparse matrices A in DSM
  # and B in DSM2 with the same key (x,y,z).
  # @param DSM2 is a DSM with the same dimensions as s.
  # @return a new DSM with the same dimensions
  def __truediv__(s, DSM2,ind=-1):
    if isinstance(ind, type(-1)):
      ind=DSM2.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=DSM2.nonzero().T
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    n=len(ind[0])
    for i in range(n):
      out[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]=np.true_divide(s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]],DSM2[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]])
    return out
  def __eq__(s,DSM2,ind=-1):
    ''' If all non-zero terms aren't equal then matrices aren't equal.
    :param: DSM2 the DSM to compare to.
    :param: ind, the nonzero indices of s. If this is input then only \
    these terms are checked. To check that there aren't more terms in \
    DSM2 then leave this term blank.

    :rtype: boolian

    :return: True if equal, False if not.
    '''
    if isinstance(ind, type(-1)):
      for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
        if scipy.sparse.linalg.norm(s[x,y,z]-DSM2[x,y,z],1) != 0:
          if scipy.sparse.linalg.norm(s[x,y,z]-DSM2[x,y,z],1)>epsilon:
            return False
          else: pass
        else: pass
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
      n=len(ind[0])
      for i in range(n):
        a=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
        b=DSM2[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
        if abs(a-b)<epsilon:
          continue
        else:
          return False
    return True
  def __self_eq__(s,ind=-1):
    ''' Check that the SMs at each grid point aren't equal.

    :param ind: Optional input, the non-zero indices.

    Method:
    Check each matrix at (x,y,z) position doesn't have the same
    non-zero terms as the other matrices.

    * If the non-zero indices aren't input then go through the x,y,z \
    options and check if there are any terms in any of the matrix at the\
    same (a,b) position which aren't equal to a term in the same position in another (x,y,z) matrix.\
    * If the indices are input then go through the list and compare terms\
     with the same a,b co-ordinate but different x,y,z co-ordinate.

    :rtype: boolian

    :return: True if equal, False if not.

    '''
    if isinstance(ind, type(-1)):
      for x1,y1,z1 in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
        for x2,y2,z2 in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
          if x1==x2 and y1==y2 and z1==z2: continue
          else:
            ind=s[x1,y1,z1].nonzero()
            n=len(ind[0])
            for i in range(n):
              if s[x1,y1,z1][ind[0][i],ind[1][i]]!=s[x2,y2,z2][ind[0][i],ind[1][i]]: return False
              else: pass
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
      n=len(ind[0])
      for i in range(n):
        x1=ind[0][i]
        y1=ind[1][i]
        z1=ind[2][i]
        a1=ind[3][i]
        b1=ind[4][i]
        for j in range(n):
          x2=ind[0][i]
          y2=ind[1][i]
          z2=ind[2][i]
          a2=ind[3][i]
          b2=ind[4][i]
          if x1==x2 and y1==y2 and z1==z2 and a1!=a2 and b1!=b2: continue
          elif s[x1,y1,z1,a1,b1]!=s[x2,y2,z2,a2,b2]: return False
          else: pass
    return True
  def __get_rad__(s,Nsur,ind=-1,plottype=str(),Nra=-1,Nre=-1,boxstr=str(),index=0,job=0):
    ''' Return a DS corresponding to the distances stored in the mesh.

    :param Nob: The number of obstacles.

    :param ind: The array of indices for non-zero terms. Defaults to -1 for a check to get it computed.

      * Initialise out=DS(Nx,Ny,Nz,1,s.shape[1])

      * Initialise RadA and RadB as 0 Nx x Ny x Nz arrays. Containing the LOS radius and the distances after reflection.

      * Go through all the nonzero x,y,z grid points.

      * Go through the nonzero columns and put the absolute value of the \
      first term in the corresponding column in out[x,y,z]

      * Pass until the next nonzero index is for a new column and repeat.

      * When non-zero terms are stored add the position [x,y,z,a,b] to \
      the array of non-zero indices indout.

    :rtype: DS(Nx,Ny,Nz,1,s.shape[1]) of real values, [Nx,Ny,Nz] numpy array,\
    [Nx,Ny,Nz] numpy array,[[x,y,z,a,b],...,[xn,yn,zn,an,bn]] numpy array

    :return: out, RadA, RadB,indout

    '''
    out=DS(s.Nx,s.Ny,s.Nz,1,s.shape[1],float)
    RadA=np.zeros((s.Nx,s.Ny,s.Nz),dtype=float)
    RadSi=np.zeros((Nsur,s.Nx,s.Ny,s.Nz),dtype=float)
    # Find the nonzero indices. There's no need to retrieve distances on 0 terms.
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    n=len(ind[0])
    check=-1
    for i in range(0,n):
      x,y,z,a,b=ind[0,i],ind[1,i],ind[2,i],ind[3,i],ind[4,i]
      l=abs(s[x,y,z][a,b])
      M=out[x,y,z]
      indM=M.nonzero()
      n2=len(indM[0])
      nob=nob_fromrow(a,Nsur)
      nre=nre_fromrow(a,Nsur)
      if s[x,y,z][:,b].getnnz()==nre+1:
        count=0
        if s[x,y,z][:,b].getnnz()==1:
          RadA[x,y,z]=l
        elif nre==1:
            #Assuming each surface is two triangles #FIXME
            RadSi[nob,x,y,z]=l
            #if dbg and l>2.0*np.sqrt(3):
            #  raise ValueError('Reflection rad is too long',l)
        if dbg and x==xcheck and y==ycheck and z==zcheck :
          logging.info('At position (%d,%d,%d,%d,%d). The distance of the ray segment is%f'%(x,y,z,0,b,l))
        if M[0,b]==0:
          out[x,y,z][0,b]=l
          if check==0:
            indicesSec=np.array([ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]])
            indout=np.vstack((indout,indicesSec))
          else:
            check=0
            indout=np.array([ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]])
    Ns=max(s.Nx,s.Ny,s.Nz)
    meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nra,Nre,Ns)
    if Nra>0 and Nre>0:
      RadAstr    =meshfolder+'/'+boxstr+'RadA_grid%dRefs%dm%d_tx%03d.npy'%(Nra,Nre,index,job)
      np.save(RadAstr,RadA)
      for j in range(0,Nsur):
        RadSistr=meshfolder+'/'+boxstr+'RadS%d_grid%dRefs%dm%d_tx%03d.npy'%(j,Nra,Nre,index,job)
        np.save(RadSistr,RadSi[j])
    return out,indout
  def __del_doubles__(s,h,Nsur,ind=-1,Ntri=-1):
    ''' Return the same DS as input but with double counted rays removed.
    Also output the onzero indices of this DSM.

    :meta public:

    :param h:   Mesh width

    :param Nob: The number of obstacles.

    :param ind: The array of indices for non-zero terms. Defaults to -1 for a check to get it computed.

      * Initialise out=DS(Nx,Ny,Nz,1,s.shape[1])

      * Initialise RadA and RadB as 0 Nx x Ny x Nz arrays. Containing the LOS radius and the distances after reflection.

      * Go through all the nonzero x,y,z grid points.

      * Go through the nonzero columns and put the absolute value of the \
      first term in the corresponding column in out[x,y,z]

      * On the future columns iterate through those already stored to check if they have the same obstacle number sequence.\
       If they do then don't store the next column as this is a repeat.

      * When a column is stored at the indices for the non-zero positions to indout.

    :rtype: DS(Nx,Ny,Nz,s.shape) of real values., np.array([x0,y0,z0,a0,b0],...,[xn,yn,zn,an,bn])

    :return: out,indout

    '''
    #if isinstance(Ntri,type(-1)):
    #  Ntri=np.ones(Nsur)
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])  # Initialise the mesh which will be output.
    # Find the nonzero indices, If ind is -1 then it is of default input
    # and needs to be found.
    # If ind[0] has length 5 and ind.T[0] does not have length five then
    #  the co-ordinates are going by row.
    # If int.T[0] has length 5 but ind[0] doesn't the co-ordinates are
    # in the right form.
    # Otherwise there are both 5 and it's not possible to tell if they
    # are the right way up so the co-ordinates need to be found again.
    if isinstance(ind, type(-1)):
      ind=s.nonzero()
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        pass #ind=ind
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        ind=ind.T
      else:
        ind=s.nonzero()
    n=ind.shape[0]                             # The number of non-zero terms
    check=-1                                   # This term is in case no non-zero indices are found
    indout=np.array([])
    x2=-1
    y2=-1
    z2=-1
    j2=-1
    for l in range(0,n):
      x,y,z,i,j=ind[l]
      #Only want to do work if this is a new column
      if x!=x2 or y!=y2 or z!=z2 or j!=j2:
        x2=x
        y2=y
        z2=z
        j2=j
        vec=s[x,y,z][:,j]
        vecnz=s[x,y,z][:,j].nonzero()[0]
        #vecnz=Correct_ObNumbers(vecnz,Ntri)
        vecnnz=s[x,y,z][:,j].getnnz()
        M=out[x,y,z]                        # The rays which have already been stored
        indM=M.nonzero()                    # The positions of the nonzero terms which are already stored.
        n2=M[0].getnnz()                    # The number of nonzero columns already stored.
        rep=False                           # Intialise the repetition variable to false. It'll become true if there's a repeat ray.
        c=-1
        for l in range(n2):
          if indM[1][l]==c:
            continue
          c=indM[1][l]
          indMrow=M[:,c].nonzero()[0]
          #indMrow=Correct_ObNumbers(indMrow,Ntri)
          rep=np.array_equal(indMrow,vecnz)
          if rep:
            break
        if not rep:
          out[x,y,z][:,j]=s[x,y,z][:,j]
          xvec=np.tile(x,(vecnnz,1))
          yvec=np.tile(y,(vecnnz,1))
          zvec=np.tile(z,(vecnnz,1))
          jvec=np.tile(j,(vecnnz,1))
          vecnz=vecnz.reshape(vecnnz,1)
          indicesSec=np.hstack((xvec,yvec,zvec,vecnz,jvec))
          if check==0:
            indout=np.vstack((indout,indicesSec))
          else:
            check=0
            indout=indicesSec
    return out,indout
  def doubles__inMat__(s,vec,po,ind=-1):
    ''' Check whether the input ray (vec) has already been counted in
    the DSM.

    :meta public:

    :param vec: The ray vector which will be set if there are no doubles.

    :param po:The x,y,z position of the SM to check.

      * Go through the nonzero columns and put the absolute value of the \
      first term in the corresponding column in out[x,y,z]

      * Pass until the next nonzero index is for a new column and repeat.

    :rtype: DS(Nx,Ny,Nz,1,s.shape[1]) of real values.

    :return: out

    '''
    if isinstance(ind,type(-1)):
      ind=vec.nonzero()[0]   # The positions of the nonzero terms in the vector looking to be stored
    M=s[po[0],po[1],po[2]]   # The matrix currently stored in the same (x,y,z) position
    M=M.toarray()
    for i in set(M.nonzero()[1]) : # the nonzero columns in the matrix M
      if arr_eq(M[:,i].nonzero()[0],ind):
        return True
    return False
  def doubles__inMat_for_ray_(s,vec,po,ind=-1):
    ''' Check whether the input ray (vec) has already been counted in
    the DSM.

    :meta public:

    :param vec: The ray vector which will be set if there are no doubles.

    :param po:The x,y,z position of the SM to check.

      * Go through the nonzero columns and put the absolute value of the \
      first term in the corresponding column in out[x,y,z]

      * Pass until the next nonzero index is for a new column and repeat.

    :rtype: DS(Nx,Ny,Nz,1,s.shape[1]) of real values.

    :return: out

    '''
    if isinstance(ind,type(-1)):
      ind=vec.nonzero()[0]   # The positions of the nonzero terms in the vector looking to be stored
    M=s[po[0],po[1],po[2]]   # The matrix currently stored in the same (x,y,z) position
    M=M.toarray()
    for i in set(M.nonzero()[1]) : # the nonzero columns in the matrix M
      if arr_eq(M[:,i].nonzero()[0],ind):
        for j in range(len(M.nonzero()[0])):
          a=M.nonzero()[0][j]
          b=M.nonzero()[1][j]
          if b==i:
            s.d[po[0],po[1],po[2]][a,b]=0
    return


  def refcoefbyterm_withmul(s,m,refindex,LOS=0,PerfRef=0, ind=-1,Nsur=6):
    ''' Using the impedance ratios of the obstacles, \
    refractive index of obstacles, the wavelength and the
    length scaling the unit mesh the reflection coefficents for each ray
    are calculated.

    :meta public:

    :param m: The vector of impedance ratios. (Repeated to line up with terms in Mesh).

    :param refindex: The vector of refractive indices of obstacles. (Repeated to line up with terms in Mesh).

    :param LOS: Line of sight yes LOS=1, Line of sight no LOS=0. \
    This overrides PerfRef, if LOS=1 then no reflection is stored regardless of PerfRef.

    :parm PerfRef: Perfect Reflection yes or no. If yes then PerfRef=1 \
    and angles are ignored at reflection. If no then PerfRef=0.

    out1 is the combinations of the reflection coefficients for the parallel to polarisation terms.

    out2 is the combinations of the reflection coefficients for the perpendicular to polarisation terms.

    * out1 and out2 are initialised to be zero everywhere with dimensions (Nx,Ny,Nz,1,nb)
    * Go through the non-zero indices (i0,i1,i2,i3,i4). If the row number (i3) is 0 and there are no other terms in the column \
    this is a line of sight ray.
    * If(i3==0):

      * If the term in out1(i0,i1,i2,0,i4) is zero then this and out2(i0,i1,i2,0,i4) should be set to 1.
      * If out1(i0,i1,i2,0,i4) is non-zero then this ray is already accounted for and nothing should be done.

    * Else:

       * Calculate the reflection coefficients.

       .. math::

          \\theta=s[i0,i1,i2,i3,i4]
          cthi=\\cos(\\theta)
          ctht=\\cos(\\arcsin(\\sin(\\theta)/n))
          S1=m[i3]*cthi
          S2=mpi3]*ctht
          Rpar=(S1-ctht)/(S1+ctht)
          Rper=(cthi-S2)/(cthi+S2)

      * out1[i0,i1,i2,0,i4]=Rper, out2[i0,i1,i2,0,i4]=Rpar

    :rtype: 2 DSMs with dimensions (Nx,Ny,Nz,1,nb)

    :return: out1 out2

    '''
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    out1=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    out2=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    n=len(ind[0])
    j2=-1
    x2=-1
    y2=-1
    z2=-1
    Nre=int((s.shape[0]-1)/Nsur)
    Ns=int(np.count_nonzero(refindex)-1/Nre)# Number of reflective surfaces
    if LOS==0 and PerfRef==0:
      for l in range(n):
        x,y,z,i,j=ind[:,l]
        if x!=x2 or y!=y2 or z!=z2 or j!=j2:
          x2=x
          y2=y
          z2=z
          j2=j
          out1[x,y,z][0,j]=1
          out2[x,y,z][0,j]=1
          if i!=0 or s[x,y,z][:,j].getnnz()!=1:
            colnz=s[x,y,z][1:,j].nonzero()[0]
            for k in colnz:
              # Reflection coefficient for non-LOS rays.
              #pdb.set_trace()
              if abs(refindex[k+1])<epsilon:
                out1[x,y,z][0,j]=0
                out2[x,y,z][0,j]=0
              else:
                theta=s[x,y,z][k+1,j]
                # The zero angle case is dealt with separately because otherwise it's not seen.
                if 4-epsilon<=theta<=4+epsilon:
                  theta=0.0
                if dbg:
                  assert -epsilon<theta<np.pi*0.5+epsilon
                    #errmsg='Reflection angle should be within 0 and pi/2 %f, epsilon %f'%(theta,epsilon)
                    #logging.error(errmsg)
                    #raise ValueError(errmsg)
                #pdb.set_trace()
                cthi=cos(theta)
                #ctht=np.cos(np.arcsin(np.sin(theta)/refindex[k+1]))
                ctht=sqrt(1-(sin(theta)/refindex[k+1])**2)
                #if dbg:
                #  pdb.set_trace()
                #pdb.set_trace()
                S1=cthi*m[k+1]
                S2=ctht*m[k+1]
                if abs(cthi)<epsilon and abs(ctht)<epsilon:
                  # Ref coef is very close to 1
                  if abs(m[k+1])>epsilon:
                    #print(ind[3][i])
                    frac=(m[k+1]-1)/(m[k+1]+1)
                    out1[x,y,z][0,j]*=frac
                    out2[x,y,z][0,j]*=frac
                else:
                  # Store the first reflection term of the ray segment
                  # Compute the Reflection coeficient perpendicular
                  # Rper=(S1-ctht)/(S1+ctht,ind)
                  # to the polarisiation Rper=(Zm/Z0cos(theta_i)-cos(theta_t))/(Zm/Z0cos(theta_i)+cos(theta_t))
                  # Compute the Reflection coeficient parallel
                  # Rpar=(cthi-S2).__truediv__(cthi+S2,ind)
                  out1[x,y,z][0,j]*=(S1-ctht)/(S1+ctht)
                  out2[x,y,z][0,j]*=(cthi-S2)/(cthi+S2)
                if dbg:
                  if s[x,y,z][:,j].getnnz()>Nre+1 and np.prod(refindex[colnz])!=0:
                    errmsg='Error at position, (%d,%d,%d,%d)'%(x,y,z,j)
                    logging.error(errmsg)
                    logging.error('The ray has reflected twice but the zero wall is not picked up')
                    errmsg='The angle element is, '+str(s[x,y,z])
                    logging.error(errmsg)
                    errmsg='The refindex terms are, '+str(refindex[colnz])
                    logging.error(errmsg)
                    errmsg='Nonzero rows'+str(colnz)
                    logging.error(errmsg)
                    errmsg='Full refindex'+str(refindex)
                    logging.error(errmsg)
                    exit()
                  if out1[x,y,z][0].getnnz()>Nre*Ns+1:
                    logging.error('Error at position (%d,%d,%d,%d,%d)'%(x,y,z,0,j))
                    logging.error('Too many rays have hit this mesh element,Position (%d,%d,%d,%d)'%(x,y,z,j))
                    errormsg='The element matrix'+str(out1[x,y,z])
                    logging.error(errormsg)
                    errormsg='The input ray matrix'+str(s[x,y,z])
                    logging.error(errormsg)
                    logging.error('col nonzero terms'+str(colnz))
                    logging.error('refindex terms'+str(refindex[colnz]))
                    exit()
    elif LOS==0 and PerfRef==1:
      '''When PerfRef==1 the refindex terms are 'a<=1' in the positions \
      corresponding to reflective surfaces and 0 elsewhere. This is not the real refractive indexes.'''
      for l in range(n):
        x,y,z,i,j=ind[:,l]
        colnz=s[x,y,z][:,j].nonzero()[0]
        if x!=x2 or y!=y2 or z!=z2 or j!=j2:
          # If you are in a new column then store the product of the \
          # reflection terms in refindex which have the same position as the non-zero terms in the angles mesh s.
          out1[x,y,z][0,j]=np.prod(refindex[colnz])
          out2[x,y,z][0,j]=np.prod(refindex[colnz])
          x2=x
          y2=y
          z2=z
          j2=j
          if dbg:
            if x==xcheck and y==ycheck and z==zcheck  :
              logging.info('At position(%d,%d,%d,%d)'%(x,y,z,j))
              logging.info('Refcoef%f+%fj'%(out1[x,y,z][0,j].real,out1[x,y,z][0,j].imag))
              logging.info('Refindex'+str(refindex[colnz])+' '+str(colnz))
              logging.info('Angles'+str(s[x,y,z][:,j]))
        if dbg:
          if refindex.getnnz()<=2:
            if s[x,y,z][:,j].getnnz()>Nre+1 and np.prod(refindex[colnz])!=0:
              errmsg='Error at position, (%d,%d,%d,%d)'%(x,y,z,j)
              logging.error(errmsg)
              logging.error('The ray has reflected twice but the zero wall is not picked up')
              errmsg='The angle element is, '+str(s[x,y,z])
              logging.error(errmsg)
              errmsg='The refindex terms are, '+str(refindex[colnz])
              logging.error(errmsg)
              errmsg='Nonzero rows'+str(colnz)
              logging.error(errmsg)
              errmsg='Full refindex'+str(refindex)
              logging.error(errmsg)
              exit()
          if out1[x,y,z][0].getnnz()>Nre*Ns+1:
            errormsg='More than two rays have hit this mesh element,Position (%d,%d,%d,%d)'%(x,y,z,j)
            logging.error(errormsg)
            errormsg='The element matrix'+str(out1[x,y,z])
            logging.error(errormsg)
            errormsg='The recent ray'+str(s[x,y,z][:,j])
            logging.error(errormsg)
            exit()
          if x==xcheck and y==ycheck and z==zcheck  :
            logging.info('At position(%d,%d,%d,%d)'%(x,y,z,j))
            logging.info('Refcoef%f+%fj'%(out1[x,y,z][0,j].real,out1[x,y,z][0,j].imag))
            logging.info('Refindex'+str(refindex[colnz])+' '+str(colnz))
            logging.info('Angles'+str(s[x,y,z][:,j]))
    elif LOS==1:
      for l in range(n):
        x,y,z,i,j=ind[:,l]
        if i==0 and s[x,y,z,:,j].getnnz()==1:
          # LOS ray has hit the cell
          #if j==22:
              #pdb.set_trace()
          out1[x,y,z][0,j]=1
          out2[x,y,z][0,j]=1
    else: raise ValueError('Incorrect Value for LOS')
    return out1,out2

  def togrid(s,ind=-1):
    ''' Compute the matrix norm at each grid point and return a numpy array.
    If there is only one term at each grid point then this converts the
    DSM which is a dictionary of sparse matrices with one term to a numpy array.

    :meta private:

    :rtype: A numpy array of shape (Nx,Ny,Nz)

    :return: Grid

    '''
    Nx=s.Nx
    Ny=s.Ny
    Nz=s.Nz
    Grid=np.zeros((Nx,Ny,Nz),dtype=np.complex128)
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    n=len(ind[0])
    for i in range(0,n):
      Grid[ind[0][i],ind[1][i],ind[2][i]]+=s[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]
    return Grid
  def check_nonzero_col(s,Nre=1,Nsur=1,nre=-1,ind=-1):
    ''' Check the number of non-zero terms in a column of a SM in s is \
    less than the maximum number of reflections+1(LOS). If nre and ind=[x,y,z,:,b] are \
    input then number of nonzero terms in columns b is equal to nre+1.

    :param Nre: The maximum number of reflections for any ray.
    :param Nsur: The number of surfaces in the envrionment. Not needed in nre and ind are input.
    :param nre: The reflection number of the ray being checked. Only input if ind is also input.
    :param ind: The position of the ray segment to be checked. If not input then all positions are checked.

    The code checks which inputs are given.

    * If nre and ind are given then the number of nonzero terms in \
    the mesh s is found. If this is not equal to nre+1 then there is the incorrect number, return false. If \
    the number of nonzero terms is nre+1 then return true.
    * If nre and ind aren't given then go through every nonzero matrix in the DSM and every nonzero column within\
    them. If the number of nonzero terms in that column is not equal to the reflection number calculated from \
    the last term in the column then return false. Otherwise return true.

    :rtype: bool
    :returns: True (Correct number of zero terms) or False. (Incorrect number of nonzero terms)
    '''
    if isinstance(ind, type(-1)):
      for x,y,z,b in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz),range(0,s.shape[1])):
        if 0==s[x,y,z][:,b].getnnz():
            continue
        elif 0<s[x,y,z][:,b].getnnz()<=Nre+1:
          indch=s[x,y,z][:,b].nonzero()
          r=indch[0][-1]
          nre=nre_fromrow(r,Nsur)
          if dbg:
            if nre+1!=s[x,y,z][:,b].getnnz():
              errmsg='Error at position(%d,%d,%d)'%(x,y,z)
              logging.error(errmsg)
              errmsg='The number of nonzero terms in column %d is not nre+1 %d'%(b,nre+1)
              logging.error(errmsg)
              logging.error(str(s[x,y,z][:,b]))
              pdb.set_trace()
              return False
        else: return False
      return True
    else:
      return s[ind[0],ind[1],ind[2]][:,ind[-1]].getnnz()==nre+1
    return True
  def asin(s,ind=-1):
    """ Finds the arcsin of the nonzero terms in the DSM.

    :meta private:

    :param ind: The nonzero indices of s. if this is not input then this is found using :func:`nonzero`.

    :math:`\\theta=\\arcsin(x)` for all terms :math:`x != 0` in \
    the DS s. Since all angles \
    :math:`\\theta` are in :math:`[0,\pi /2]`, \
    :math:`\\arcsin(x)` is not a problem.

    :returns: DSM  with :math:`\\arcsin(s)=\\theta` in \
     the same positions as the corresponding theta terms.

    :rtype: DSM with dimensions (Nx,Ny,Nz, na,nb) (same as input DSM).

    """
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
        #pass
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
        #ind=ind.T
      else:
        ind=s.nonzero().T
    na,nb=s.shape
    asinDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    n=len(ind[0])
    for j in range(0,n):
      asinDSM[ind[0][j],ind[1][j],ind[2][j]][ind[3][j],ind[4][j]]=np.arcsin(s[ind[0][j],ind[1][j],ind[2][j]][ind[3][j],ind[4][j]])
    return asinDSM
  ## Finds cos(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with cos(theta) in the
  # same position as the corresponding theta terms.
  def cos(s,ind=-1):
    """ Finds :math:`\\cos(\\theta)` for all terms \
    :math:`\\theta != 0` in the DS s.

    :rtype: A DSM with the same dimensions with \
    :math:`\\cos(\\theta)` in the \
     same position as the corresponding theta terms.

    """
    if isinstance(ind, type(-1)):
      ind=s.nonzero()
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero()
    na,nb=s.shape
    CosDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    n=len(ind[0])
    for j in range(0,n):
      CosDSM[ind[0][j],ind[1][j],ind[2][j]][ind[3][j],ind[4][j]]=np.cos(s[ind[0][j],ind[1][j],ind[2][j]][ind[3][j],ind[4][j]])
    return CosDSM
  ## Finds sin(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with sin(theta) in the
  # same position as the corresponding theta terms.
  def sin(s,ind=-1):
    """ Finds :math:`\\sin(\\theta)` for all terms \

    :meta private:

    :math:`\\theta != 0` in the DS s.

    :return: A DSM with the same dimensions with \
    :math:`\\sin(\\theta)` in the \
     same position as the corresponding theta terms.

    """
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
        #pass
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
        #ind=ind.T
      else:
        ind=s.nonzero().T
    na,nb=s.shape
    SinDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    #ind=np.transpose(ind)
    n=len(ind[0])
    for j in range(0,n):
      SinDSM[ind[0][j],ind[1][j],ind[2][j]][ind[3][j],ind[4][j]]=np.sin(s[ind[0][j],ind[1][j],ind[2][j]][ind[3][j],ind[4][j]])
    return SinDSM
  def image_real_parts(s,ind=-1):
    """ Return the imaginary and Real Parts of s. As DSMs with the same shape
    """
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
        #pass
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
        #ind=ind.T
      else:
        ind=s.nonzero().T
    na,nb=s.shape
    RealDSM=DS(s.Nx,s.Ny,s.Nz,na,nb,dt=float)
    ImageDSM=DS(s.Nx,s.Ny,s.Nz,na,nb,dt=float)
    #ind=np.transpose(ind)
    n=len(ind[0])
    for j in range(0,n):
      x,y,z,a,b=ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]
      ImageDSM[x,y,z][a,b]=s[x,y,z][a,b].imag
      RealDSM[x,y,z][a,b]=s[x,y,z][a,b].real
    return RealDSM,ImageDSM
  ## Finds the angles theta which are the arguments of the nonzero
  # complex terms in the DSM s.
  # @return a DSM with the same dimensions with theta in the
  # same position as the corresponding complex terms.
  def sparse_angles(s,ind=-1):
    """ Finds the angles :math:`\\theta` which are the arguments \
    of the nonzero complex terms in the DSM s.

    :param ind: The non-zero indices of s in the form [[x0,y0,z0,a0,b0],...,[xn,yn,zn,an,bn]] \
    if ind is not input then the function :func:`nonzero` is run at the start to find it.

    * Go through the nonzero terms (i0,i1,i2,i3,i4) in s.

    :math:`\\theta`  is the angle of the term at s[i0,i1,i2,i3,i4]

    * AngDSM[i0,i1,i2,i3,i4]=:math:`\\theta`

    :rtype: A DSM with the same dimensions as s

    :return: AngDSM

    """
    AngDSM=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1],dt=float)
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    n=len(ind[0])
    for j in range(0,n):
      x,y,z,a,b=ind[0,j],ind[1,j],ind[2,j],ind[3,j],ind[4,j]
      #pdb.set_trace()
      AngDSM[x,y,z][a,b]=np.angle(s[x,y,z][a,b])
      if abs(np.angle(s[x,y,z][a,b]))<epsilon:
        AngDSM[x,y,z][a,b]=4
    return AngDSM
  def opti_func_mats(s,RealPer,RealPar,ImagePer,ImagePar,khat,L,lam,Pol,Nra,ind):
    '''
    Computes the optimal antenna gains discretised into Nra rays.

    :param Realper: The real parts of the products of the perpendicular reflection coefficients.
    :param Realpar: The real parts of the products of the parallel reflection coefficients.
    :param Imageper: The imaginary parts of the products of the perpendicular reflection coefficients.'
    :param Imageper: The imaginary parts of the products of the parallel reflection coefficients.
    :param khat: Non-dimensional wavenumber.
    :param L: lengthscale
    :param lam: wavelength
    :param Pol: polarisation
    :param ind: nonzero indices.

    :math: Hx=\sqrt(|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*ImagePer*Pol[0]|**2
              +|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*ImagePar*Pol[1]|**2)
              \sqrt(|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*RealPer*Pol[0]|**2
              +|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*RealPar*Pol[1]|**2)

    :math: Fx=\sqrt(|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*ImagePer*Pol[0]|**2
              +|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*ImagePar*Pol[1]|**2)
              \sqrt(|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*RealPer*Pol[0]|**2
              +|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*RealPar*Pol[1]|**2)

    :rtype: 2x DSM
    :returns: Hx, Fx

    '''
    Hx=DS(s.Nx,s.Ny,s.Nz,1,Nra,dt=float)
    Fx=DS(s.Nx,s.Ny,s.Nz,1,Nra,dt=float)
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    Ni=len(ind[0])
    for l in range(0,Ni):
      x,y,z,a,b=ind[:,l]
      if s[x,y,z][a,b]!=0:
        nra=b%Nra
        r=s[x,y,z][a,b]
        coskdiv=cos(khat*r*(L**2))/r
        sinkdiv=sin(khat*r*(L**2))/r
        Hx[x,y,z,0,nra]+=coskdiv*(abs(ImagePer[x,y,z][a,b]*Pol[0])**2+abs(ImagePar[x,y,z][a,b]*Pol[1])**2)
        Hx[x,y,z,0,nra]+=sinkdiv*(abs(RealPer[x,y,z][a,b]*Pol[0])**2+abs(RealPar[x,y,z][a,b]*Pol[1])**2)
        Fx[x,y,z,0,nra]+=coskdiv*(abs(RealPer[x,y,z][a,b]*Pol[0])**2+abs(RealPar[x,y,z][a,b]*Pol[1])**2)
        Fx[x,y,z,0,nra]+=sinkdiv*(abs(ImagePer[x,y,z][a,b]*Pol[0])**2+abs(ImagePar[x,y,z][a,b]*Pol[1])**2)
    return Hx,Fx
  def opti_combo_inverse(s,Fx,Nra):
    '''
    Computes the optimal antenna gains discretised into Nra rays.

    :param s: s is the Hx imaginary matrix from :py:func:'opti_func_mats'
    :math: Hx=\sqrt(|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*ImagePer*Pol[0]|**2
              +|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*ImagePar*Pol[1]|**2)
              \sqrt(|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*RealPer*Pol[0]|**2
              +|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*RealPar*Pol[1]|**2)
    :param Fx:
    :math: Fx=\sqrt(|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*ImagePer*Pol[0]|**2
              +|(\sum_{nre} \sin(\wavenumber s*L**2)/s)*ImagePar*Pol[1]|**2)
              \sqrt(|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*RealPer*Pol[0]|**2
              +|(\sum_{nre} \cos(\wavenumber s*L**2)/s)*RealPar*Pol[1]|**2)

     Ainv=(Fx.^*Fx+Hx.THx)^{-1} at every x.

    :rtype: 2x DSM
    :returns: Hx, Fx

    '''
    na,nb=s.shape
    Ainv=DS(s.Nx,s.Ny,s.Nz,Nra,Nra,dt=float)
    Atot=np.zeros((Nra,Nra),dtype=float)
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      H=s.d[x,y,z].toarray().copy()
      F=Fx[x,y,z].toarray().copy()
      Amat=np.matmul(F.transpose(),F)+np.matmul(H.transpose(),H)
      for i,j in product(range(Nra),range(Nra)):
        Atot[i,j]+=Amat[i,j].copy()
    Ainv=pinv(Atot)
    Aout=np.zeros((Nra,1),float)
    for i,j in product(range(Nra),range(Nra)):
      Aout[i,0]+=Ainv[i,j]
    Aout/=leng(Aout)
    return Aout
  def gain_phase_rad_ref_mul_add(s,Com1,Com2,G,khat,L,lam,Nra=0,ind=-1):
    """ Multiply all terms of s elementwise with Com1/Rad and each row by Gt.
        Multiply all terms of s elementwise with Com2/Rad and each row by Gt.

    :param G: a row vector with length na.
    :param Rad: A DSM with size Nx, Ny,Nz, 1,na
    :param Com1: A DSM with size Nx, Ny,Nz, 1,na
    :param Com2: A DSM with size Nx, Ny,Nz, 1,na

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out1[x,y,z]+=G[j]*s[x,y,z,k,j]*Rad[x,y,z,k,j]*Com1[x,y,z,k,j]
       out2[x,y,z]+=G[j]*s[x,y,z,k,j]*Rad[x,y,z,k,j]*Com2[x,y,z,k,j]

    :rtype: A np.array 'out' of size Nx, Ny,Nz

    :returns: out

     """
    na,nb=s.shape
    out1=np.zeros((s.Nx,s.Ny,s.Nz),dtype=np.complex128)
    out2=np.zeros((s.Nx,s.Ny,s.Nz),dtype=np.complex128)
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    Ni=len(ind[0])
    for l in range(0,Ni):
      x,y,z,a,b=ind[:,l]
      if s[x,y,z][a,b]!=0:
        k=FieldEquation(s[x,y,z][a,b],khat,L,lam)
        if Nra>0:
          nra=b%Nra
        else:
          nra=b
        out1[x,y,z]+=np.sqrt(G[nra-1,0])*k*Com1[x,y,z][a,b]
        out2[x,y,z]+=np.sqrt(G[nra-1,0])*k*Com2[x,y,z][a,b]
      if x==xcheck and y==ycheck and z==zcheck and dbg:
        logging.info('This is an information message')
        logmessage='At position (%d,%d,%d)'%(x,y,z)
        logging.info(logmessage)
        logmessage='Value stored in parallel to polarisation'+str(out1[x,y,z])
        logging.info(logmessage)
        logmessage='Value stored in perpendicular to polarisation'+str(out2[x,y,z])
        logging.info(logmessage)
        logmessage='Value on the current ray segment %f'%k
        logging.info(logmessage)
        logmessage='Antenna parameters, wavenumber %f, wavelength %f and lengthscale %f'%(kat,lam,L)
        logginginfo(logmessage)
    return out1,out2
  def dict_row_vec_multiply(s,vec,ind=-1):
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
    if isinstance(ind, type(-1)):
      ind=s.nonzero()
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        pass
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        ind=ind.T
      else:
        ind=s.nonzero()
    Ni=len(np.transpose(ind)[4]) #FIXME find the nonzero columns without repeat column index for each term
    j=-1
    x,y,z=ind[0][0:3]
    for l in range(0,Ni):
      outDSM[ind[l][0],ind[l][1],ind[l][2]][ind[l][3],ind[l][4]]=vec[ind[l][4]]*s[ind[l][0],ind[l][1],ind[l][2]][ind[l][3],ind[l][4]]
    return outDSM
  def dict_DSM_divideby_vec(s,vec,ind=-1):
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
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    n=len(ind[0])
    for i in range(n):
      outDSM[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]=s[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]/vec[ind[3][i]]
        # x=ind[0][i]
        # y=ind[1][i]
        # z=ind[2][i]
        # bi=ind[4][i]
        # if x==p and y==l and z==m and q==bi:
          # pass
        # else:
          # outDSM[x,y,z,:,bi]=s[x,y,z,:,bi]/vec
          # q=bi
          # p=x
          # l=y
          # m=z
    return outDSM

  def dict_col_mult_(s,ind=-1):
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
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    n=len(ind[0])
    for i in range(0,n):
      if out[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]==0:
        out[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]=s[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]
      else:
        out[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]*=s[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]
    return out
  def double_dict_col_mult_(s,DSM,ind=-1):
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
    out2=DS(s.Nx,s.Ny,s.Nz,1,s.shape[1])
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    n=len(ind[0])
    for i in range(0,n):
      if out[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]==0:
        out[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]=s[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]
        out2[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]=DSM[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]
      else:
        out[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]*=s[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]
        out2[ind[0][i],ind[1][i],ind[2][i]][0,ind[4][i]]*=DSM[ind[0][i],ind[1][i],ind[2][i]][ind[3][i],ind[4][i]]
    return out,out2

  def dict_vec_divideby_DSM(s,vec, ind=-1):
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
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=s.nonzero().T
    Ni=len(indices[0])
    j=-1
    x,y,z=indices[0][0:3]
    for l in range(0,Ni):
      outDSM[indices[l][0],indices[l][1],indices[l][2]][indices[l][3],indices[l][4]]=np.divide(vec[indices[l][3]],s[indices[l][0],indices[l][1],indices[l][2]][indices[l][3],indices[l][4]])
    return outDSM
  ## Save the DSM s.
  # @param filename_ the name of the file to save to.
  # @return nothing
  def save_dict(s, filename_):
    """ Save the DSM s.

    :meta private:

    :param filename_: the name of the file to save to.

    :return: nothing

    """
    for x,y,z in s.d.keys():
      filename_out=filename_+'%02dx%02dy%02dz'%(x,y,z)
      out=s.d[x,y,z].tocsr()
      save_npz(filename_out+'.npz',out)
        #print('Program continued by Mesh is not saved')
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

    :meta public:

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
    check=-1
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      indicesM=s.d[x,y,z].nonzero()
      NI=len(indicesM[0])
      if abs(NI)<epsilon:
        pass
      else:
        if check==-1:
          check=0
          indices=np.array([x,y,z,indicesM[0][0],indicesM[1][0]])
          indicesSec=np.c_[np.tile(np.array([x,y,z]),(NI-1,1)),indicesM[0][1:],indicesM[1][1:]]
          indices=np.vstack((indices,indicesSec))
        else:
          indicesSec=np.c_[np.tile(np.array([x,y,z]),(NI,1)),indicesM[0][0:],indicesM[1][0:]]
          indices=np.vstack((indices,indicesSec))
    if check==-1:
      indices=np.array([])
    return indices
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
  def stopcheck(s,i,j,k):
    """ Check if the index [i,j,k] is valid.

    :param i: is the index for the x axis.

    :param j: is the index for the y axis.

    :param k: is the index for the z axis.

    :param p1: is the point at the end of the ray.

    :param h: is the mesh width

    :return: 1 if valid, 0 if not.

    .. todo:: add the inside check to this function

    .. todo:: add the check for the end of the ray.

    :warning: This currently only checks a point is \
    inside a room, it doesn't account for if you have gone inside an object.

    """
    #FIXME add the inside check to this function
    #FIXME add the check for the end of the ray.
    #if i>=p1[0] and j>=p1[1] and k>=p1[2]:
    #  return 0
    return (0<=int(i)<s.Nx and 0<=int(j)<s.Ny and 0<=int(k)<s.Nz)
  def stopchecklist(s,ps,p3,n):
    """ Check if the list of points is valid.

    :param ps: the indices for the points in the list

    :param p1: the end of the ray

    :param h: the meshwidth

    :param p3: the points on the cone vectors

    :param n: the normal vectors forming the cone.

    * start=0 if no points were valid.
    * if at least 1 point was valid,

      * ps=[[i1,j1,k1],...,[in,jn,kn]] the indices of the valid points,
      * p3=[[x1,y1,z1],...,[xn,yn,zn]] co-ordinates of the valid points,
      * N=[n0,...,nN] the normal vectors corresponding to the valid points.

    :return: start, ps, p3, N

    """
    start=0
    newps=np.array([])
    newp3=np.array([])
    newn =np.array([])
    j=0
    if isinstance(ps[0],(float,int,np.int64, np.complex128 )):
      check=s.stopcheck(ps[0],ps[1],ps[2])
      if check==1:
        newps=np.array([ps[0],ps[1],ps[2]])
        newp3=np.array([p3[0,0],p3[0,1],p3[0,2]])
        newn =np.array([n[0,0],n[0,1],n[0,2]])
      else:
        pass
    else:
     for k in ps:
      check=s.stopcheck(k[0],k[1],k[2])
      if check==1:
        if start==0:
          newps=np.array([k[0],k[1],k[2]])
          newp3=np.array([p3[j,0],p3[j,1],p3[j,2]])
          newn =np.array([n[j,0], n[j,1],n[j,2]])
          start=1
        else:
          newps=np.vstack((newps,np.array([k[0],k[1],k[2]])))
          newp3=np.vstack((newp3,np.array([p3[j,0],p3[j,1],p3[j,2]])))
          newn =np.vstack((newn, np.array([n[j,0],n[j,1], n[j,2]])))
      else:
        pass
      j+=1
    return start, newps, newp3, newn

#=======================================================================
# FUNCTIONS CONNECTED TO DS BUT AREN'T PART OF THE OBJECT
#=======================================================================
def Watts_to_db(P):
  return 10*np.log10(P,where=(P!=0))

def db_to_Watts(P):
  nz=np.nonzero(P)
  out=np.zeros(P.shape)
  out[nz]=10**(0.1*P[nz])
  return out

def Correct_ObNumbers(rvec,Ntri):
    '''Take in the triangle position and output the position of the first triangle which lies on that surface.
    :param rvec: The positions in the obstacle list of the triangles
    :param Ntri: The number of triangles that form each surface.'''
    n=len(rvec)
    rvecout=np.zeros(n)
    Nsur=len(Ntri)
    count=0
    for j in range(n):
      if rvec[j]!=0:
        for i in range(Nsur):
          if rvec[j]-Ntri[i]-count>0:
            count+=Ntri[i]
          else:
            rvecout[j]=count+1
            break
    return rvecout

def Correct_test():
    Nob=12
    Ntri=2*np.ones(int(Nob*0.5))
    row=np.arange(12)
    return Correct_ObNumbers(row,Ntri)

def stopcheck(i,j,k,Nx,Ny,Nz):
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
    if i>Nx-1 or j>Ny-1 or k>Nz-1 or i<0 or j<0 or k<0:
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
def stopchecklist(ps,p3,h,Nx,Ny,Nz):
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
      i1,j1,k1=k//h
      check=stopcheck(i1,j1,k1,Nx,Ny,Nz)
      if start==1:
        if k[0] in newps[0]:
          kindex=np.where(newp3[0]==k[0])
          if k[1]==newps[1][kindex] and k[2]==newps[2][kindex]:
            check==0
      if check==1:
        if start==0:
          newps=np.array([k[0],k[1],k[2]])
          newp3=np.array([p3[j][0],p3[j][1],p3[j][2]])
          newn =np.array([n[j][0], n[j][1], n[j][2]])
          start=1
        else:
          newps=np.vstack((newps,np.array([k[0],k[1],k[2]])))
          newp3=np.vstack((newp3,np.array([p3[j][0],p3[j][1],p3[j][2]])))
          newn =np.vstack((newn, np.array([n[j][0],n[j][1], n[j][2]])))
      else:
        pass
      j+=1
    return start, newps, newp3, newn
def optimum_gains(foldtype,plottype,Mesh,room,Znobrat,refindex,Antpar, Pol,Nra,Nre,job,index,LOS=0,PerfRef=0,ind=-1):
  ''' Compute the optimal transmitter gains from a Mesh of ray information and the physical \
  parameters.

  :param Mesh:    The :py:class:`DS` mesh of ray information. \
  Containing the non-dimensionalised ray lengths and their angles of reflection.
  :param Znobrat: An Nsur x Nre+1 array containing tiles of the impedance \
  of obstacles divided by the impedance of air.
  :param refindex: An Nsur x Nre+1 array containing tiles of the refractive\
  index of obstacles.
  :param Antpar: Numpy array containing the wavenumber, wavelength and lengthscale.
  :param Gt:     Array of the transmitting antenna gains.
  :param Pol:    2x1 numpy array containing the polarisation terms.
  :param Nra:    The number of rays in the ray tracer.
  :param Nre:    The number of reflections in the ray tracer.
  :param Ns:     The number of terms on each axis.
  :param LOS:    Line of sight, 1 for yes 0 for no, default is 0.
  :param ind:    The non-zero indices of the Mesh, default is -1, then \
  the indices are found after a check.

  Method:

    * First compute the angles of reflection using py:func:`Mesh.sparse_angles()`.\
    If the angles are already saved from previous calculations then they are loaded.
    * Compute the combined reflection coefficients for the parallel and \
    perpendicular to polarisation terms using \
    :py:class:`DS`.:py:func:`Mesh.refcoefbyterm_withmu(Nre,refindex,LOS=0,PerfRef=0, ind=-1)`=Comper,Compar.
    * Extract the distance each ray travelled using \
    :py:class:`DS`. :py:func:`__get_rad__()`
    * Multiply by the gains for the corresponding ray, the phase, \
    the combined reflection coefficients and divide by the distance the \
    ray travelled to get the power in the different polarisation directions.\
    Using :py:class:`DS`.:py:func:`gain_phase_rad_ref_mul_add(Comper,Compar,Gt,khat,L,lam,ind)`
    * Multiply by initial polarisation vectors and combine.
    * Ignore dividing by initial phi as when converting to power in db \
    this disappears.
    * Take the amplitude and square.

  :rtype: Nx x Ny x Nz numpy array of real values.

  :return: Grid

  '''
  #print('----------------------------------------------------------')
  #print('Start computing the power from the Mesh')
  #print('----------------------------------------------------------')
  t0=t.time()
  # Retrieve the parameters
  khat,lam,L = Antpar # khat is the non-dimensional wave number,
                      # lam is the non-dimensionalised wave length.
                      # L is the length scale for the dimensions.
  # Check in the nonzero indices have been input or not, if not then find them.
  #print('power start')
  Nx,Ny,Nz=Mesh.Nx,Mesh.Ny,Mesh.Nz
  na,nb=Mesh.shape[0],Mesh.shape[1]
  if isinstance(ind, type(-1)):
    ind=Mesh.nonzero().T
    indout=ind
  else:
    ind=ind.T
    indout=ind
  Ns=max(Nx,Ny,Nz)
  h=1.0/Ns
  powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nra,Nre,Ns)
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nra,Nre,Ns)
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
    os.makedirs('./Mesh/'+foldtype)
    os.makedirs(meshfolder)
    os.makedirs('./Mesh'+plottype)
    os.makedirs(powerfolder)
  if not os.path.exists('./Mesh/'+foldtype):
    os.makedirs('./Mesh/'+foldtype)
    os.makedirs(meshfolder)
  if not os.path.exists(meshfolder):
    os.makedirs(meshfolder)
  if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
      os.makedirs(powerfolder)
  if not os.path.exists(powerfolder):
      os.makedirs(powerfolder)
  if room.Nsur>6:
    boxstr='Box'
  else:
    boxstr='NoBox'
  # Check if the reflections angles are saved, if not then find them.
  angstr='ang%03dRefs%03dNs%0d_tx%03d'%(Nra,Nre,Ns,job)
  angfile=meshfolder+'/'+boxstr+angstr
  angeg=angfile+'%02dx%02dy%02dz.npz'%(0,0,0)
  if newvar:
    AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
    AngDSM.save_dict(angfile)
  else:
    if os.path.isfile(angeg):
      AngDSM=load_dict(angfile,Nx,Ny,Nz)
    else:
      AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
      AngDSM.save_dict(angfile)
  Comper,Compar=AngDSM.refcoefbyterm_withmul(Znobrat,refindex,LOS,PerfRef,ind)
  Realper,Imageper=Comper.image_real_parts(ind)
  Realpar,Imagepar=Compar.image_real_parts(ind)
  for x,y,z in product(range(Nx),range(Ny),range(Nz)):
      n=Comper[x,y,z].getnnz()
      inp=Comper[x,y,z].nonzero()
      for l in range(n):
        b=inp[1][l]
        a=AngDSM[x,y,z][:,b].nonzero()[0][-1]
        if abs(Comper[x,y,z][a,b]-1)<epsilon:
          AngDSM[x,y,z][a,b]=0
  if not LOS:
    #print(Mesh)
    Theta=AngDSM.togrid(ind)
    np.save(powerfolder+'/'+boxstr+'AngNpy%03dRefs%03dNs%03d_tx%03d.npy'%(Nra,Nre,Ns,job),Theta)
  rstr='rad%dRefs%dNs%d'%(Nra,Nre,Ns)
  rfile=meshfolder+'/'+boxstr+rstr+'_tx%03d'%(job)
  reg=rfile+'%02dx%02dy%02dz.npy'%(0,0,0)
  Nob=room.Nob
  Nsur=room.Nsur
  if newvar:
    RadMesh,ind=Mesh.__get_rad__(Nsur,ind,foldtype,Nra,Nre,boxstr,index,job)
    RadMesh.save_dict(rfile)
  else:
    if os.path.isfile(reg):
      RadMesh=load_dict(rfile,Nx,Ny,Nz)
      ind=RadMesh.nonzero()
    else:
      RadMesh,ind=Mesh.__get_rad__(Nsur,ind,foldtype,Nra,Nre,boxstr,index,job)
      RadMesh.save_dict(rfile)
  t4=t.time()
  Hx,Fx=RadMesh.opti_func_mats(Realper,Realpar,Imageper,Imagepar,khat,L,lam,Pol,Nra,ind)
  Gt=Hx.opti_combo_inverse(Fx,Nra)
  Gt=np.power(Gt,2)
  return Gt
def power_compute(foldtype,plottype,Mesh,room,Znobrat,refindex,Antpar,Gt, Pol,Nra,Nre,job,index,LOS=0,PerfRef=0,ind=-1):
  ''' Compute the field from a Mesh of ray information and the physical \
  parameters.

  :param Mesh:    The :py:class:`DS` mesh of ray information. \
  Containing the non-dimensionalised ray lengths and their angles of reflection.
  :param Znobrat: An Nsur x Nre+1 array containing tiles of the impedance \
  of obstacles divided by the impedance of air.
  :param refindex: An Nsur x Nre+1 array containing tiles of the refractive\
  index of obstacles.
  :param Antpar: Numpy array containing the wavenumber, wavelength and lengthscale.
  :param Gt:     Array of the transmitting antenna gains.
  :param Pol:    2x1 numpy array containing the polarisation terms.
  :param Nra:    The number of rays in the ray tracer.
  :param Nre:    The number of reflections in the ray tracer.
  :param Ns:     The number of terms on each axis.
  :param LOS:    Line of sight, 1 for yes 0 for no, default is 0.
  :param ind:    The non-zero indices of the Mesh, default is -1, then \
  the indices are found after a check.

  Method:

    * First compute the angles of reflection using py:func:`Mesh.sparse_angles()`.\
    If the angles are already saved from previous calculations then they are loaded.
    * Compute the combined reflection coefficients for the parallel and \
    perpendicular to polarisation terms using \
    :py:class:`DS`.:py:func:`Mesh.refcoefbyterm_withmu(Nre,refindex,LOS=0,PerfRef=0, ind=-1)`=Comper,Compar.
    * Extract the distance each ray travelled using \
    :py:class:`DS`. :py:func:`__get_rad__()`
    * Multiply by the gains for the corresponding ray, the phase, \
    the combined reflection coefficients and divide by the distance the \
    ray travelled to get the power in the different polarisation directions.\
    Using :py:class:`DS`.:py:func:`gain_phase_rad_ref_mul_add(Comper,Compar,Gt,khat,L,lam,ind)`
    * Multiply by initial polarisation vectors and combine.
    * Ignore dividing by initial phi as when converting to power in db \
    this disappears.
    * Take the amplitude and square.
    * Take :math:`10log10()` to get the db Power.

  :rtype: Nx x Ny x Nz numpy array of real values.

  :return: Grid

  '''
  #print('----------------------------------------------------------')
  #print('Start computing the power from the Mesh')
  #print('----------------------------------------------------------')
  t0=t.time()
  # Retrieve the parameters
  khat,lam,L = Antpar # khat is the non-dimensional wave number,
                      # lam is the non-dimensionalised wave length.
                      # L is the length scale for the dimensions.
  # Check in the nonzero indices have been input or not, if not then find them.
  #print('power start')
  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  na=Mesh.shape[0]
  nb=Mesh.shape[1]
  if isinstance(ind, type(-1)):
    ind=Mesh.nonzero().T
    indout=ind
  else:
    ind=ind.T
    indout=ind
  Ns=max(Nx,Ny,Nz)
  h=1.0/Ns
  if room.Nsur>6:
    boxstr='Box'
  else:
    boxstr='NoBox'
  powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nra,Nre,Ns)
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nra,Nre,Ns)
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
    os.makedirs('./Mesh/'+foldtype)
    os.makedirs(meshfolder)
    os.makedirs('./Mesh/'+plottype)
    os.makedirs(powerfolder)
  if not os.path.exists('./Mesh/'+foldtype):
    os.makedirs('./Mesh/'+foldtype)
    os.makedirs(meshfolder)
  if not os.path.exists(meshfolder):
    os.makedirs(meshfolder)
  if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
      os.makedirs(powerfolder)
  if not os.path.exists(powerfolder):
      os.makedirs(powerfolder)
  # Check if the reflections angles are saved, if not then find them.
  angstr='ang%03dRefs%03dNs%0d_tx%03d'%(Nra,Nre,Ns,job)
  angfile=meshfolder+'/'+boxstr+angstr
  angeg=angfile+'%02dx%02dy%02dz.npz'%(0,0,0)
  if newvar:
    AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
    AngDSM.save_dict(angfile)
  else:
    if os.path.isfile(angeg):
      AngDSM=load_dict(angfile,Nx,Ny,Nz)
    else:
      AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
      AngDSM.save_dict(angfile)
  Comper,Compar=AngDSM.refcoefbyterm_withmul(Znobrat,refindex,LOS,PerfRef,ind)
  for x,y,z in product(range(Nx),range(Ny),range(Nz)):
      #AngNpy[x,y,z]=AngDSM[x,y,z].multiply(Comper[x,y,z])
      n=Comper[x,y,z].getnnz()
      inp=Comper[x,y,z].nonzero()
      for l in range(n):
        b=inp[1][l]
        a=AngDSM[x,y,z][:,b].nonzero()[0][-1]
        if abs(Comper[x,y,z][a,b]-1)<epsilon:
          AngDSM[x,y,z][a,b]=0
  if dbg:
    if LOS==1:
      for x,y,z in product(range(Nx),range(Ny),range(Nz)):
        if Comper[x,y,z].getnnz()>1 or Compar[x,y,z].getnnz()>1:
          errmsg='Checking LOS case but more than one term is in the reflection matrix'
          logging.info('Position (%d,%d,%d), number of nonzero terms per %d, par %d'%(x,y,z,Comper[x,y,z].getnnz(),Compar[x,y,z].getnnz()))
          logging.error('Comper '+str(Comper[x,y,z])+' Compar  '+str(Compar[x,y,z]))
          raise ValueError(errmsg)
    Nsur=int((np.count_nonzero(refindex)-1)*0.5)# Each planar surface is formed of two triangles
    if LOS==0:
      Maxnonzero=(Nsur**(Nre+1)-1)/(Nsur-1)
      for x,y,z in product(range(Mesh.Nx),range(Mesh.Ny),range(Mesh.Nz)):
        if Comper[x,y,z].getnnz()>Maxnonzero or Compar[x,y,z].getnnz()>Maxnonzero:
          pdb.set_trace()
          errmsg='Checking reflection case and more than %d terms is in the reflection matrix'%(Maxnonzero)
          print('Position (%d,%d,%d)'%(x,y,z))
          print('Number of terms in reflection coefficient matrices',Comper[x,y,z].getnnz(),Compar[x,y,z].getnnz())
          raise ValueError(errmsg)
  if not LOS:
    #print(Mesh)
    Theta=AngDSM.togrid(ind)
    np.save(powerfolder+'/'+boxstr+'AngNpy%03dRefs%03dNs%03d_tx%03d.npy'%(Nra,Nre,Ns,job),Theta)
  rstr='rad%dRefs%dNs%d'%(Nra,Nre,Ns)
  rfile=meshfolder+'/'+boxstr+rstr+'_tx%03d'%(job)
  reg=rfile+'%02dx%02dy%02dz.npy'%(0,0,0)
  Nob=room.Nob
  Nsur=room.Nsur
  if newvar:
    RadMesh,ind=Mesh.__get_rad__(Nsur,ind,foldtype,Nra,Nre,boxstr,index,job)
    RadMesh.save_dict(rfile)
  else:
    if os.path.isfile(reg):
      RadMesh=load_dict(rfile,Nx,Ny,Nz)
      ind=RadMesh.nonzero()
    else:
      RadMesh,ind=Mesh.__get_rad__(Nsur,ind,foldtype,Nra,Nre,boxstr,index,job)
      RadMesh.save_dict(rfile)
  t4=t.time()
  Gridpe, Gridpa=RadMesh.gain_phase_rad_ref_mul_add(Comper,Compar,Gt,khat,L,lam,Nra,ind)
  P=np.absolute(Gridpe*Pol[0])**2+np.absolute(Gridpa*Pol[1])**2
  P=Watts_to_db(P)
  # if dbg:
    # TrGrid=np.load('./Mesh/True/'+plottype+'/True.npy')
    # for x,y,z in product(range(Mesh.Nx),range(Mesh.Ny),range(Mesh.Nz)):
      # if abs(TrGrid[x,y,z]-P[x,y,z])>10**4*epsilon:
        # pdb.set_trace()
        # #pass
  return P,indout


def quality_compute(foldtype,plottype,Mesh,Grid,room,Znobrat,refindex,Antpar,Gt, Pol,Nra,Nre,job,index,LOS,PerfRef,ind=-1):
  ''' Compute the field from a Mesh of ray information and the physical \
  parameters.

  :param Mesh: The :py:class:`DS` mesh of ray information.
  :param Znobrat: An Nsur x Nre+1 array containing tiles of the impedance \
    of obstacles divided by the impedance of air.
  :param refindex: An Nsur x Nre+1 array containing tiles of the refractive\
    index of obstacles.
  :param Antpar: Numpy array containing the wavenumber, wavelength and lengthscale.
  :param Gt: Array of the transmitting antenna gains.
  :param Pol: 2x1 numpy array containing the polarisation terms.

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
  #print('----------------------------------------------------------')
  #print('Start computing the power from the Mesh')
  #print('----------------------------------------------------------')
  t0=t.time()
  # Retrieve the parameters
  khat,lam,L = Antpar
  if isinstance(ind, type(-1)):
    ind=Mesh.nonzero().T
    indout=ind
  else:
    ind=ind.T
    indout=ind
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  Nra=int(Nra)
  Nre=int(Nre)
  Ns=max(Mesh.Nx,Mesh.Ny,Mesh.Nz)
  h=1.0/Ns
  if room.Nsur>6:
    boxstr='Box'
  else:
    boxstr='NoBox'
  powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nra,Nre,Ns)
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nra,Nre,Ns)
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
    os.makedirs('./Mesh/'+foldtype)
    os.makedirs(meshfolder)
    os.makedirs('./Mesh/'+plottype)
    os.makedirs(powerfolder)
  if not os.path.exists('./Mesh/'+foldtype):
    os.makedirs('./Mesh/'+foldtype)
    os.makedirs(meshfolder)
  if not os.path.exists(meshfolder):
    os.makedirs(meshfolder)
  if not os.path.exists('./Mesh/'+plottype):
      os.makedirs('./Mesh/'+plottype)
      os.makedirs(powerfolder)
  if not os.path.exists(powerfolder):
      os.makedirs(powerfolder)
  # Check if the reflections angles are saved, if not then find them.
  angstr='ang%03dRefs%03dNs%0d_tx%03d'%(Nra,Nre,Ns,job)
  angfile=meshfolder+'/'+boxstr+angstr
  angeg=angfile+'%02dx%02dy%02dz.npz'%(0,0,0)
  AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
  AngDSM.save_dict(angfile)
  Comper,Compar=AngDSM.refcoefbyterm_withmul(Znobrat,refindex,LOS,PerfRef,ind)
  rfile=meshfolder+'/rad%dRefs%dNs%d'%(Nra,Nre,Ns)
  rstr='rad%dRefs%dNs%d'%(Nra,Nre,Ns)
  rfile=meshfolder+'/'+boxstr+rstr+'_tx%03d'%(job)
  reg=rfile+'%02dx%02dy%02dz.npy'%(0,0,0)
  Nob=room.Nob
  Nsur=room.Nsur
  RadMesh,ind=Mesh.__get_rad__(Nsur,ind,foldtype,Nra,Nre,boxstr,index,job)
  RadMesh.save_dict(rfile)
  t4=t.time()
  Gridpe, Gridpa=RadMesh.gain_phase_rad_ref_mul_add(Comper,Compar,Gt,khat,L,lam,Nra,ind)
  P=np.zeros((Mesh.Nx,Mesh.Ny,Mesh.Nz),dtype=np.longdouble)
  P=np.absolute(Gridpe*Pol[0])**2+np.absolute(Gridpa*Pol[1])**2
  P=Watts_to_db(P)
  Q=QualityFromPower(P)
  return Q,ind

def nob_fromrow(r,Nob):
  if r ==0: return 0
  elif Nob==0:
    raise ValueError('The number of obstacles has been input as 0')
  else: return (r-1)%Nob

def nre_fromrow(r,Nob):
  if r==0:
    return 0
  elif Nob==0:
    raise ValueError('The number of obstacles has been input as 0')
  else:
    nob=nob_fromrow(r,Nob)
    return int(((r-nob-1)/Nob)+1)

def FieldEquation(r,khat,L,lam):
  ''' The equation for calculating the field.

  :param r: The distance travelled.
  :param khat: Non-dimensionalised wave number.
  :param lam: wavelength
  :param L: Length scale
  :param Pol: Polarisation

  :rtype: Array with dimensions of Pol
  :return: :math:`(lam/(4*ma.pi*r))*np.exp(1j*khat*r*(L**2))*Pol`
  '''
  return (lam/(4.0*np.pi*r))*np.exp(1j*khat*r*(L**2))

def QualityFromPower(P):
   '''Calculate the quality of coverage from Power.
   :param P: Power as a Nx x Ny x Nz array

   :rtype: float

   :returns: :math:`sum(P)/Nx*Ny*Nz`
   '''
   return np.sum(P)/(P.shape[0]*P.shape[1]*P.shape[2])

def QualityPercentileFromPower(P):
   '''Calculate the quality of coverage from Power.
   :param P: Power as a Nx x Ny x Nz array

   :rtype: float

   :returns: the 10th percentile of P
   '''
   return np.percentile(P, 10)

def QualityMinFromPower(P):
   '''Calculate the quality of coverage from Power.
   :param P: Power as a Nx x Ny x Nz array

   :rtype: float

   :returns: the 10th percentile of P
   '''
   return np.amin(P)

def nonzero_bycol(SM):
  ''' Find the index pairs for the nonzero terms in a sparse matrix.
  Go through each column and find the nonzero rows.

  :param SM: sparse matrix.

  :return: [[i(0j0),i(1j0),...,i(nj0),...,i(njn)],\
    [j0,...,j0,...,jn,...,jn]]
  '''
  inddum=SM.transpose().nonzero()
  ind=[inddum[1],inddum[0]]
  return ind

def singletype(x):
  if isinstance(x,(float,int,np.int32,np.int64, np.complex128 )):
    return True
  else: return False

def load_dict(filename_,Nx=0,Ny=0,Nz=0):
  ''' Load a DS as a dictionary and construct the DS again.

  :param filename_: the name of the DS saved

  .. code::

     Nx=max(Keys[0])-min(Keys[0])
     Ny=max(Keys[1])-min(Keys[1])
     Nz=max(Keys[2])-min(Keys[2])

  :returns: nothing

  '''
  out=load_npz(filename_+'%02dx%02dy%02dz.npz'%(0,0,0))
  na=out.shape[0]
  nb=out.shape[1]
  outDS=DS(Nx,Ny,Nz,na,nb)
  outDS.__setitem__((0,0,0),out.todok())
  for x,y,z in product(range(0,Nx),range(0,Ny),range(0,Nz)):
    if x==0 and y==0 and z==0:
      continue
    filename_out=filename_+'%02dx%02dy%02dz'%(x,y,z)
    out=load_npz(filename_out+'.npz')
    outDS.__setitem__((x,y,z),out.todok())
  return outDS

def ref_coef(Mesh,Znobrat,refindex,Nra,Nre,Ns,ind=-1):
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
  # FIXME rewrite this whole section so loops aren't repeated.
  t0=t.time()
  #print('----------------------------------------------------------')
  #print('Retrieving the angles of reflection')
  #print('----------------------------------------------------------')
  #print('Getting nonzero indices')
  if isinstance(ind, type(-1)):
    ind=Mesh.nonzero()
  else:
    pass
  t1=t.time()
  #print('Make DS for angles')
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  angfile='./Mesh/ang%dRefs%dNs%d'%(Nra,Nre,Ns)
  try:
    AngDSM=load_dict(angfile,Nx,Ny,Nz)
  except:
    AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
    AngDSM.save_dict(angfile)
  t2=t.time()
  AngDSM=Mesh.sparse_angles(ind)
  t3=t.time()
  # cthi=DS(Mesh.Nx,Mesh.Ny,Mesh.Nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta)
  # ctht=DS(Mesh.Nx,Mesh.Ny,Mesh.Nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta_t) #FIXME
  # t3=t.time()
  # cthfile=str('./Mesh/cthi'+str(int(Nra))+'Refs'+str(int(Nre))+'Ns'+str(int(Ns))+'.npy')
  # cfile=Path(cthfile)
  # #cthtfile=str('./Mesh/ctht'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  # if cfile.is_file():
    # cthi=load_dict(cthfile)
  # else:
    # cthi=AngDSM.cos(ind)                                   # Compute cos(theta_i)
    # cthi.save_dict(cthfile)
  # t4=t.time()
  # ctht=AngDSM.costhetat(refindex,ind)
  # t5=t.time()
  #print('----------------------------------------------------------')
  #print('Multiplying by the impedances and computing ratio for refcoef')
  #print('----------------------------------------------------------')
  ind=AngDSM.nonzero()
  Rper,Rpar=AngDSM.refcoefbyterm(Znobrat,refindex,ind)
  t6=t.time()
  print('----------------------------------------------------------')
  print('Reflection coefficients found, time taken ', t6-t2)
  # print('Time computing costhetat', t5-t4)
  # print('Time computing cos thetai', t4-t3)
  # print('Time initialising cthi ctht SIN', t3-t2)
  print('Time getting angles from mesh', t2-t1)
  print('Time getting nonzero indices', t1-t0)
  print('----------------------------------------------------------')
  print('Total time to compute reflection coefficients from Mesh ', t6-t0)
  print('----------------------------------------------------------')
  return Rper, Rpar, ind

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
  return DSM.cos(indices).__eq__(CosAngM)

## Attempt to find angle of nonzero element of SM inside dictionary
def test_13():
  Nx=2
  Ny=2
  Nz=1
  na=3
  nb=3
  DSM=test_03c(Nx,Ny,Nz,na,nb)
  ang=sparse_angles(DSM)
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
  Znob=np.tile(Znob,Nre)                      # The number of rows is Nob*Nre+1. Repeat Nob
  Znob=np.insert(Znob,0,complex(0.0,0.0))     # Use a zero for placement in the LOS row
  #Znob=np.transpose(np.tile(Znob,(Nb,1)))    # Tile the obstacle coefficient number to be the same size as a mesh array.
  Znobrat=Znob/Z0
  refindex=np.sqrt(np.multiply(mur,epsr))     # Refractive index of the obstacles
  refindex=np.tile(refindex,Nre)
  refindex=np.insert(refindex,0,complex(0,0))

  Rper,Rpar=ref_coef(ds,Znobrat,refindex)
  Comper=Rpre.dict_col_mult_()
  Compar=Rpar.dict_col_mult_()
  Comper2,Compar2=ds.refcoefbyterm_withmul(Znobrat,refindex)
  return Compar.__eq__(Compar2),Comper.__eq__(Comper2)

def test_15():
  ''' Testing multiplying nonzero terms in columns '''
  DS=test_14()
  out=DS.dict_col_mult()
  return out

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
  ds=test_03c( Nx , Ny , Nz , na , nb )         # test_03() initialises a
                                               # DSM with values on the
                                               # diagonal of each mesh element
  filename='testDS'
  ds.save_dict(filename)
  ds=load_dict(filename,Nx,Ny,Nz)
  return Nx,Ny,Nz

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
      if D2[x,y,x,0,j] != np.math.factorial(j+1):
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
    col=int(nb*0.5)
    for j in range(na):
      Mesh[x,y,z,j,col]=count*vec[j]
      count+=1
  if Mesh.__self_eq__():
    #print(Mesh) #DEBUG
    return 1
  count=0
  Mesh=DS(Nx,Ny,Nz,na,nb)
  for x,y,z in product(range(Nx),range(Ny),range(Nz)):
    col=int(nb*0.5)
    Mesh[x,y,z,:,col]=count*vec
    count+=1
  if Mesh.__self_eq__():
    #print(Mesh)
    return 1
  else:
    return 0

def test_24():
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Ns             =np.load('Parameters/Ns.npy')
  Nra=int(Nra)
  Nre=int(Nre)
  Nob            =np.load('Parameters/Nob.npy')

  #PI.ObstacleCoefficients()
  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/TxGains.npy')
  freq          = np.load('Parameters/frequency.npy')
  Freespace     = np.load('Parameters/Freespace.npy')
  c             =Freespace[3]
  khat          =freq*L/c
  lam           =(2*np.pi*c)/freq
  Antpar        =np.array([khat,lam,L])

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat.npy')
  refindex     =np.load('Parameters/refindex.npy')

  ##----Retrieve the Mesh--------------------------------------
  meshname='DSM%dRefs%dm'(Nra,Nre)
  Mesh= load_dict(meshname,Ns,Ns,Ns)
  ind=Mesh.nonzero()
  AngDSM=Mesh.sparse_angles()
  return AngDSM

def doubles_in_test():


  ##----Retrieve the Mesh--------------------------------------
  Ns=np.load('Parameters/Ns.npy')
  meshname='testDS'
  Mesh= load_dict(meshname,5,5,10)
  newtime=0
  oldtime=0
  Avenum=500
  Avetot=0
  Nsur=np.load('Parameters/Nsur.npy')
  Nre,_,_,_    =np.load('Parameters/Raytracing.npy')
  rn=Mesh.shape[0]
  cn=Mesh.shape[1]
  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  h=1.0/max(Nx,Ny,Nz)
  for j in range(Avenum):
    vec=SM((rn,1),dtype=np.complex128)
    po=np.array([np.random.randint(Nx),np.random.randint(Ny),np.random.randint(Nz)])
    for i in range(int(Nre)):
      r=np.random.randint(0,rn)
      vec[r]=np.random.rand()
    if j%5==1:
      c=np.random.randint(0,cn)
      vec=Mesh[po[0],po[1],po[2]][:,c]
    t0=t.time()
    doub=Mesh.doubles__inMat__(vec,po)
    if j%5==1 and len(vec.nonzero()[0]>0):
      assert doub
    t1=t.time()
    #print('after')
    #print(altcoposout,copos2out,p3out,normout)
    newtime+=t1-t0
    Avetot+=1
  print('new method time',newtime/(Avetot))
    # The 'new' method is slower so the old method is kept
  return 0


if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  #job_server = pp.Server()
  test_18()
  doubles_in_test()
  exit()

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
from pathlib import Path
import timeit
import os
#from collections import defaultdict

epsilon=sys.float_info.epsilon
#----------------------------------------------------------------------
# NOTATION IN COMMENTS
#----------------------------------------------------------------------
# dk is dictionary key, smk is sparse matrix key, SM is a sparse matrix
# DS or DSM is a DS object which is a dictionary of sparse matrices.

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
  def __init__(s,Nx=1,Ny=1,Nz=1,na=1,nb=1,dt=np.complex128):
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
  def __get_SM__(s,smk,dk,n):
    ''' Get a SM at the position dk=[x,y,z].
    * n indicates whether a whole SM is set, a row or a column.
    * If n==0 a whole SM.
    * If n==1 a row or rows.
      * n2 is the number of rows.
    * If n==2 a column or columns.
      * n2 is the number of columns.
    '''
    #out=SM(s.shape,dtype=np.complex128)
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
        p=0
        nkeys=len(smk)
        nb=s.shape[1]
        out=np.zeros((nkeys,nb),dtype=dt)
        for j in smk:
          out[j,:]=s.d[dk][j,:]
          p+=1
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
  def __mul__(s, DSM2,ind=-1):
    ''' Multiply DSM2 with s
    Perform matrix multiplication AB for all sparse matrices A in s and \
    B in DSM2 with the same key (x,y,z)

    :param DSM2: is a DSM with the same dimensions as s
    :returns: s[x,y,z]DSM2[x,y,z] :math: `\\forall x, y, z, \in [0,Nx], [0,Ny], [0,Nz]`
    :rtype: a new DSM with the same dimensions
    '''
    out=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    if isinstance(ind, type(-1)):
      for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
        out[x,y,z]=s[x,y,z].multiply(DSM2[x,y,z])
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
      for i in range(n):
        x=ind[0][i]
        y=ind[1][i]
        z=ind[2][i]
        if x==p and y==l and z==m:
          pass
        else:
          p=x
          l=y
          m=z
          out[x,y,z]=s[x,y,z].multiply(DSM2[x,y,z])
    return out
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
  # Rper=(S1-ctht).__truediv__(S1+ctht,ind)     # Compute the Reflection coeficient perpendicular
                                              # # to the polarisiation Rper=(Zm/Z0cos(theta_i)-cos(theta_t))/(Zm/Z0cos(theta_i)+cos(theta_t))
  # Rpar=(cthi-S2).__truediv__(cthi+S2,ind)     # Compute the Reflection coeficient parallel

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
  def __get_rad__(s, h,Nra,ind=-1):
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
      l=abs(s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]])
      M=out[ind[0][i],ind[1][i],ind[2][i]]
      indM=M.nonzero()
      n2=len(indM[0])
      rep=1
      if np.sum(s[ind[0][i],ind[1][i],ind[2][i],1:-1,ind[4][i]])==0 and l!=0:
        for j in range(0,n2):
          if np.sum(s[ind[0][i],ind[1][i],ind[2][i],1:-1,indM[1][j]])==0 and abs(M[indM[0][j],indM[1][j]]-l)<h/4 and indM[1][j]!=ind[4][i]:
           rep=0
           #repcol=indM[1][j]
          else:
            pass
      if rep==0:
        out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=0
        #out[ind[0][i],ind[1][i],ind[2][i],0,repcol]=0
      else:
        if M[0,ind[4][i]]==0:
          out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=l
          if check==0:
            indicesSec=np.array([ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]])
            indout=np.vstack((indout,indicesSec))
            del indicesSec
          else:
            check=0
            indout=np.array([ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]])
      if check==-1:
        indout=np.array([])
    return out,indout
  def refcoefdiv(s,S2,cthi,ctht, ind=-1):
    if isinstance(ind, type(-1)):
      ind=ctht.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        pass
      else:
        ind=ctht.nonzero().T
    out1=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    out2=DS(s.Nx,s.Ny,s.Nz,s.shape[0],s.shape[1])
    n=len(ind[0])
    for i in range(n):
      x=ind[0][i]
      y=ind[1][i]
      z=ind[2][i]
      a=ind[3][i]
      b=ind[4][i]
      out1[x,y,z,a,b]=np.true_divide(s[x,y,z,a,b]-ctht[x,y,z,a,b],s[x,y,z,a,b]+ctht[x,y,z,a,b])
      out2[x,y,z,a,b]=np.true_divide(cthi[x,y,z,a,b]-S2[x,y,z,a,b],cthi[x,y,z,a,b]+S2[x,y,z,a,b])
    return out1,out2
  def refcoefmultdiv(s,ctht,m, ind=-1):
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
    for i in range(n):
      x=ind[0][i]
      y=ind[1][i]
      z=ind[2][i]
      a=ind[3][i]
      b=ind[4][i]
      out1[x,y,z,a,b]=np.true_divide(s[x,y,z,a,b]*m[a]-ctht[x,y,z,a,b],s[x,y,z,a,b]*m[a]+ctht[x,y,z,a,b])
      out2[x,y,z,a,b]=np.true_divide(s[x,y,z,a,b]-ctht[x,y,z,a,b]*m[a],s[x,y,z,a,b]+ctht[x,y,z,a,b]*m[a])
    return out1,out2
  def refcoefbyterm(s,m,refindex, ind=-1):
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
    for i in range(n):
      if ind[3][i]==0:
        out1[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]=1.0
        out2[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]=1.0
      else:
        x=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
        cthi=np.cos(x)
        ctht=np.sqrt(1-(np.sin(x)/refindex[ind[3][i]])**2)
        S1=cthi*m[ind[3][i]]
        S2=ctht*m[ind[3][i]]
        out1[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]=(S1-ctht)/(S1+ctht)
        out2[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]=(cthi-S2)/(cthi+S2)
    del cthi,ctht,x,n,S1,S2
    return out1,out2
  def refcoefbyterm_withmul(s,m,refindex,lam,L, ind=-1):
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
    for i in range(n):
      #if out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]==0:
       # if ind[3][i]==0:
        #  out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=(lam/(L*4*math.pi))
         # out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=(lam/(L*4*math.pi))
        #else:
         # pass
          # theta=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
          # cthi=np.cos(theta)
          # ctht=np.sqrt(1-(np.sin(theta)/refindex[ind[3][i]])**2)
          # S1=cthi*m[ind[3][i]]
          # S2=ctht*m[ind[3][i]]
          # out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=((S1-ctht)/(S1+ctht))*(lam/(L*4*math.pi))
          # out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=((cthi-S2)/(cthi+S2))*(lam/(L*4*math.pi))
      #else:
        if ind[3][i]==0 and np.sum(s[ind[0][i],ind[1][i],ind[2][i],1:-1,ind[4][i]])==0:
          out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=1
          out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=1
        elif ind[3][i]==0 and np.sum(s[ind[0][i],ind[1][i],ind[2][i],1:-1,ind[4][i]])!=0:
              out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=0
              out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=0
              #FIXME This is only to get just LOS
        else:
            theta=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
            cthi=np.cos(theta)
            ctht=np.cos(np.arcsin(np.sin(theta)/refindex[ind[3][i]]))
            #np.sqrt(1-(np.sin(theta)/refindex[ind[3][i]])**2)
            S1=cthi*m[ind[3][i]]
            S2=ctht*m[ind[3][i]]
            if cthi==0 and ctht==0:
              out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]*=0
              out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]*=0
            elif out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]==0:
              out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=0#(S1-ctht)/(S1+ctht)
              out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=0#(cthi-S2)/(cthi+S2)
            else:
              out1[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]*=0#(S1-ctht)/(S1+ctht)
              out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]*=0#(cthi-S2)/(cthi+S2)
    return out1,out2

  def togrid(s,ind):
    ''' Computethe matrix norm at each grid point and return a \
    3d numpy array.

    :rtype: Nx x Ny x Nz numpy array

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
    x,y,z=-1,-1,-1
    for i in range(0,n):
      if x==ind[0][i] and y==ind[1][i] and z==ind[2][i]:
        pass
      else:
        Grid[ind[0][i],ind[1][i],ind[2][i]]=s[ind[0][i],ind[1][i],ind[2][i]].sum()
        x,y,z=ind[0][i],ind[1][i],ind[2][i]
    return Grid

  def asin(s,ind=-1):
    """ Finds

    :math:`\\theta=\\arcsin(x)` for all terms :math:`x != 0` in \
    the DS s. Since all angles \
    :math:`\\theta` are in :math:`[0,\pi /2]`, \
    :math:`\\arcsin(x)` is not a problem.

    :returns: DSM with the same dimensions as s, with \
    :math:`\\arcsin(s)=\\theta` in \
     the same positions as the corresponding theta terms.

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
      asinDSM[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]]=np.arcsin(s[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]])
    return asinDSM
  ## Finds cos(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with cos(theta) in the
  # same position as the corresponding theta terms.
  def cos(s,ind=-1):
    """ Finds :math:`\\cos(\\theta)` for all terms \
    :math:`\\theta != 0` in the DS s.

    :returns: A DSM with the same dimensions with \
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
      CosDSM[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]]=np.cos(s[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]])
    return CosDSM
  def cos_asin(s,ind=-1):
    """ Finds :math:`\\cos( \\asin( \\theta))` for all terms \
    :math:`\\theta != 0` in the DS s.

    :returns: A DSM with the same dimensions with \
    :math:`\\cos( \\asin( \\theta))` in the \
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
    CosDSM=DS(s.Nx,s.Ny,s.Nz,na,nb)
    #ind=np.transpose(ind)
    n=len(ind[0])
    for j in range(0,n):
      CosDSM[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]]=np.cos(np.arcsin(s[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]]))
    return CosDSM
  ## Finds sin(theta) for all terms theta \!= 0 in DSM.
  # @return a DSM with the same dimensions with sin(theta) in the
  # same position as the corresponding theta terms.
  def sin(s,ind=-1):
    """ Finds :math:`\\sin(\\theta)` for all terms \
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
      SinDSM[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]]=np.sin(s[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]])
    return SinDSM
  ## Finds the angles theta which are the arguments of the nonzero
  # complex terms in the DSM s.
  # @return a DSM with the same dimensions with theta in the
  # same position as the corresponding complex terms.
  def sparse_angles(s,ind=-1):
    """ Finds the angles :math:`\\theta` which are the arguments \
    of the nonzero complex terms in the DSM s.

    :return: A DSM with the same dimensions with \
    :math:`\\theta` in the same \
     position as the corresponding complex terms.

    """
    #print('start angles')
    #t0=t.time()
    na,nb=s.shape
    AngDSM=DS(s.Nx,s.Ny,s.Nz,na,nb,dt=float)
    #t1=t.time()
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        #ind=s.nonzero().T
        ind=ind.T
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        #ind=s.nonzero().T
        pass
      else:
        ind=s.nonzero().T
    n=len(ind[0])
    for j in range(0,n):
      AngDSM[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]]=np.angle(s[ind[0][j],ind[1][j],ind[2][j],ind[3][j],ind[4][j]])
    #t4=t.time()
    #print('time creating AngDSM',t1-t0)
    #print('time getting indices and finding angles',t3-t1)
    #print('time setting line of sight dummy angles',t4-t3)
    #print('time for all of sparse angles',t4-t0)
    return AngDSM
  ## Multiply every column of the DSM s elementwise with the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=vec[k]*DSM[x,y,z,k,j]
  def dict_scal_mult(s,scal,ind=-1):
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
  def double_dict_vec_multiply(s,DSM2,vec,ind=-1):
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
    outDSM2=DS(s.Nx,s.Ny,s.Nz,na,nb)
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        pass
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        ind=ind.T
      else:
        ind=s.nonzero().T
    Ni=len(np.transpose(ind)[0])
    j=-1
    x,y,z=ind[0][0:3]
    for l in range(0,Ni):
      outDSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=vec[ind[l][3]]*s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]
      outDSM2[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=vec[ind[l][3]]*DSM2[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]
      # if x==ind[l][0] and y==ind[l][1] and z==ind[l][2]:
        # pass
      # else:
        # x,y,z=ind[l][0:3]
        # j=-1
      # else:
        # pass
      # if ind[l][4]==j:
        # pass
      # else:
        # outDSM[ind[l][0],ind[l][1],ind[l][2],:,ind[l][4]]=np.multiply(vec,s[ind[l][0],ind[l][1],ind[l][2],:,ind[l][4]])
        # j=ind[l][4]
    return outDSM,outDSM2
  def dict_vec_multiply(s,vec,ind=-1):
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
    if isinstance(ind, type(-1)):
      ind=s.nonzero().T
    else:
      if len(ind[0])==5 and len(ind.T[0]!=5):
        pass
      elif len(ind[0])!=5 and len(ind.T[0]==5):
        ind=ind.T
      else:
        ind=s.nonzero().T
    Ni=len(np.transpose(ind)[0])
    j=-1
    x,y,z=ind[0][0:3]
    for l in range(0,Ni):
      outDSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=vec[ind[l][3]]*s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]
      # if x==ind[l][0] and y==ind[l][1] and z==ind[l][2]:
        # pass
      # else:
        # x,y,z=ind[l][0:3]
        # j=-1
      # else:
        # pass
      # if ind[l][4]==j:
        # pass
      # else:
        # outDSM[ind[l][0],ind[l][1],ind[l][2],:,ind[l][4]]=np.multiply(vec,s[ind[l][0],ind[l][1],ind[l][2],:,ind[l][4]])
        # j=ind[l][4]
    return outDSM
  ## Divide every column of the DSM s elementwise with the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=DSM[x,y,z,k,j]/vec[k]
  ## Divide every column of the DSM s elementwise with the vector vec.
  # @param vec a row vector with length na.
  # @return a DSM 'out' with the same dimensions as s.
  # out[x,y,z,k,j]=DSM[x,y,z,k,j]/vec[k]
  def gain_phase_ref_mul(s,Com1,Com2,G,ind=-1):
    """
        Multiply all terms of s element wise with Com1 and each row by Gt.
        Multiply all terms of s elementwise with Com2 and each row by Gt.

    :param G: a row vector with length na.
    :param Com1: A DSM with size Nx, Ny,Nz, 1,na
    :param Com2: A DSM with size Nx, Ny,Nz, 1,na

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out1[x,y,z,k,j]=G[j]*s[x,y,z,k,j]*Com1[x,y,z,k,j]
       out2[x,y,z,k,j]=G[j]*s[x,y,z,k,j]*Com2[x,y,z,k,j]

    :rtype: A DSM 'out' with the same dimensions as s.

    :returns: out

     """
    na,nb=s.shape
    out1=DS(s.Nx,s.Ny,s.Nz,na,nb)
    out2=DS(s.Nx,s.Ny,s.Nz,na,nb)
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
    for l in range(0,Ni):
      x,y,z,a,b=ind[l]
      k=G[b]*s[x,y,z,a,b]
      out1[x,y,z,a,b]=k*Com1[x,y,z,a,b]
      out2[x,y,z,a,b]=k*Com2[x,y,z,a,b]
    del x,y,z,a,b,Ni
    return out1,out2
  def gain_phase_rad_ref_mul(s,Com1,Com2,G,khat,L,ind=-1):
    """ Multiply all terms of s element wise with Com1/Rad and each row by Gt.
        Multiply all terms of s elementwise with Com2/Rad and each row by Gt.

    :param G: a row vector with length na.
    :param Rad: A DSM with size Nx, Ny,Nz, 1,na
    :param Com1: A DSM with size Nx, Ny,Nz, 1,na
    :param Com2: A DSM with size Nx, Ny,Nz, 1,na

    For integers :math:`x,y,z,k` and :math:`j` such that,
    :math:`x \in [0,Nx), y \in [0,Ny), z \in [0,Nz), k \in [0,na),j \in [0,nb)`,

    .. code::

       out1[x,y,z,k,j]=G[j]*s[x,y,z,k,j]*Rad[x,y,z,k,j]*Com1[x,y,z,k,j]
       out2[x,y,z,k,j]=G[j]*s[x,y,z,k,j]*Rad[x,y,z,k,j]*Com2[x,y,z,k,j]

    :rtype: A DSM 'out' with the same dimensions as s.

    :returns: out

     """
    na,nb=s.shape
    out1=DS(s.Nx,s.Ny,s.Nz,na,nb)
    out2=DS(s.Nx,s.Ny,s.Nz,na,nb)
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
    for l in range(0,Ni):
      x,y,z,a,b=ind[l]
      m=s[x,y,z,a,b]*khat*(L**2)
      k=G[b]*(np.cos(m)+np.sin(m)*1j)/(s[x,y,z,a,b])
      out1[x,y,z,a,b]=k*Com1[x,y,z,a,b]
      out2[x,y,z,a,b]=k*Com2[x,y,z,a,b]
    del x,y,z,a,b,Ni,m,k
    return out1,out2

  def gain_phase_rad_ref_mul_add(s,Com1,Com2,G,khat,L,lam,ind=-1):
    """ Multiply all terms of s element wise with Com1/Rad and each row by Gt.
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
    out1=np.zeros((s.Nx,s.Ny,s.Nz),dtype=np.complex128)#DS(s.Nx,s.Ny,s.Nz,na,nb)
    out2=np.zeros((s.Nx,s.Ny,s.Nz),dtype=np.complex128)#DS(s.Nx,s.Ny,s.Nz,na,nb)
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
    for l in range(0,Ni):
      x,y,z,a,b=ind[l]
      if s[x,y,z,a,b]==0:
        pass
      else:
        m=s[x,y,z,a,b]*khat*(L**2)
        k=lam*(np.sqrt(G[b])*(np.cos(m)+np.sin(m)*1j)/(s[x,y,z,a,b]*4*np.pi*L))[0]
        #if x==0 and y==5 and z==4:
        #  print('top',s[x,y,z,a,b],k,Com1[x,y,z,a,b],a,b)
        #if x==9 and y==4 and z==4:
        #  print('bottom',s[x,y,z,a,b],k,Com1[x,y,z,a,b],a,b)
        out1[x,y,z]+=k*Com1[x,y,z,a,b]
        out2[x,y,z]+=k*Com2[x,y,z,a,b]
        del x,y,z,a,b,m,k
    del Ni
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
      outDSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]=vec[ind[l][4]]*s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]]
      #out=np.multiply(vec,s[ind[l][3]],s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],ind[l][4]])
      # if ind[l][0]!=x or ind[l][1]!=y or ind[l][2]!=z:
        # x,y,z=ind[l][0:3]
        # j=-1
      # else:
        # pass
      # if ind[l][3]==j:
        # pass
      # else:
        # outDSM[ind[l][0],ind[l][1],ind[l][2],ind[l][3],:]=np.multiply(vec.T, s[ind[l][0],ind[l][1],ind[l][2],ind[l][3],:].todense()) #FIXME don't need out just set outDSM
        # j=ind[l][3]
    # del x,y,z, j#,out
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
      outDSM[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]/vec[ind[3][i]]
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
    #del p,lm,q,x,y,z,bi,n
    return outDSM

  def costhetat(s,refindex,ind=-1):
    ''' Takes in a Mesh of angles with nonzero terms at ind. Computes
    cos of thetat at those angles using the refractive index's.
    :param ind: The indices of the nonzero terms.
    :param refindex: The refractive index's of the obstacles in a vector.

    .. code::

       SIN=sin(s)
       thetat=asin(SIN/refindex)
       ctht=cos(thetat)

    :rtype: DSM
    :returns: ctht'''
    na,nb=s.shape
    ctht=DS(s.Nx,s.Ny,s.Nz,na,nb)
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
        x=ind[0][i]
        y=ind[1][i]
        z=ind[2][i]
        ai=ind[3][i]
        bi=ind[4][i]
        ctht[x,y,z,ai,bi]=np.cos(np.arcsin(np.sin(s[x,y,z,ai,bi])/refindex[ai]))
    del x,y,z,ai,bi,n
    return ctht
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
      if out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]==0:
        out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
      else:
        out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]*=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
    del  n
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
      if out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]==0:
        out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
        out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]=DSM[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
      else:
        out[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]*=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
        out2[ind[0][i],ind[1][i],ind[2][i],0,ind[4][i]]*=DSM[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
    del  n
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
      # if x==indices[l][0] and  y==indices[l][1] and z==indices[l][2]:
        # pass
      # else:
        # x,y,z=indices[l][0:3]
        # j=-1
      # if j==indices[l][3]:
        # pass
      # else:
      outDSM[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]]=np.divide(vec[indices[l][3]],s[indices[l][0],indices[l][1],indices[l][2],indices[l][3],indices[l][4]])
      #outDSM[indices[l][0],indices[l][1],indices[l][2],:,indices[l][4]]=out
      #  j=indices[l][3]
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
          del indicesSec
        else:
          indicesSec=np.c_[np.tile(np.array([x,y,z]),(NI,1)),indicesM[0][0:],indicesM[1][0:]]
          indices=np.vstack((indices,indicesSec))
          del indicesSec
    if check==-1:
      indices=np.array([])
    return indices
  def row_sum(s,ind=-1):
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
      out[ind[0][i],ind[1][i],ind[2][i],ind[3][i],0]+=s[ind[0][i],ind[1][i],ind[2][i],ind[3][i],ind[4][i]]
    return out
  def xyznonzero(s):
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
    check=-1
    for x,y,z in product(range(0,s.Nx),range(0,s.Ny),range(0,s.Nz)):
      indicesM=s.d[x,y,z].nonzero()
      NI=len(indicesM[0])
      if abs(NI)<epsilon:
        pass
      else:
        if check==-1:
          check=0
          indices=np.array([x,y,z])
        else:
          indices=np.vstack((indices,np.array([x,y,z])))
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
  def phase_calc(s,khat,L,ind=-1):
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
    na,nb=s.shape
    out=DS(s.Nx,s.Ny,s.Nz,na,nb)
    for j in range(n):
      x=ind[0][j]
      y=ind[1][j]
      z=ind[2][j]
      a=ind[3][j]
      b=ind[4][j]
      m=s[x,y,z,a,b]*khat*(L**2)
      out[x,y,z,a,b]=np.cos(m)+np.sin(m)*1j
    return out
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
    if i>s.Nx-1 or j>s.Ny-1 or k>s.Nz-1 or i<0 or j<0 or k<0:
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
    if isinstance(ps[0],(float,int,np.int64, np.complex128 )):
      check=s.stopcheck(ps[0],ps[1],ps[2],p1,h)
      if check==1:
        newps=np.array([[ps[0]],[ps[1]],[ps[2]]])
        newp3=np.array([[p3[0]],[p3[1]],[p3[2]]])
        newn =np.array([[n[0]], [n[1]], [n[2]]])
      else:
        pass
    else:
     for k in ps:
      check=s.stopcheck(k[0],k[1],k[2],p1,h)
      # if check==1 and start==1:
        # if k[0] in newps[0]:
          # kindex=np.where(newp3[0]==k[0])
          # if k[1]==newps[1][kindex] and k[2]==newps[2][kindex]:
            # check==0
          # else:
            # pass
        # else:
          # pass
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

def stopcheck(i,j,k,p1,h,Nx,Ny,Nz):
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
def stopchecklist(ps,p1,h,p3,n,Nx,Ny,Nz):
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
      check=stopcheck(k[0],k[1],k[2],p1,h,Nx,Ny,Nz)
      if start==1:
        if k[0] in newps[0]:
          kindex=np.where(newp3[0]==k[0])
          if k[1]==newps[1][kindex] and k[2]==newps[2][kindex]:
            check==0
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

def phase_calc(RadMesh,khat,L,ind=-1):
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
  if isinstance(ind, type(-1)):
    ind=RadMesh.nonzero()
  else:
    pass
  S2=RadMesh.dict_scal_mult(khat*(L**2),ind)
  #S2=S1.dict_scal_mult(L**2,ind)
  out=S2.cos(ind)+S2.sin(ind).dict_scal_mult(1j,ind)
  return out


def power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt, Pol,Nra,Nre,Ns,ind=-1):
  ''' Compute the field from a Mesh of ray information and the physical \
  parameters.

  :param Mesh: The :py:class:`DS` mesh of ray information.
  :param Znobrat: An Nob x Nre+1 array containing tiles of the impedance \
    of obstacles divided by the impedance of air.
  :param refindex: An Nob x Nre+1 array containing tiles of the refractive\
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
  # Compute the reflection coefficients
  #t1=t.time()
  if not os.path.exists('./Mesh'):
    os.makedirs('./Mesh')
  angfile=str('./Mesh/ang'+str(int(Nra))+'Refs'+str(int(Nre))+'Ns'+str(int(Ns))+'.npy')
  afile=Path(angfile)
  if afile.is_file():
    AngDSM=load_dict(angfile)
  else:
    AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
    AngDSM.save_dict(angfile)
  #AngDSM=Mesh.sparse_angles(ind)
  #t2=t.time()
  #(AngDSM[0,0,5])
  Comper,Compar=AngDSM.refcoefbyterm_withmul(Znobrat,refindex,lam,L,ind)
  #print(Comper[0,0,5], Compar[0,0,5])
  #print(Rper,Rpar)
  # Combine the reflection coefficients to get the reflection loss on each ray.
  #t3=t.time()
  #Comper,Compar=Rper.double_dict_col_mult_(Rpar,ind) # with ind
  #t2=t.time()
  # Get the distances for each ray segment from the Mesh
  rfile=str('./Mesh/rad'+str(int(Nra))+'Refs'+str(int(Nre))+'Ns'+str(int(Ns))+'.npy')
  radfile = Path(rfile)
  h=1/Mesh.Nx
  if radfile.is_file():
    RadMesh=load_dict(rfile)
    ind=RadMesh.nonzero()
  else:
    RadMesh,ind=Mesh.__get_rad__(h,ind)
    RadMesh.save_dict(rfile)
  t4=t.time()
  #print(RadMesh[0,0,5])
  # Compute the mesh of phases
   #FIXME try removing this line and using the previous indices
  #pha=RadMesh.phase_calc(khat,L,ind)
 # print(pha)
  #t4=t.time()
  # Divide by the rads
  # pharad=pha.__truediv__(RadMesh,ind)
  # t5=t.time()
  # # Multiply by the gains.
  # #ind=pharad.nonzero()
  # Gtpha=pharad.dict_row_vec_multiply(np.sqrt(Gt),ind)
  # #print(Gtpha.nonzero())
  # # Combine Gains, phase and reflection
  # GtphaRpe=Gtpha.__mul__(Comper,ind)
  # GtphaRpa=Gtpha.__mul__(Compar,ind)
  # t6=t.time()
  # GtphaRpe,GtphaRpa=pharad.gain_phase_ref_mul(Comper,Compar,Gt,ind)
  # t7=t.time()
  Gridpe, Gridpa=RadMesh.gain_phase_rad_ref_mul_add(Comper,Compar,Gt,khat,L,lam,ind)
  #t7=t.time()
  # if t7-t6>t6-t5:
    # print("method 1",t7-t6,t6-t5)
  # else:
    # print("method 2",t7-t6,t6-t5)
  # if t8-t7>t6-t4:
    # print("method A",t8-t7,t6-t4)
  # else:
    # print("method B",t8-t7,t6-t4)
  #FIXME
  #t7=t.time()
  #Gridpe=GtphaRpe.togrid(ind) #Add all terms in each SM to give the term in the np array
  #Gridpa=GtphaRpa.togrid(ind) #Add all terms in each SM to give the term in the np array
  #t7=t.time()
  # Multiply by the lambda\L
  #Gridpe*=(lam/(L*4*math.pi))
  #Gridpa*=(lam/(L*4*math.pi))
  #t8=t.time()
  # Polarisation
  #apar=Pol[0]
  #aper=Pol[1]
  #t9=t.time()
  # Power
  P=np.zeros((Mesh.Nx,Mesh.Ny,Mesh.Nz),dtype=np.longdouble)
  P=np.absolute(Gridpe*Pol[0])**2+np.absolute(Gridpa*Pol[1])**2
  P=10*np.log10(P,where=(P!=0))
  #t10=t.time()
  # print('----------------------------------------------------------')
  # print('Total time to find power', t10-t0)
  # print('----------------------------------------------------------')
  # print('----------------------------------------------------------')
  # print('Time computing square and log ', t10-t9)
  # del t10
  # print('Time assigning polarisation ', t9-t8)
  # del t9
  # print('Time multiplying by wavelength term ', t8-t7)
  # del t8
  # print('Time combining reflection coefficents, gains and phase, rad div add and convert to np array', t7-t4)
  # del t7
  #print('Time multiplying phase terms by gains ', t6-t5)
  #del t6
  #print('Time dividing phase terms by radius ', t5-t4)
  #del t5
  #print('Time computing phase,', t5-t4)
  #del t4
  # print('Time getting distances ', t4-t3)
  # del t4
  # print('Time computing reflection coefficients ', t3-t2)
  # del t3
  # print('Time finding angles', t2-t1)
  # del t2, t1, t0
  # print('----------------------------------------------------------')
  return P,indout


def quality_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt, Pol,Nra,Nre,Ns,ind=-1):
  ''' Compute the field from a Mesh of ray information and the physical \
  parameters.

  :param Mesh: The :py:class:`DS` mesh of ray information.
  :param Znobrat: An Nob x Nre+1 array containing tiles of the impedance \
    of obstacles divided by the impedance of air.
  :param refindex: An Nob x Nre+1 array containing tiles of the refractive\
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
  angfile=str('./Mesh/ang'+str(int(Nra))+'Refs'+str(int(Nre))+'Ns'+str(int(Ns))+'.npy')
  afile=Path(angfile)
  if afile.is_file():
    AngDSM=load_dict(angfile)
  else:
    AngDSM=Mesh.sparse_angles(ind)                       # Get the angles of incidence from the mesh.
    AngDSM.save_dict(angfile)
  Comper,Compar=AngDSM.refcoefbyterm_withmul(Znobrat,refindex,lam,L,ind)
  rfile=str('./Mesh/rad'+str(int(Nra))+'Refs'+str(int(Nre))+'Ns'+str(int(Ns))+'.npy')
  radfile = Path(rfile)
  h=1/Mesh.Nx
  if radfile.is_file():
    RadMesh=load_dict(rfile)
    ind=RadMesh.nonzero()
  else:
    RadMesh,ind=Mesh.__get_rad__(h,ind)
    RadMesh.save_dict(rfile)
  t4=t.time()
  Gridpe, Gridpa=RadMesh.gain_phase_rad_ref_mul_add(Comper,Compar,Gt,khat,L,lam,ind)
  P=np.zeros((Mesh.Nx,Mesh.Ny,Mesh.Nz),dtype=np.longdouble)
  P=np.absolute(Gridpe*Pol[0])**2+np.absolute(Gridpa*Pol[1])**2
  P=10*np.log10(P,where=(P!=0))
  Q=np.sum(P)/(Mesh.Nx*Mesh.Ny*Mesh.Nz)
  return Q,ind


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

#=======================================================================
# FUNCTIONS CONNECTED TO DS BUT AREN'T PART OF THE OBJECT
#=======================================================================

def singletype(x):
  if isinstance(x,(float,int,np.int32,np.int64, np.complex128 )):
    return True
  else: return False

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
  Nx=max(Keys)[0]-min(Keys)[0]+1
  Ny=max(Keys)[1]-min(Keys)[1]+1
  Nz=max(Keys)[2]-min(Keys)[2]+1
  na=ret_di[0,0,0].shape[0]
  nb=ret_di[0,0,0].shape[1]
  ret_ds=DS(Nx,Ny,Nz,na,nb)
  default_value=SM((na,nb),dtype=np.complex128)
  for k in Keys:
    ret_ds.__setitem__(k,ret_di[k])
  return ret_ds

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
  angfile=str('./Mesh/ang'+str(int(Nra))+'Refs'+str(int(Nre))+'Ns'+str(int(Ns))+'.npy')
  afile=Path(angfile)
  if afile.is_file():
    AngDSM=load_dict(angfile)
  else:
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
  # del t5
  # print('Time computing cos thetai', t4-t3)
  # del t4
  # print('Time initialising cthi ctht SIN', t3-t2)
  #del t3
  print('Time getting angles from mesh', t2-t1)
  del t2
  print('Time getting nonzero indices', t1-t0)
  del t1
  print('----------------------------------------------------------')
  print('Total time to compute reflection coefficients from Mesh ', t6-t0)
  del t6, t0
  print('----------------------------------------------------------')
  return Rper, Rpar, ind

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
  # print(ind)
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
    #print(Mesh) #DEBUG
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
  else:
    return 0

def test_23():
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
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
  meshname=str('DSM'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  Mesh= load_dict(meshname)
  ind=Mesh.nonzero()
  AngDSM=Mesh.sparse_angles()

  ctht=DS(Mesh.Nx,Mesh.Ny,Mesh.Nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta_t) #FIXME
  SIN=AngDSM.sin(ind)
  Div=SIN.dict_DSM_divideby_vec_withind(refindex,ind)             # Divide each column in DSM with refindex elementwise. Set any 0 term to 0.
  ctht=Div.cos_asin(ind)
  return ctht

def test_24():
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
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
  meshname=str('DSM'+str(Nra)+'Refs'+str(Nre)+'m.npy')
  Mesh= load_dict(meshname)
  ind=Mesh.nonzero()
  AngDSM=Mesh.sparse_angles()

  ctht=DS(Mesh.Nx,Mesh.Ny,Mesh.Nz,Mesh.shape[0],Mesh.shape[1])  # Initialise a DSM which will be cos(theta_t) #FIXME
  ctht=AngDSM.costhetat(refindex,ind)
  return ctht


if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  #job_server = pp.Server()
  t1=t.time()
  ctht=test_24()
  t2=t.time()
  print(t2-t1)
  ctht=test_23()
  t3=t.time()
  print(t3-t2)
  print(timeit.timeit("test_23()",setup="from __main__ import test_23"))
  print(timeit.timeit("test_24()",setup="from __main__ import test_24"))

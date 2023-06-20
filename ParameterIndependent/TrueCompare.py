#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  TrueCompare.py
#
#  Copyright 2020 Hayley Wragg <hw454@ThinkPad-X1-Carbon-4th>
#
#  This program is free software; you can redistribute it and/or modify
import numpy as np
import ParameterLoad as PI
import sys
import Room as rom
from itertools import product
import math as ma
import intersection as inst
import os
import DictionarySparseMatrix as DSM
import RayTracerMainProgram as RTM
import matplotlib.pyplot as mp

epsilon=sys.float_info.epsilon

def makematrix_perfectreflection(job,Ns,index=3):
  print('-------------------------------')
  print('True values for perfect reflection')
  print('-------------------------------')
  # Run the ParameterInput file
  SN='InputSheet.xlsx'
  out1=PI.DeclareParameters(SN)
  out2=PI.ObstacleCoefficients(SN)
  if not (out1==0 and out2==0): raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  h=1.0/Ns
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  ## The main ray tracing code runs within a unit square then the results are scaled at the power evaluation.
  ## this code finds the true power values for the environment in it's true length scale (not unit).
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index)         # The obstacles are within a uni square and need to be scaled.
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
  InnerOb=np.load('Parameters/InnerOb%d.npy'%index)

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/Tx%03dGains%03d.npy'%(Nra[0],index))       # Antenna Gains
  freq          = np.load('Parameters/frequency%03d.npy'%index)     # Frequency
  Freespace     = np.load('Parameters/Freespace%03d.npy'%index)     #-[mu0 (permeability of air), eps0 (permittivity of air),Z0 (characteristic impedance of air), c (speed of light)]
  Pol           = np.load('Parameters/Pol%03d.npy'%index)           # Polarisation of the antenna
  khat, lam, L  = np.load('Parameters/Antpar%03d.npy'%index)         # non-dimensional wave number, wave length and length scale
  refindex      =np.load('Parameters/refindex%03d.npy'%index)
  c             =Freespace[3]       # Speed of light in Air
  Ntri         =np.load('Parameters/NtriOut.npy')
  NtriOb       =np.load('Parameters/NtriOb.npy')
  if InnerOb:
    Ntri=np.concatenate(Ntri,NtriOb)
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  ##----Find the dimensions in the x y and z axis
  Nx=int(1+(Room.maxxleng()//h))
  Ny=int(1+(Room.maxyleng()//h))
  Nz=int(1+(Room.maxzleng()//h))
  Ns=max(Nx,Ny,Nz)
  h=1.0/Ns
  Tx=RTM.MoveTx(job,Nx,Ny,Nz,h)
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)

  for o,r in enumerate(refindex):
     if abs(r)>epsilon:
       ob=o
       continue
  nob=Room.nob_from_sur(ob)
  Tri=Oblist[nob].astype(float)
  Tri2=Oblist[nob+1].astype(float)
  p0,p1,p2=Tri
  n=np.cross(p0-p1,p2-p1)
  n/=np.sqrt(np.dot(n,n))
  nleng=np.linalg.norm(n)
  Tx=Tx.astype(float)
  y=(Tx-np.dot((Tx-p0),n)*n/np.dot(n,n))
  # Txhat is the image transmitter location
  Txhat=2*y-Tx
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      # Find the co-ordinate from the indexing positions
      x=Room.coordinate(h,i,j,k)[0]
      # Find the length from the transmitter to the point
      Txleng=np.linalg.norm(x-Tx)
      xhatleng=np.linalg.norm(x-Txhat)
      #print(x,Tx,Txhat,xhatleng,Txleng)
      if Txleng!=0:
        field=DSM.FieldEquation(Txleng,khat,L,lam)*Pol
      else:
        field=0
      field+=DSM.FieldEquation(xhatleng,khat,L,lam)*Pol
      P=(np.absolute(field[0])**2+np.absolute(field[1])**2)
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=xhatleng
  return Mesh,RadMeshA,RadMeshB

def makematrix_circle_LOS(Ns,index=4):
  print('-------------------------------')
  print('True values for perfect reflection')
  print('-------------------------------')
  # Run the ParameterInput file
  SN='InputSheet.xlsx'
  out1=PI.DeclareParameters(SN)
  out2=PI.ObstacleCoefficients(SN)
  if not (out1==0 and out2==0): raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)
  h=1.0/Ns

  ##----Retrieve the environment--------------------------------------
  ## The main ray tracing code runs within a unit square then the results are scaled at the power evaluation.
  ## this code finds the true power values for the environment in it's true length scale (not unit).
  Oblist        =np.load('Parameters/Obstacles.npy')         # The obstacles are within a uni square and need to be scaled.
  Tx            =np.load('Parameters/Origin.npy')             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain


  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/Tx%03dGains%03d.npy'%(Nra[0],index))       # Antenna Gains
  freq          = np.load('Parameters/frequency%03d.npy'%index)     # Frequency
  Freespace     = np.load('Parameters/Freespace%03d.npy'%index)     #-[mu0 (permeability of air), eps0 (permittivity of air),Z0 (characteristic impedance of air), c (speed of light)]
  Pol           = np.load('Parameters/Pol%03d.npy'%index)           # Polarisation of the antenna
  khat, lam, L  = np.load('Parameters/Antpar%03d.npy'%index)         # non-dimensional wave number, wave length and length scale
  c             =Freespace[3]       # Speed of light in Air


  ##----Find the dimensions in the x y and z axis
  Ns=np.load('Parameters/Ns.npy')
  Nx=int(1/(h))
  Ny=int(1/(h))
  Nz=int(1/(h))
  h=Room.maxxleng()/Ns
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)
  k=khat*L
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      # Find the co-ordinate from the indexing positions
      x=Room.coordinate(h,i,j,k)[0]
      # Find the length from the transmitter to the point
      Txleng=np.linalg.norm(x-Tx)
      if Txleng>0.5 or abs(Txleng)<epsilon:
        field=0
      else:
        field=DSM.FieldEquation(Txleng,khat,L,lam)
      P=(np.absolute(field*Pol[0])**2+np.absolute(field*Pol[1])**2)
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=1.0-Txleng
  return Mesh,RadMeshA,RadMeshB

def makematrix_circle_perfectreflection(Ns,index=0):
  print('-------------------------------')
  print('True values for perfect reflection')
  print('-------------------------------')
  # Run the ParameterInput file
  SN='InputSheet.xlsx'
  out1=PI.DeclareParameters(SN)
  out2=PI.ObstacleCoefficients(SN)
  if not (out1==0 and out2==0): raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)
  h=1.0/Ns
  ##----Retrieve the environment--------------------------------------
  ## The main ray tracing code runs within a unit square then the results are scaled at the power evaluation.
  ## this code finds the true power values for the environment in it's true length scale (not unit).
  Oblist        =np.load('Parameters/Obstacles.npy')         # The obstacles are within a uni square and need to be scaled.
  Tx            =np.load('Parameters/Origin.npy')             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain


  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/Tx%03dGains%03d.npy'%(Nra[0],index))      # Antenna Gains
  freq          = np.load('Parameters/frequency%03d.npy'%index)     # Frequency
  Freespace     = np.load('Parameters/Freespace%03d.npy'%index)     #-[mu0 (permeability of air), eps0 (permittivity of air),Z0 (characteristic impedance of air), c (speed of light)]
  Pol           = np.load('Parameters/Pol%03d.npy'%index)           # Polarisation of the antenna
  khat, lam, L  = np.load('Parameters/Antpar%03d.npy'%index)         # non-dimensional wave number, wave length and length scale
  c             =Freespace[3]       # Speed of light in Air


  ##----Find the dimensions in the x y and z axis
  Nx=int(1/(h))
  Ny=int(1/(h))
  Nz=int(1/(h))
  h=Room.maxxleng()/Ns
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)
  k=khat*L
  Ref=1.0
  A=1.0/(4*np.pi*(np.absolute(np.exp(k*1.0j*L)+Ref*np.exp(-k*1.0j*L))**2))
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      # Find the co-ordinate from the indexing positions
      x=Room.coordinate(h,i,j,k)[0]
      # Find the length from the transmitter to the point
      Txleng=np.linalg.norm(x-Tx)
      if Txleng>0.5:
        field=DSM.FieldEquation(Txleng,khat,L,lam)*(1-Ref)
      elif abs(Txleng)>epsilon:
        field=DSM.FieldEquation(1-Txleng,khat,L,lam)*Ref
        field+=DSM.FieldEquation(Txleng,khat,L,lam)
      P=(np.absolute(field*Pol[0])**2+np.absolute(field*Pol[1])**2)
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=1.0-Txleng
  return Mesh,RadMeshA,RadMeshB

def makematrix_circle_withloss(Ns,index=4):
  print('-------------------------------')
  print('True values for reflection loss in a sphere')
  print('-------------------------------')
  # Run the ParameterInput file
  SN='InputSheet.xlsx'
  out1=PI.DeclareParameters(SN)
  out2=PI.ObstacleCoefficients(SN)
  if not (out1==0 and out2==0): raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)
  h=1.0/Ns

  ##----Retrieve the environment--------------------------------------
  ## The main ray tracing code runs within a unit square then the results are scaled at the power evaluation.
  ## this code finds the true power values for the environment in it's true length scale (not unit).
  Oblist        =np.load('Parameters/Obstacles.npy')         # The obstacles are within a uni square and need to be scaled.
  Tx            =np.load('Parameters/Origin.npy')             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain


  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/Tx%03dGains%03d.npy'%(Nra[0],index))       # Antenna Gains
  freq          = np.load('Parameters/frequency%03d.npy'%index)     # Frequency
  Freespace     = np.load('Parameters/Freespace%03d.npy'%index)     #-[mu0 (permeability of air), eps0 (permittivity of air),Z0 (characteristic impedance of air), c (speed of light)]
  Pol           = np.load('Parameters/Pol%03d.npy'%index)           # Polarisation of the antenna
  khat, lam, L  = np.load('Parameters/Antpar%03d.npy'%index)         # non-dimensional wave number, wave length and length scale
  c             =Freespace[3]       # Speed of light in Air


  ##----Find the dimensions in the x y and z axis
  Ns=np.load('Parameters/Ns.npy')
  Nx=int(1/(h))
  Ny=int(1/(h))
  Nz=int(1/(h))
  h=Room.maxxleng()/Ns
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  Mesht    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  refindex     =np.load('Parameters/refindex%03d.npy'%index)
  Znobr=Znobrat[3]
  refind=refindex[3]
  theta=0
  cthi=np.cos(theta)
  ctht=np.cos(np.arcsin(np.sin(theta)/refind))
  S1=cthi*Znobr
  S2=ctht*Znobr
  Refcoef=np.array([[(S1-ctht)/(S1+ctht),.0+0.0j],[.0+0.0j,(cthi-S2)/(cthi+S2)]])
  ThetaMesh=np.zeros((Nx,Ny,Nz),dtype=float)
  onevec=np.ones((1,2),dtype=np.complex128)
  field=np.zeros((2,1),dtype=np.complex128)
  TranCoef=onevec-np.matmul(Refcoef,Pol)
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      # Find the co-ordinate from the indexing positions
      x=Room.coordinate(h,i,j,k)[0]
      # Find the length from the transmitter to the point
      Txleng=np.linalg.norm(x-Tx)
      if Txleng>0.5:
        field[0]=DSM.FieldEquation(Txleng,khat,L,lam)*TranCoef[0,0]
        field[1]=DSM.FieldEquation(Txleng,khat,L,lam)*TranCoef[0,1]
        P=(np.absolute(field[0])**2+np.absolute(field[1])**2)
        Mesht[i,j,k]=DSM.Watts_to_db(P)
        Mesh[i,j,k]=DSM.Watts_to_db(0)
      elif abs(Txleng)>epsilon:
        fielda=DSM.FieldEquation(Txleng,khat,L,lam)
        field[0]=fielda+DSM.FieldEquation(1-Txleng,khat,L,lam)*(np.matmul(Refcoef,Pol))[0]
        field[1]=fielda+DSM.FieldEquation(1-Txleng,khat,L,lam)*(np.matmul(Refcoef,Pol))[1]
        RadMeshB[i,j,k]=1.0-Txleng
        P=(np.absolute(field[0])**2+np.absolute(field[1])**2)
        Mesh[i,j,k]=DSM.Watts_to_db(P)
        Mesht[i,j,k]=DSM.Watts_to_db(0)
      #print(P,field,Mesh[i,j,k])
      RadMeshA[i,j,k]=Txleng
  return Mesh,Mesht,RadMeshA,RadMeshB

def makematrix_LOS(job,Ns,index=4):
  print('-------------------------------')
  print('True values for LOS')
  print('-------------------------------')
  # Run the ParameterInput file
  SN='InputSheet.xlsx'
  out1=PI.DeclareParameters(SN)
  out2=PI.ObstacleCoefficients(SN)
  if out1==0 and out2==0: pass
  else:
      raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index)          # The obstacles which are within the outerboundary
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/Tx%03dGains%03d.npy'%(Nra[0],index))
  freq          = np.load('Parameters/frequency%03d.npy'%index)
  Freespace     = np.load('Parameters/Freespace%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  c             =Freespace[3]
  khat,lam,L    =np.load('Parameters/Antpar%03d.npy'%index)

  # Currently Co-ordinates are for a unit length room. Rescale for the length scale.
  h=1.0/Ns
  Nx=int(1+Room.maxxleng()//h)
  Ny=int(1+Room.maxyleng()//h)
  Nz=int(1+Room.maxzleng()//h)
  Tx=RTM.MoveTx(job,Nx,Ny,Nz,h)
  Mesh   =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMesh=np.zeros((Nx,Ny,Nz),dtype=float)
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      x=Room.coordinate(h,i,j,k)[0]
      Txleng=np.linalg.norm(x-Tx)
      if abs(Txleng)>0:
        field=DSM.FieldEquation(Txleng,khat,L,lam)*Pol
      P=(np.absolute(field[0])**2+np.absolute(field[1])**2)
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMesh[i,j,k]=Txleng
  return Mesh,RadMesh

def makematrix_withsingle_reflection(job,Ns,index=3):
  print('-------------------------------')
  print('True values for lossy reflection')
  print('-------------------------------')
  # Run the ParameterInput file
  SN='InputSheet.xlsx'
  out1=PI.DeclareParameters(SN)
  out2=PI.ObstacleCoefficients(SN)
  if out1==0 & out2==0: pass
  else:
      raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,_,L    =np.load('Parameters/Raytracing.npy')[0:3]
  h=1.0/Ns
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index)          # The obstacles which are within the outerboundary
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
  InnerOb       =np.load('Parameters/InnerOb%d.npy'%index)
  # Room contains all the obstacles and walls.

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/Tx%03dGains%03d.npy'%(Nra[0],index))
  freq          = np.load('Parameters/frequency%03d.npy'%index)
  Freespace     = np.load('Parameters/Freespace%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  Antpar        = np.load('Parameters/Antpar%03d.npy'%index)
  khat,lam,L    =Antpar
  c             =Freespace[3]

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  refindex     =np.load('Parameters/refindex%03d.npy'%index)
  Ntri         =np.load('Parameters/NtriOut.npy')
  NtriOb       =np.load('Parameters/NtriOb.npy')
  if InnerOb:
    Ntri=np.concatenate(Ntri,NtriOb)
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Nx=int(1+(Room.maxxleng()//h))
  Ny=int(1+(Room.maxyleng()//h))
  Nz=int(1+(Room.maxzleng()//h))
  Ns=max(Nx,Ny,Nz)
  h=1.0/Ns
  Tx=RTM.MoveTx(job,Nx,Ny,Nz,h)
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)
  ThetaMesh=np.zeros((Nx,Ny,Nz),dtype=float)
  # y is the intersection point on surface 0 which is closest to the transmitter.
  for o,r in enumerate(refindex):
     if abs(r)>epsilon:
       ob=o
       continue
  refind=refindex[ob]
  Znobr=Znobrat[ob]
  nob=Room.nob_from_sur(ob)
  Tri=Oblist[nob].astype(float)
  Tri2=Oblist[nob+1].astype(float)
  p0,p1,p2=Tri
  n=np.cross(p0-p1,p2-p1)
  n/=np.sqrt(n@n)
  nleng=np.linalg.norm(n)
  Tx=Tx.astype(float)
  y=(p0-Tx)@n*n/(n@n)
  # Txhat is the image transmitter location
  Txhat=Tx+2*y
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      x=Room.coordinate(h,i,j,k)[0]
      d=x-Tx
      Txleng=np.linalg.norm(d)
      d2=x-Txhat
      d2/=np.linalg.norm(d2)
      I2=((p0-Txhat)@n/(d2@n))*d2+Txhat
      chck=inst.InsideCheck(I2,Tri)
      chck2=inst.InsideCheck(I2,Tri2)
      xhatleng=np.linalg.norm(Txhat-x)
      alpha=(((Txleng**2/d@n)+d@n)/np.linalg.norm(d/(d@n)+n))
      theta=np.arctan((d@n)/((2*(p0-Tx)@n-alpha)))
      if theta>np.pi/2:
        theta=np.pi-theta
      cthi=np.cos(theta)
      if chck==1:
        ctht=np.cos(np.arcsin(np.sin(theta)/refind))
        S1=cthi*Znobr
        S2=ctht*Znobr
      elif chck2==1:
        ctht=np.cos(np.arcsin(np.sin(theta)/refind))
        S1=cthi*Znobr
        S2=ctht*Znobr
      if Txleng!=0:
        field=DSM.FieldEquation(Txleng,khat,L,lam)*Pol
      else:
        field=np.array([0+0j,0+0j])
      Refcoef=np.array([[(S1-ctht)/(S1+ctht),.0+0.0j],[.0+0.0j,(cthi-S2)/(cthi+S2)]])
      field+=DSM.FieldEquation(xhatleng,khat,L,lam)*np.matmul(Refcoef,Pol)
      P=np.absolute(field[0])**2+np.absolute(field[1])**2
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=xhatleng
      ThetaMesh[i,j,k]=theta
  return Mesh,RadMeshA, RadMeshB,ThetaMesh


def plot_mesh(Mesh,Meshfolder,Meshname,lb,ub):
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  n=Mesh.shape[2]

  for i in range(n):
      mp.figure(n+i)
      #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
      mp.imshow(Mesh[:,:,i], cmap=cmapopt,  vmax=ub,vmin=lb)
      mp.colorbar()
      if not os.path.exists(Meshfolder):
        os.makedirs(Meshfolder)
      filename=Meshname+'slice%dof%d.jpg'%(i+1,n)
      mp.savefig(filename)
      mp.clf()
  return

def jobfromTx(Tx,h):
  H=(Tx[2]-0.5*h)//h
  t=(Tx[0]-0.5*h)//h
  u=(Tx[1]-0.5*h)//h
  return H*100+t*10+u

def main():
  joblist=np.array([652,665])
  index=3
  InnerOb    =np.load('Parameters/InnerOb%d.npy'%index)
  if InnerOb:
    box='Box'
  else:
    box='NoBox'
  Nre,_,_,_    =np.load('Parameters/Raytracing.npy')
  Nrs =np.load('Parameters/Nrs%d.npy'%index)
  Nsur=np.load('Parameters/Nsur%d.npy'%index)
  refindex=np.load('Parameters/refindex%03d.npy'%index)
  obnumbers=np.zeros((Nrs,1))
  k=0
  Obstr=''
  if 0<Nrs<Nsur:
    for ob, refind in enumerate(refindex):
      if abs(refind)>epsilon:
        obnumbers[k]=ob
        k+=1
        Obstr=Obstr+'Ob%02d'%ob
  Ns=11
  MeshLOS,RadMeshLOSa,RadMeshLOSb=makematrix_circle_LOS(Ns)
  MeshPerfRef,RadMeshPerfRefa,RadMeshPerfRefb=makematrix_circle_perfectreflection(Ns)
  MeshRef,MeshTRef,RadMeshRefa,RadMeshRefb=makematrix_circle_withloss(Ns)
  for job in joblist:
    Mesh1,RadMesh1=makematrix_LOS(job,Ns)
    Q1=DSM.QualityFromPower(Mesh1)
    Mesh2,RadMesh2a,RadMesh2b=makematrix_perfectreflection(job,Ns,index)
    Q2=DSM.QualityFromPower(Mesh2)
    Mesh4,RadMesh4a,RadMesh4b,ThetaMesh4=makematrix_withsingle_reflection(job,Ns,index)
    Q4=DSM.QualityFromPower(Mesh4)
    if job==665:
      loca='Centre'
    else:
      loca='OffCentre'
    if not os.path.exists('./Mesh'):
        os.makedirs('./Mesh/')
        os.makedirs('./Mesh/True')
        os.makedirs('./Mesh/True/LOS'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca+'/MultiRef'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca)
        os.makedirs('./Mesh/True/SingleRef'+box+loca)
        os.makedirs('./Mesh/True/SinglePerfRef'+box+loca)
        os.makedirs('./Mesh/True/MultiRef'+box+loca)
        os.makedirs('./Mesh/True/MultiPerfRef'+box+loca)
        os.makedirs('./Mesh/True/twoRef'+box+loca)
        os.makedirs('./Mesh/True/twoPerfRef'+box+loca)
    if not os.path.exists('./Mesh/True'):
        os.makedirs('./Mesh/True')
        os.makedirs('./Mesh/True/LOS'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca+'/MultiRef'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca)
        os.makedirs('./Mesh/True/SingleRef'+box+loca)
        os.makedirs('./Mesh/True/SinglePerfRef'+box+loca)
        os.makedirs('./Mesh/True/MultiRef'+box+loca)
        os.makedirs('./Mesh/True/MultiPerfRef'+box+loca)
        os.makedirs('./Mesh/True/twoRef'+box+loca)
        os.makedirs('./Mesh/True/twoPerfRef'+box+loca)
    if not os.path.exists('./Mesh/True/LOS'+box+loca):
        os.makedirs('./Mesh/True/LOS'+box+loca)
    if not    os.path.exists('./Mesh/True/SingleRef'+box+loca):
        os.makedirs('./Mesh/True/SingleRef'+box+loca)
    if not os.path.exists('./Mesh/True/SinglePerfRef'+box+loca):
        os.makedirs('./Mesh/True/SinglePerfRef'+box+loca)
    if not os.path.exists('./Mesh/True/twoRef'+box+loca):
        os.makedirs('./Mesh/True/twoRef'+box+loca)
    if not os.path.exists('./Mesh/True/twoPerfRef'+box+loca):
        os.makedirs('./Mesh/True/twoPerfRef'+box+loca)
    if not os.path.exists('./Mesh/True/MultiRef'+box+loca):
        os.makedirs('./Mesh/True/MultiRef'+box+loca)
    if not os.path.exists('./Mesh/True/MultiPerfRef'+box+loca):
        os.makedirs('./Mesh/True/MultiPerfRef'+box+loca)
    if not os.path.exists('./Mesh/True/Circle'+box+loca):
        os.makedirs('./Mesh/True/Circle'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca)
        os.makedirs('./Mesh/True/Circle'+box+loca+'/MultiRef'+box+loca)
    if not os.path.exists('./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca):
        os.makedirs('./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca)
    if not os.path.exists('./Mesh/True/Circle'+box+loca+'/MultiRef'+box+loca):
        os.makedirs('./Mesh/True/Circle'+box+loca+'/MultiRef'+box+loca)
    Truename='./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca+'/'+box+'True_tx%03d.npy'%job
    TrueRadname='./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca+'/'+box+'TrueRadA_tx%03d.npy'%job
    TrueRadBname='./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca+'/'+box+'TrueRadA_tx%03d.npy'%job
    TrueQname='./Mesh/True/Circle'+box+loca+'/twoRef'+box+loca+'/'+box+'TrueQ_tx%03d.npy'%job
    # True results for sphere
    TrueFolder='./GeneralMethodPowerFigures/Circle'+box+loca+'/LOS'+box+loca+'/Tx%03d/TrueSlice'%job
    TruePlotName=TrueFolder+'/NoBoxTrueSliceNref%d'%Nre
    ub=np.amax(MeshLOS)
    lb=np.amin(MeshLOS)
    plot_mesh(MeshLOS,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/Circle'+box+loca+'/PerfRef'+box+loca+'/Tx%03d/TrueSlice'%job
    TruePlotName=TrueFolder+'/NoBoxTrueSliceNref%d'%Nre
    ub=np.amax(MeshPerfRef)
    lb=np.amin(MeshPerfRef)
    plot_mesh(MeshPerfRef,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/Circle'+box+loca+'/twoRef'+box+loca+'/Tx%03d/TrueSlice'%job
    TruePlotName=TrueFolder+'/NoBoxTrueSliceNref%d'%Nre
    ub=np.amax(MeshRef)
    lb=np.amin(MeshRef)
    plot_mesh(MeshRef,TrueFolder,TruePlotName,lb,ub)
    TruePlotName=TrueFolder+'/NoBoxTransmissionSliceNref%d'%Nre
    ub=np.amax(MeshTRef)
    lb=np.amin(MeshTRef)
    plot_mesh(MeshTRef,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/Circle'+box+loca+'/twoRef'+box+loca+'/Tx%03d/TrueSlice/Rad'%job
    TruePlotName=TrueFolder+'/NoBoxRadASliceNref%d'%Nre
    ub=max(np.amax(RadMeshRefa),np.amax(RadMeshRefb))
    lb=min(np.amin(RadMeshRefa),np.amin(RadMeshRefb))
    plot_mesh(RadMeshRefa,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/Circle'+box+loca+'/twoRef'+box+loca+'/Tx%03d/TrueSlice/Rad'%job
    TruePlotName=TrueFolder+'/'+box+'RadBSliceNref%d'%Nre
    plot_mesh(RadMeshRefb,TrueFolder,TruePlotName,lb,ub)
    print('True mesh saved at', Truename)
    Truename='./Mesh/True/LOS'+box+loca+'/'+box+'True_tx%03d.npy'%job
    TrueRadname='./Mesh/True/LOS'+box+loca+'/'+box+'TrueRadA_tx%03d.npy'%job
    TrueQname='./Mesh/True/LOS'+box+loca+'/'+box+'TrueQ_tx%03d.npy'%job
    np.save(Truename,Mesh1)
    np.save(TrueRadname,RadMesh1)
    np.save(TrueQname,Q1)
    print('True mesh saved at', Truename)
    Truename='./Mesh/True/SinglePerfRef'+box+loca+'/'+box+Obstr+'True_tx%03d.npy'%job
    TrueRadname='./Mesh/True/SinglePerfRef'+box+loca+'/'+box+Obstr+'TrueRadA_tx%03d.npy'%job
    TrueRadBname='./Mesh/True/SinglePerfRef'+box+loca+'/'+box+Obstr+'TrueRadB_tx%03d.npy'%job
    TrueQname='./Mesh/True/SinglePerfRef'+box+loca+'/'+box+Obstr+'TrueQ_tx%03d.npy'%job
    np.save(Truename,Mesh2)
    np.save(TrueRadname,RadMesh2a)
    np.save(TrueRadBname,RadMesh2b)
    np.save(TrueQname,Q2)
    print('True mesh saved at', Truename)
    Truename='./Mesh/True/SingleRef'+box+loca+'/'+box+Obstr+'True_tx%03d.npy'%job
    TrueRadname='./Mesh/True/SingleRef'+box+loca+'/'+box+Obstr+'TrueRadA_tx%03d.npy'%job
    TrueRadBname='./Mesh/True/SingleRef'+box+loca+'/'+box+Obstr+'TrueRadB_tx%03d.npy'%job
    TrueQname='./Mesh/True/SingleRef'+box+loca+'/'+box+Obstr+'TrueQ_tx%03d.npy'%job
    np.save(Truename,Mesh4)
    np.save(TrueRadname,RadMesh4a)
    np.save(TrueRadBname,RadMesh4b)
    np.save(TrueQname,Q4)
    print('True mesh saved at', Truename)
    TrueFolder='./GeneralMethodPowerFigures/LOS'+box+loca+'/Tx%03d/TrueSlice'%job
    TruePlotName=TrueFolder+'/'+box+'TrueSliceNref%d'%Nre
    ub=np.amax(Mesh1)
    lb=np.amin(Mesh1)
    plot_mesh(Mesh1,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/LOS'+box+loca+'/Tx%03d/TrueSlice/Rad'%job
    TruePlotName=TrueFolder+''+box+'RadSliceNref%d'%Nre
    ub=max(np.amax(RadMesh1),np.amax(RadMesh1))
    lb=min(np.amin(RadMesh1),np.amin(RadMesh1))
    plot_mesh(RadMesh1,TrueFolder,TruePlotName,lb,ub)
    plot_mesh(Mesh2,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/SinglePerfRef'+box+loca+'/Tx%03d/TrueSlice'%job
    TruePlotName=TrueFolder+'/'+box+Obstr+'TrueSliceNref%d'%Nre
    ub=np.amax(Mesh2)
    lb=np.amin(Mesh2)
    plot_mesh(Mesh2,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/SinglePerfRef'+box+loca+'/Tx%03d/TrueSlice/Rad'%job
    TruePlotName=TrueFolder+''+box+Obstr+'RadASliceNref%d'%Nre
    ub=max(np.amax(RadMesh2a),np.amax(RadMesh2a))
    lb=min(np.amin(RadMesh2a),np.amin(RadMesh2a))
    plot_mesh(RadMesh2a,TrueFolder,TruePlotName,lb,ub)
    TruePlotName=TrueFolder+''+box+'RadBSliceNref%d'%Nre
    ub=max(np.amax(RadMesh2b),np.amax(RadMesh2b))
    lb=min(np.amin(RadMesh2b),np.amin(RadMesh2b))
    plot_mesh(RadMesh2b,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/SingleRef'+box+loca+'/Tx%03d/TrueSlice'%job
    TruePlotName=TrueFolder+'/'+box+Obstr+'TrueSliceNref%d'%Nre
    ub=np.amax(Mesh4)
    lb=np.amin(Mesh4)
    plot_mesh(Mesh4,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/SingleRef'+box+loca+'/Tx%03d/TrueSlice/Rad'%job
    TruePlotName=TrueFolder+''+box+Obstr+'RadASliceNref%d'%Nre
    ub=max(np.amax(RadMesh4a),np.amax(RadMesh4a))
    lb=min(np.amin(RadMesh4a),np.amin(RadMesh4a))
    plot_mesh(RadMesh4a,TrueFolder,TruePlotName,lb,ub)
    TruePlotName=TrueFolder+''+box+Obstr+'RadBSliceNref%d'%Nre
    ub=max(np.amax(RadMesh4b),np.amax(RadMesh4b))
    lb=min(np.amin(RadMesh4b),np.amin(RadMesh4b))
    plot_mesh(RadMesh4b,TrueFolder,TruePlotName,lb,ub)
  return 0


if __name__ == '__main__':
   main()
   #sys.exit(main(sys.argv))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  TrueCompare.py
#
#  Copyright 2020 Hayley Wragg <hw454@ThinkPad-X1-Carbon-4th>
#
#  This program is free software; you can redistribute it and/or modify
import numpy as np
import ParameterInput as PI
import sys
import Room as rom
from itertools import product
import math as ma
import intersection as inst
import os
import DictionarySparseMatrix as DSM
import matplotlib.pyplot as mp

epsilon=sys.float_info.epsilon

def makematrix_perfectreflection(index=0):
  print('-------------------------------')
  print('True values for perfect reflection')
  print('-------------------------------')
  # Run the ParameterInput file
  out1=PI.DeclareParameters()
  out2=PI.ObstacleCoefficients()
  if out1==0 & out2==0: pass
  else:
      raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L    =np.load('Parameters/Raytracing.npy')
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
  Oblist        =np.load('Parameters/Obstacles.npy')        # The obstacles are within a uni square and need to be scaled.
  Tx            =np.load('Parameters/Origin.npy')/L             # The location of the source antenna (origin of every ray)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Direc         =np.load('Parameters/Directions.npy')         # Matrix of ray directions
  deltheta      =np.load('Parameters/delangle.npy')
  Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain


  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  np.save('Parameters/Nob.npy',Nob)

  ##----Retrieve the antenna parameters--------------------------------------
  Gt            = np.load('Parameters/TxGains'+str(index)+'.npy')       # Antenna Gains
  freq          = np.load('Parameters/frequency'+str(index)+'.npy')     # Frequency
  Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')     #-[mu0 (permeability of air), eps0 (permittivity of air),Z0 (characteristic impedance of air), c (speed of light)]
  Pol           = np.load('Parameters/Pol'+str(index)+'.npy')           # Polarisation of the antenna
  khat, lam, L  = np.load('Parameters/Antpar'+str(index)+'.npy')         # non-dimensional wave number, wave length and length scale
  c             =Freespace[3]       # Speed of light in Air


  ##----Find the dimensions in the x y and z axis
  Ns=np.load('Parameters/Ns.npy')
  Nx=int(1/(h))
  Ny=int(1/(h))
  Nz=int(1/(h))
  h=Room.maxleng()/Ns
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)

  ##---Find the image transmitter if only the first obstacle reflects
  Tri=Oblist[0]
  p0,p1,p2=Tri
  n=np.cross(p0-p1,p0-p2)
  # y is the intersection point on surface 0 which is closest to the transmitter.
  y=(Tx-np.dot((Tx-p0),n)*n/np.dot(n,n))
  # Txhat is the image transmitter location
  Txhat=2*y-Tx
  print(Tx,Txhat)
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      # Find the co-ordinate from the indexing positions
      x=Room.coordinate(h,i,j,k)
      # Find the length from the transmitter to the point
      Txleng=np.linalg.norm(x-Tx)
      xhatleng=np.linalg.norm(x-Txhat)
      if i==5 and j==0 and k==5:
          print(xhatleng, Txleng,x,Txhat)
      field=(lam/(4*ma.pi*Txleng))*np.exp(1j*khat*Txleng*(L**2))*Pol
      field+=(lam/(4*ma.pi*xhatleng))*np.exp(1j*khat*(xhatleng)*(L**2))*Pol
      Mesh[i,j,k]=10*np.log10((np.absolute(field[0])**2+np.absolute(field[1])**2))
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=xhatleng
  return Mesh,RadMeshA,RadMeshB

def makematrix_LOS(index=0):
  print('-------------------------------')
  print('True values for LOS')
  print('-------------------------------')
  # Run the ParameterInput file
  out1=PI.DeclareParameters()
  out2=PI.ObstacleCoefficients()
  if out1==0 & out2==0: pass
  else:
      raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
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
  Gt            = np.load('Parameters/TxGains'+str(index)+'.npy')
  freq          = np.load('Parameters/frequency'+str(index)+'.npy')
  Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')
  Pol           = np.load('Parameters/Pol'+str(index)+'.npy')
  c             =Freespace[3]
  khat,lam,L    =np.load('Parameters/Antpar'+str(index)+'.npy')

  # Currently Co-ordinates are for a unit length room. Rescale for the length scale.
  Nx=int(1/(h))
  Ny=int(1/(h))
  Nz=int(1/(h))
  Mesh   =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMesh=np.zeros((Nx,Ny,Nz),dtype=float)

  h*=L
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      x=Room.coordinate(h,i,j,k)
      Txleng=np.linalg.norm(x-Tx)
      field=(lam*L/(4*ma.pi*Txleng))*np.exp(1j*khat*Txleng*L)*Pol
      Mesh[i,j,k]=10*np.log10((np.absolute(field[0])**2+np.absolute(field[1])**2))
      RadMesh[i,j,k]=Txleng
  return Mesh,RadMesh

def makematrix_withreflection(index=0):
  print('-------------------------------')
  print('True values for lossy reflection')
  print('-------------------------------')
  # Run the ParameterInput file
  out1=PI.DeclareParameters()
  out2=PI.ObstacleCoefficients()
  if out1==0 & out2==0: pass
  else:
      raise('Error occured in parameter declaration')

  ##---- Define the room co-ordinates----------------------------------
  # Obstacles are triangles stored as three 3D co-ordinates

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
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
  Gt            = np.load('Parameters/TxGains'+str(index)+'.npy')
  freq          = np.load('Parameters/frequency'+str(index)+'.npy')
  Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')
  Pol           = np.load('Parameters/Pol'+str(index)+'.npy')
  Antpar        = np.load('Parameters/Antpar'+str(index)+'.npy')
  khat,lam,L    =Antpar
  c             =Freespace[3]

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat'+str(index)+'.npy')
  refindex     =np.load('Parameters/refindex'+str(index)+'.npy')

  Nx=int(1/(h))
  Ny=int(1/(h))
  Nz=int(1/(h))
  h*=L
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)
  Tri=Oblist[0]
  Tri2=Oblist[1]
  p0,p1,p2=Tri
  n=np.cross(p0-p1,p0-p2)
  n/=np.sqrt(np.dot(n,n))
  # y is the intersection point on surface 0 which is closest to the transmitter.
  y=(Tx-np.dot((Tx-p0),n)*n)
  # Txhat is the image transmitter location
  Txhat=2*y-Tx
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      x=Room.coordinate(h,i,j,k)
      d=x-Tx
      Txleng=np.linalg.norm(d)
      d2=x-Txhat
      d2/=np.linalg.norm(d2)
      I2=(np.dot(p0-Txhat,n)/(np.dot(d2,n)))*d2+Txhat
      chck=inst.InsideCheck(I2,Tri)
      chck2=inst.InsideCheck(I2,Tri2)
      #    d/=Txleng
      #   I=(np.dot(p0-Tx,n)/(np.dot(d,n)))*d+Tx
      #    chck=max(inst.InsideCheck(I,Tri),inst.InsideCheck(I,Tri2))
      blah=(np.dot(n,Tx-x)/(Txleng))
      theta=np.arcsin(blah)
      cthi=np.cos(theta)
      if chck==1:
        ctht=np.cos(np.arcsin(np.sin(theta)/refindex[0]))
        S1=cthi*Znobrat[0]
        S2=ctht*Znobrat[0]
      elif chck2==1:
        ctht=np.cos(np.arcsin(np.sin(theta)/refindex[1]))
        S1=cthi*Znobrat[1]
        S2=ctht*Znobrat[1]
      field=(lam*L/(4*ma.pi*Txleng))*np.exp(1j*khat*Txleng*L)*Pol
      xhatleng=np.linalg.norm(Txhat-x)
      Refcoef=np.array([[(S1-ctht)/(S1+ctht),.0+0.0j],[.0+0.0j,(cthi-S2)/(cthi+S2)]])
      field+=(L*lam/(4*ma.pi*(xhatleng)))*np.exp(1j*khat*(xhatleng)*L)*np.matmul(Refcoef,Pol)
      Mesh[i,j,k]=10*np.log10((np.absolute(field[0])**2+np.absolute(field[1])**2))
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=xhatleng
  return Mesh,RadMeshA, RadMeshB

def plot_mesh(Mesh,Meshfolder,Meshname):
  n=Mesh.shape[2]
  lb=np.amin(Mesh)
  ub=np.amax(Mesh)
  for i in range(n):
      mp.figure(n+i)
      #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
      mp.imshow(Mesh[:,:,i], cmap='viridis',  vmax=ub,vmin=lb)
      mp.colorbar()
      if not os.path.exists(Meshfolder):
        os.makedirs(Meshfolder)
      filename=str(Meshname+'slice'+str(int(i+1))+'of'+str(int(n))+'.jpg')#.eps')
      mp.savefig(filename)
      mp.clf()
  return


if __name__ == '__main__':
    Mesh1,RadMesh1=makematrix_LOS()
    Q1=DSM.QualityFromPower(Mesh1)
    Mesh2,RadMesh2a,RadMesh2b=makematrix_perfectreflection()
    Q2=DSM.QualityFromPower(Mesh2)
    Mesh3,RadMesh3a,RadMesh3b=makematrix_withreflection()
    Q3=DSM.QualityFromPower(Mesh3)
    loca=str('Centre')
    RTPar         =np.load('Parameters/Raytracing.npy')
    Nre,h,L       =RTPar
    if not os.path.exists('./Parameters'):
        os.makedirs('./Parameters')
        os.makedirs('./Parameters/LOS'+loca)
        os.makedirs('./Parameters/PerfectRef'+loca)
        os.makedirs('./Parameters/SingleRef'+loca)
    if not os.path.exists('./Parameters/LOS'+loca):
        os.makedirs('./Parameters/LOS'+loca)
    if not os.path.exists('./Parameters/PerfectRef'+loca):
        os.makedirs('./Parameters/PerfectRef'+loca)
    if not os.path.exists('./Parameters/SingleRef'+loca):
        os.makedirs('./Parameters/SingleRef'+loca)
    Truename=str('Parameters/LOS'+loca+'/True.npy')
    TrueRadname=str('Parameters/LOS'+loca+'/TrueRad.npy')
    TrueQname=str('Parameters/LOS'+loca+'/TrueQ.npy')
    np.save(Truename,Mesh1)
    np.save(TrueRadname,RadMesh1)
    np.save(TrueQname,Q1)
    TrueFolder=str('./GeneralMethodPowerFigures/LOS'+loca+'/TrueSlice')
    TruePlotName=str(TrueFolder+'/NoBoxTrueSliceNref'+str(int(Nre)))
    plot_mesh(Mesh1,TrueFolder,TruePlotName)
    TrueFolder=str('./GeneralMethodPowerFigures/LOS'+loca+'/TrueSlice/Rad')
    TruePlotName=str(TrueFolder+'/NoBoxTrueSliceNref'+str(int(Nre)))
    plot_mesh(RadMesh1,TrueFolder,TruePlotName)
    print('True mesh saved at', Truename)
    print('Quality',Q1)
    Truename=str('Parameters/PerfectRef'+loca+'/True.npy')
    TrueRadnameA=str('Parameters/PerfectRef'+loca+'/TrueRadA.npy')
    TrueRadnameB=str('Parameters/PerfectRef'+loca+'/TrueRadB.npy')
    TrueQname=str('Parameters/PerfectRef'+loca+'/TrueQ.npy')
    np.save(Truename,Mesh2)
    np.save(TrueRadnameA,RadMesh2a)
    np.save(TrueRadnameB,RadMesh2b)
    np.save(TrueQname,Q2)
    TrueFolder=str('./GeneralMethodPowerFigures/PerfectRef'+loca+'/TrueSlice')
    TruePlotName=str(TrueFolder+'/NoBoxTrueSliceNref'+str(int(Nre)))
    plot_mesh(Mesh2,TrueFolder,TruePlotName)
    TrueFolder=str('./GeneralMethodPowerFigures/PerfectRef'+loca+'/TrueSlice/RadA')
    TruePlotName=str(TrueFolder+'/NoBoxTrueRadSliceNref'+str(int(Nre)))
    plot_mesh(RadMesh2a,TrueFolder,TruePlotName)
    TrueFolder=str('./GeneralMethodPowerFigures/PerfectRef'+loca+'/TrueSlice/RadB')
    TruePlotName=str(TrueFolder+'/NoBoxTrueRadSliceNref'+str(int(Nre)))
    plot_mesh(RadMesh2b,TrueFolder,TruePlotName)
    print('True mesh saved at', Truename)
    print('Quality',Q2)
    Truename=str('Parameters/SingleRef'+loca+'/True.npy')
    TrueRadnameA=str('Parameters/SingleRef'+loca+'/TrueRadA.npy')
    TrueRadnameB=str('Parameters/SingleRef'+loca+'/TrueRadB.npy')
    TrueQname=str('Parameters/SingleRef'+loca+'/TrueQ.npy')
    np.save(Truename,Mesh3)
    np.save(TrueRadnameA,RadMesh3a)
    np.save(TrueRadnameB,RadMesh3b)
    np.save(TrueQname,Q3)
    TrueFolder=str('./GeneralMethodPowerFigures/SingleRef'+loca+'/TrueSlice')
    TruePlotName=str(TrueFolder+'/NoBoxTrueSliceNref'+str(int(Nre)))
    plot_mesh(Mesh3,TrueFolder,TruePlotName)
    TrueFolder=str('./GeneralMethodPowerFigures/SingleRef'+loca+'/TrueSlice/RadA')
    TruePlotName=str(TrueFolder+'/NoBoxTrueRadSliceNref'+str(int(Nre)))
    plot_mesh(RadMesh3a,TrueFolder,TruePlotName)
    TrueFolder=str('./GeneralMethodPowerFigures/SingleRef'+loca+'/TrueSlice/RadB')
    TruePlotName=str(TrueFolder+'/NoBoxTrueRadSliceNref'+str(int(Nre)))
    plot_mesh(RadMesh3b,TrueFolder,TruePlotName)
    print('True mesh saved at', Truename)
    print('Quality',Q3)
    #sys.exit(main(sys.argv))

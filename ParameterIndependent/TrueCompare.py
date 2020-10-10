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
import matplotlib.pyplot as mp

epsilon=sys.float_info.epsilon

def makematrix_perfectreflection(index=0):
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
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
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
  h=Room.maxxleng()/Ns
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
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      # Find the co-ordinate from the indexing positions
      x=Room.coordinate(h,i,j,k)
      # Find the length from the transmitter to the point
      Txleng=np.linalg.norm(x-Tx)
      xhatleng=np.linalg.norm(x-Txhat)
      #print(x,Tx,Txhat,xhatleng,Txleng)
      if Txleng!=0:
        field=DSM.FieldEquation(Txleng,khat,L,lam)*Pol
      else:
        field=0
      field+=DSM.FieldEquation(xhatleng,khat,L,lam)*Pol
      if i==1 and j==0 and k==0:
        print(Txleng,xhatleng)
        print(DSM.FieldEquation(Txleng,khat,L,lam)*Pol,DSM.FieldEquation(xhatleng,khat,L,lam)*Pol)
      P=(np.absolute(field[0])**2+np.absolute(field[1])**2)
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=xhatleng
  print(Mesh[1,0,0])
  return Mesh,RadMeshA,RadMeshB

def makematrix_circle_perfectreflection(index=0):
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
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
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
  h=Room.maxxleng()/Ns
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)
  k=khat*L
  Ref=1.0
  A=1.0/(4*np.pi*(np.absolute(np.exp(k*1.0j*L)+Ref*np.exp(-k*1.0j*L))**2))
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      # Find the co-ordinate from the indexing positions
      x=Room.coordinate(h,i,j,k)
      # Find the length from the transmitter to the point
      Txleng=np.linalg.norm(x-Tx)
      if Txleng>0.5 or abs(Txleng)<epsilon:
        continue
      #print(x,Tx,Txhat,xhatleng,Txleng)
      field=np.exp(-k*1.0j*(Txleng-1)*L)+Ref*np.exp(k*1.0j*(Txleng-1)*L)/Txleng
      P=(A*np.absolute(field*Pol[0])**2+A*np.absolute(field*Pol[1])**2)
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=1.0-Txleng
  return Mesh,RadMeshA,RadMeshB

def makematrix_LOS(index=0):
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
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
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

  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  Mesh   =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMesh=np.zeros((Nx,Ny,Nz),dtype=float)

  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      x=Room.coordinate(h,i,j,k)
      Txleng=np.linalg.norm(x-Tx)
      if Txleng!=0:
        field=DSM.FieldEquation(Txleng,khat,L,lam)*Pol
      P=(np.absolute(field[0])**2+np.absolute(field[1])**2)
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMesh[i,j,k]=Txleng
  return Mesh,RadMesh

def makematrix_withreflection(index=0):
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
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
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
  print('Transmitter',Tx)
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
  Mesh    =np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshA=np.zeros((Nx,Ny,Nz),dtype=float)
  RadMeshB=np.zeros((Nx,Ny,Nz),dtype=float)
  ThetaMesh=np.zeros((Nx,Ny,Nz),dtype=float)
  Tri=Oblist[0].astype(float)
  Tri2=Oblist[1].astype(float)
  p0,p1,p2=Tri
  n=np.cross(p0-p1,p2-p1)
  n/=np.sqrt(np.dot(n,n))
  nleng=np.linalg.norm(n)
  # y is the intersection point on surface 0 which is closest to the transmitter.
  Tx=Tx.astype(float)
  y=(Tx-np.dot((Tx-p0),n)*n/np.dot(n,n))
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
      xhatleng=np.linalg.norm(Txhat-x)
      d2leng=np.linalg.norm(d2)
      frac=np.dot(d2,n)/(nleng*d2leng)
      theta=np.arccos(frac)
      if theta>np.pi/2:
        theta=np.pi-theta
      cthi=np.cos(theta)
      if chck==1:
        ctht=np.cos(np.arcsin(np.sin(theta)/refindex[0]))
        S1=cthi*Znobrat[0]
        S2=ctht*Znobrat[0]
      elif chck2==1:
        ctht=np.cos(np.arcsin(np.sin(theta)/refindex[1]))
        S1=cthi*Znobrat[1]
        S2=ctht*Znobrat[1]
      if Txleng!=0:
        field=DSM.FieldEquation(Txleng,khat,L,lam)*Pol
      else:
        field=np.array([0+0j,0+0j])
      if i==0 and j==0 and k==2:
        print(theta,d,n,np.dot(d,n)/(nleng*Txleng),np.arccos(np.dot(d,n)/(nleng*Txleng)))
      Refcoef=np.array([[(S1-ctht)/(S1+ctht),.0+0.0j],[.0+0.0j,(cthi-S2)/(cthi+S2)]])
      field+=DSM.FieldEquation(xhatleng,khat,L,lam)*np.matmul(Refcoef,Pol)
      P=np.absolute(field[0])**2+np.absolute(field[1])**2
      Mesh[i,j,k]=DSM.Watts_to_db(P)
      RadMeshA[i,j,k]=Txleng
      RadMeshB[i,j,k]=xhatleng
      ThetaMesh[i,j,k]=theta
  return Mesh,RadMeshA, RadMeshB,ThetaMesh

def plot_mesh(Mesh,Meshfolder,Meshname,lb,ub):
  cmapopt=str('plasma')
  n=Mesh.shape[2]

  for i in range(n):
      mp.figure(n+i)
      #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
      mp.imshow(Mesh[:,:,i], cmap=cmapopt,  vmax=ub,vmin=lb)
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
    Mesh,RadMesha,RadMeshb=makematrix_circle_perfectreflection()
    Q=DSM.QualityFromPower(Mesh)
    # Mesh2,RadMesh2a,RadMesh2b=makematrix_perfectreflection()
    # Q2=DSM.QualityFromPower(Mesh2)
    # Mesh3,RadMesh3a,RadMesh3b,ThetaMesh=makematrix_withreflection()
    # Q3=DSM.QualityFromPower(Mesh3)
    loca=str('OffCentre')
    RTPar         =np.load('Parameters/Raytracing.npy')
    Nre,h,L       =RTPar[0:3]
    if not os.path.exists('./Mesh'):
        os.makedirs('./Mesh/')
        os.makedirs('./Mesh/True')
        os.makedirs('./Mesh/True/LOS'+loca)
        os.makedirs('./Mesh/True/Circle')
        os.makedirs('./Mesh/True/Circle/MultiRef'+loca)
        #os.makedirs('./Mesh/True/SingleRef'+loca)
    if not os.path.exists('./Mesh/True'):
        os.makedirs('./Mesh/True')
        os.makedirs('./Mesh/True/LOS'+loca)
        os.makedirs('./Mesh/True/Circle')
        os.makedirs('./Mesh/True/Circle/MultiRef'+loca)
        #os.makedirs('./Mesh/True/PerfectRef'+loca)
        #os.makedirs('./Mesh/True/SingleRef'+loca)
    if not os.path.exists('./Mesh/True/LOS'+loca):
        os.makedirs('./Mesh/True/LOS'+loca)
    if not os.path.exists('./Mesh/True/Circle'):
        os.makedirs('./Mesh/True/Circle')
        os.makedirs('./Mesh/True/Circle/MultiRef'+loca)
    if not os.path.exists('./Mesh/True/Circle/MultiRef'+loca):
        os.makedirs('./Mesh/True/Circle/MultiRef'+loca)
    Truename='./Mesh/True/Circle/MultiRef'+loca+'/True.npy'
    TrueRadname='./Mesh/True/Circle/MultiRef'+loca+'/TrueRadA.npy'
    TrueRadBname='./Mesh/True/Circle/MultiRef'+loca+'/TrueRadA.npy'
    TrueQname='./Mesh/True/Circle/MultiRef'+loca+'/TrueQ.npy'
    np.save(Truename,Mesh)
    np.save(TrueRadname,RadMesha)
    np.save(TrueRadBname,RadMeshb)
    np.save(TrueQname,Q)
    TrueFolder='./GeneralMethodPowerFigures/Circle/MultiRef'+loca+'/TrueSlice'
    TruePlotName=TrueFolder+'/NoBoxTrueSliceNref%d'%Nre
    ub=np.amax(Mesh)
    lb=np.amin(Mesh)
    plot_mesh(Mesh,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/Circle/MultiRef'+loca+'/TrueSlice/Rad'
    TruePlotName=TrueFolder+'/NoBoxRadASliceNref%d'%Nre
    ub=max(np.amax(RadMesha),np.amax(RadMeshb))
    lb=min(np.amin(RadMesha),np.amin(RadMeshb))
    plot_mesh(RadMesha,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/Circle/MultiRef'+loca+'/TrueSlice/Rad'
    TruePlotName=TrueFolder+'/NoBoxRadBSliceNref%d'%Nre
    plot_mesh(RadMeshb,TrueFolder,TruePlotName,lb,ub)
    print('True mesh saved at', Truename)
    print('Quality',Q)
    Truename='./Mesh/True/LOS'+loca+'/True.npy'
    TrueRadname='./Mesh/True/LOS'+loca+'/TrueRadA.npy'
    TrueQname='./Mesh/True/LOS'+loca+'/TrueQ.npy'
    np.save(Truename,Mesh1)
    np.save(TrueRadname,RadMesh1)
    np.save(TrueQname,Q1)
    TrueFolder='./GeneralMethodPowerFigures/LOS'+loca+'/TrueSlice'
    TruePlotName=TrueFolder+'/NoBoxTrueSliceNref%d'%Nre
    ub=np.amax(Mesh1)
    lb=np.amin(Mesh1)
    plot_mesh(Mesh1,TrueFolder,TruePlotName,lb,ub)
    TrueFolder='./GeneralMethodPowerFigures/LOS'+loca+'/TrueSlice/Rad'
    TruePlotName=TrueFolder+'/NoBoxRadSliceNref%d'%Nre
    ub=max(np.amax(RadMesh1),np.amax(RadMesh1))
    lb=min(np.amin(RadMesh1),np.amin(RadMesh1))
    plot_mesh(RadMesh1,TrueFolder,TruePlotName,lb,ub)
    print('True mesh saved at', Truename)
    print('Quality',Q)
    #sys.exit(main(sys.argv))

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

epsilon=sys.float_info.epsilon

def makematrix(index=0):
  print('-------------------------------')
  print('True values')
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
  Nra,Nre,h,L    =np.load('Parameters/Raytracing.npy')
  Nra=int(Nra)
  Nre=int(Nre)

  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
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
  Gt            = np.load('Parameters/TxGains'+str(index)+'.npy')
  freq          = np.load('Parameters/frequency'+str(index)+'.npy')
  Freespace     = np.load('Parameters/Freespace'+str(index)+'.npy')
  Pol           = np.load('Parameters/Pol'+str(index)+'.npy')
  c             =Freespace[3]
  khat          =freq*L/c
  lam           =(2*np.pi*c)/freq
  Antpar        =np.array([khat,lam,L])

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat'+str(index)+'.npy')
  refindex     =np.load('Parameters/refindex'+str(index)+'.npy')

  Nx=int(Room.maxxleng()/(h))
  Ny=int(Room.maxyleng()/(h))
  Nz=int(Room.maxzleng()/(h))
  Mesh=np.zeros((Nx,Ny,Nz),dtype=float)

  Tri=Oblist[0]*L
  p0,p1,p2=Tri
  n=np.cross(p0-p1,p0-p2)
  n/=np.sqrt(np.dot(n,n))
  Tx*=L
  y=(Tx-np.dot((p0-Tx),n)*n)*L
  Txhat=2*y-Tx
  #Txyleng=np.sqrt(np.dot(np.dot((p0-Tx),n)*n,np.dot((p0-Tx),n)*n))
  for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      x=Room.coordinate(h,i,j,k)*L
      d=x-Tx
      Txleng=np.linalg.norm(d)
      if i==5 and j==5 and k==5:
          print(x,Tx,d,Txleng,h)
          print(i,j,k)
      if np.dot(d,n)==0:
        # Ray is parallel to plane
        chck=0
        chck2=0
      else:
        d2=x-Txhat
        d2/=np.linalg.norm(d2)
        I2=(np.dot(p0-Txhat,n)/(np.dot(d2,n)))*d2+Txhat
        chck2=inst.InsideCheck(I2,Tri)
        if abs(d).any()>epsilon:
          d/=Txleng
          I=(np.dot(p0-Tx,n)/(np.dot(d,n)))*d+Tx
          chck=inst.InsideCheck(I,Tri)
          blah=(np.dot(n,Tx-x)/(Txleng))
          #print(blah,x,y,Tx,Txleng,Txyleng,np.dot(x,Tx-y))
          theta=np.arcsin(blah)
          #print(Txleng)
          field=(lam/(4*ma.pi*Txleng))*np.exp(1j*khat*Txleng*L)*Pol
        else:
          theta=0
          field=(lam/(4*ma.pi))*(1+0j)*Pol
      if chck==0 and chck2==0:
          pass
      else:
          xhatleng=np.sqrt(np.dot(Txhat-x,Txhat-x))
          cthi=np.cos(theta)
          ctht=np.sqrt(1-(np.sin(theta)/refindex[0,0])**2)
          Refcoef=np.array([[(cthi-Znobrat[0,0]*ctht)/(Znobrat[0,0]*ctht+cthi),.0+0.0j],[.0+0.0j,(Znobrat[0,0]*cthi-ctht)/(Znobrat[0,0]*cthi+ctht)]])
          #field+=(1.0*lam/(4*ma.pi*(xhatleng)))*np.exp(1j*khat*(xhatleng)*L)*np.matmul(Refcoef,Pol)
      Mesh[i,j,k]=10*np.log10((np.absolute(field[0])**2+np.absolute(field[1])**2))
  #i,j,k=Room.position(Tx/L,h)
  #Mesh[i,j,k]+=10*np.log10(Nra)
  return Mesh

if __name__ == '__main__':
    Mesh=makematrix()
    np.save('Parameters/True.npy',Mesh)
    #sys.exit(main(sys.argv))

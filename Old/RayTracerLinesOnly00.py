#!/usr/bin/env python3
# Updated Hayley Wragg 2019-03-15
''' Code to trace rays around a room. This code computes the trajectories only.'''
import numpy as np
import matplotlib.pyplot as mp
import Room as rom
import raytracerfunction as rayt
import sys

if __name__=='__main__':
  #----Retrieve the Raytracing Parameters-----------------------------
  Nra,Nre,h     =np.load('Parameters/Raytracing.npy')

  #----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')
  Tx            =np.load('Parameters/Origin.npy')
  Nob           =len(Oblist)
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')
  Oblist        =np.concatenate((Oblist,OuterBoundary),axis=0)
  Room=rom.room(Oblist,Nob)

  # Calculate the Ray trajectories
  Rays=Room.ray_bounce(Tx, int(Nre), int(Nra))
  filename=str('RayPoints'+str(int(Nra))+'Refs'+str(int(Nre))+'n.npy')
  np.save(filename,Rays)
  print('Ray trajectories saved in:')
  print(filename)
  print('Computation time: ',Room.time)
  exit()


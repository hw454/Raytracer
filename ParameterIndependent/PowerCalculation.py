#!/usr/bin/env python3
# Hayley Wragg 2018-05-29
''' Code to trace rays around a room using cones to account for
spreading. This version does not remove points inside an object.'''
import numpy as np
import matplotlib.pyplot as mp
import roommesh20 as rmes
import room as rom



if __name__=='__main__':
  ##---- Define the room co-ordinates----------------------------------

  # Obstacle 1
  # Obstacle 2
  # Obstacle 3
  # Walls
  # Router location -Tx

  ##----Parameters for the ray tracer----------------------------------
  # No. of reflections -Nre
  # No. of rays -Nra
  # Mesh width -h

  ##----Construct the environment--------------------------------------
  # Oblist=list(ob1,ob2,ob3)
  # Nob=length(Nob)
  # Room=rom.room(Walls,Oblist) - construct the room as a list of the obstacles
  # Mesh=rmes.mesh(Room,h,Nra,Nre) - Construct the mesh using the dimensions of the room

  ##----Run the raytracer to get the mesh--------------------------------------
  # Mesh=Raytracer(Room, Tx, Nra, Nre, h) - Run the ray tracer and get the mesh which contains the information about the field.
  # Save the Mesh

  exit()


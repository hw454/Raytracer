#!/usr/bin/env python3
# Started Hayley Wragg 2018-05-29
# Updated Hayley Wragg 2018-11-21
''' Code to trace rays around a room using cones to account for
spreading. This version will compute this independent of object
parameters and object an array with signatures for the field'''
import numpy as np
import matplotlib.pyplot as mp
import Room as rom
import raytracerfunction as rayt
import DictionarySparseMatrix as DSM

if __name__=='__main__':
  ##---- Define the room co-ordinates----------------------------------
  ##COMMENT In 2D obstacles are just lines which are join co-ordinates.
  ##COMMENT But in 3D this needs to be reconsidered.
  ##COMMENT These lines will only define the boundary and the plane surface of
  ##COMMENT the object needs to be defined.

  # Obstacles are all triangles.
  ob1=np.array([(0.25,0.25,1.0),(0.5 ,0.25,0.0),(0.25,0.5 ,0.0)]) # Obstacle 1
  ob2=np.array([(0.75,0.75,1.0),(0.5 ,0.75,0.0),(0.75,0.5 ,0.0)]) # Obstacle 2
  ob3=np.array([(0.5 ,0.25,1.0),(0.5 ,0.75,0.0),(0.25,0.75,0.0)])# Obstacle 3

  OuterBoundary1=np.array([(0.0,0.0,1.0),(1.0,0.0,0.0),(0.0,0.0,0.0)])
  OuterBoundary2=np.array([(1.0,0.0,1.0),(1.0,1.0,0.0),(1.0,0.0,0.0)])
  OuterBoundary3=np.array([(1.0,1.0,1.0),(0.0,1.0,0.0),(1.0,1.0,0.0)])
  OuterBoundary4=np.array([(0.0,1.0,1.0),(0.0,1.0,0.0),(0.0,0.0,0.0)])
  OuterBoundary=list((OuterBoundary1,OuterBoundary2, OuterBoundary3, OuterBoundary4))
  #- Outer Boundary -
  ##COMMENT list 3D co-ordinates defining the outer boundary of the
  # propagation area being considered. When modelling 2D the third term
  # in all the co-ordinates in 0. For propagation in one room in 2D this
  # is the list of co-ordinates defining the edges of the walls.

  Tx=np.array([0.75,0.25,0.0]) # -Router location -co-ordinate of three real numbers
  #(the third is zero when modelling in 2D).

  ##----Parameters for the ray tracer----------------------------------
  Nre=1 # -No. of reflections - Integer
  Nra=4 #-No. of rays -Integer
  h=0.5 #- Mesh width -Real number in (0,1]

  ##----Construct the environment--------------------------------------
  Oblist=list((ob1,ob2,ob3))
  Nob=len(Oblist)
  Oblist=Oblist+OuterBoundary
  # list of all the obstacles in the room.

  Room=rom.room(Oblist,Nob)
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  # Room contains all the obstacles and walls.

  Mesh=DSM.DS(Nx,Ny,Nz,Nre*(Nra+1),Nob*(Nre+1))
  # This large mesh is initialised as empty. It contains reference to
  # every segment at every position in the room.
  # The history of the ray up to that point is stored in a vector at that reference point.

  # Calculate the Ray trajectories
  Rays=Room.ray_bounce(Tx, Nre, Nra)


  exit()


#!/usr/bin/env python3
# Hayley Wragg 2017-07-10
# Hayley wragg 2017-07-11
''' Code to Reflect a line in an edge without using Shapely '''
import numpy as np
import matplotlib.pyplot as mp
import HayleysPlotting as hp
import reflection as ref
import intersection as ins
import linefunctions as lf
import ray_tracer_test as rtest
import objectsLayered as ob
import roommesh as rmes
import math
import time


if __name__=='__main__':
  np.set_printoptions(precision=2,threshold=1e-12,suppress=True)
  # DEFINE PARAMETERS
  # Define the walls and construct the room
  wall1=ob.Wall_segment(np.array([-1.0,-1.0]),np.array([ 6.0,-1.0]))
  wall2=ob.Wall_segment(np.array([-1.0,-1.0]),np.array([-1.0, 4.0]))
  wall3=ob.Wall_segment(np.array([-1.0, 4.0]),np.array([ 6.0, 4.0]))
  wall4=ob.Wall_segment(np.array([ 6.0, 4.0]),np.array([ 6.0,-1.0]))
  # Define the object1 in the room
  Box1=ob.Wall_segment(np.array([-0.5, 0.0]),np.array([-0.5, 1.0]))
  Box2=ob.Wall_segment(np.array([-0.5, 1.0]),np.array([ 0.0, 1.0]))
  Box3=ob.Wall_segment(np.array([ 0.0, 1.0]),np.array([ 0.0, 0.0]))
  Box4=ob.Wall_segment(np.array([ 0.0, 0.0]),np.array([-0.5, 0.0]))
  # Define Object 2 in the room
  sofa1=ob.Wall_segment(np.array([ 1.0, 0.0]), np.array([ 1.0, 2.0]))
  sofa2=ob.Wall_segment(np.array([ 1.0, 2.0]), np.array([ 3.0, 2.0]))
  sofa3=ob.Wall_segment(np.array([ 3.0, 2.0]), np.array([ 3.0, 0.0]))
  sofa4=ob.Wall_segment(np.array([ 3.0, 0.0]), np.array([ 2.5, 0.0]))
  sofa5=ob.Wall_segment(np.array([ 2.5, 0.0]), np.array([ 2.5, 1.0]))
  sofa6=ob.Wall_segment(np.array([ 2.5, 1.0]), np.array([ 1.5, 1.0]))
  sofa7=ob.Wall_segment(np.array([ 1.5, 1.0]), np.array([ 1.5, 0.0]))
  sofa8=ob.Wall_segment(np.array([ 1.5, 0.0]), np.array([ 1.0, 0.0]))
  # Define the height of the objects
  wallheight=3
  sofaheight=1
  boxheight=0.5
  # Define the ray tracing parameters
  n=100                     # number of rays emitted from source
  origin=(5,1)              # source of the signal
  i=1                       # The figure number for the room plot
  frequency=2.4*1.0E+9      # The wave frequency in Hertz
  streg=100.0 # The initial signal power in db
  spacing=0.1  # Spacing in the grid spaces.
  bounds= np.array([10**-9, 10**2])               # The bounds within which the signal power is useful
  refloss=20

  # Compute Some More Parameters
  m=int(math.ceil(np.log(streg/bounds[0])/np.log(refloss)))     # number of reflections observed
  print('Maximum number of reflections to get lower bound ',m)
  heights=[0,sofaheight,boxheight,wallheight]
  heights.sort()
  # CONSTRUCT THE OBJECTS
  # Contain all the edges of the room
  # Lower level objects
  obstacles1=(wall1,wall2,wall3,wall4,Box1,Box2,Box3,Box4,sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  # Medium level objects
  if sofaheight>boxheight:
    obstacles2=(wall1,wall2,wall3,wall4,sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  else:
    obstacles2=(wall1,wall2,wall3,wall4,Box1,Box2,Box3,Box4)
  # Upper level objects
  walls=(wall1,wall2,wall3,wall4)
  points=list(walls)
  Room=ob.room((walls),heights)
  Room.roomconstruct((walls,obstacles1,obstacles2))
  Room.uniform_ray_tracer(origin,n,i,spacing,frequency,streg,m,refloss)
  Room.uniform_ray_tracer_bounded(origin,n,6,spacing,frequency,streg,m,bounds,refloss)
  filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
  f=open(filename,"w+")
  (x,y)=Room.time
  f.write("Run times for first source location %.8f, %.8f" % (x,y))
  f.close()
  origin=(0,2)
  Room.uniform_ray_tracer(origin,n,11,spacing,frequency,streg,m,refloss)
  Room.uniform_ray_tracer_bounded(origin,n,17,spacing,frequency,streg,m,bounds,refloss)
  # Save run times to file
  filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
  f=open(filename,"a+")
  (x,y)=Room.time
  f.write("Run times for first source location %.8f, %.8f" % (x,y))
  f.close()
  mp.show()
  # TEST err=rtest.ray_tracer_test(Room, origin)
  # TEST PRINT print('error after rtest on room', err)
  exit()

#!/usr/bin/env python3
# Hayley wragg 2017-12-14
''' Code to Run a ray tracer which reflects lines in a two dimensional environment. '''
import numpy as np
import matplotlib.pyplot as mp
import HayleysPlotting as hp
import reflection as ref
import intersection as ins
import linefunctions as lf
import ray_tracer_test as rtest
import objects as ob
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
  # Define the ray tracing parameters

  #n=10                      # number of rays emitted from source
  origin=(5,1)              # source of the signal
  i=1                       # The figure number for the room plot
  frequency=2.4*1.0E+9      # The wave frequency in Mega Hertz
  streg=100.0 # The initial signal power in db
  spacing=0.1  # Spacing in the grid spaces.
  bounds= np.array([10**-9, 10**2])               # The bounds within which the signal power is useful
  refloss=20
  m=int(math.ceil(np.log(streg/bounds[0])/np.log(refloss)))     # number of reflections observed
  print(m)
  # CONSTRUCT THE OBJECTS
  # Contain all the edges of the room
  obstacles=(wall1,wall2,wall3,wall4,Box1,Box2,Box3,Box4,sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  Room=ob.room((obstacles[0]))
  Room.roomconstruct(obstacles)
  for j in range(1,2):
      n=j*100
      print(n)
      i=Room.uniform_ray_tracer(origin,n,i,spacing,frequency,streg,m,refloss)
<<<<<<< HEAD
      i=Room.uniform_ray_tracer_bounded(origin,n,i+1,spacing,frequency,streg,m,bounds,refloss)
=======
      #i=Room.uniform_ray_tracer_bounded(origin,n,i+1,spacing,frequency,streg,m,bounds,refloss)
>>>>>>> 18ecc2cc0c1f654edbe4508ef89bacd728bb08ac
      filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
      f=open(filename,"w+")
      (x,y)=Room.time
      f.write("Run times for first source location %.8f, %.8f" % (x,y))
      #f.write("Estimated P value" % y)
      f.close()
      origin=(0,2)              # source of the signal
      i=Room.uniform_ray_tracer(origin,n,i+1,spacing,frequency,streg,m,refloss)
<<<<<<< HEAD
      i=Room.uniform_ray_tracer_bounded(origin,n,i+1,spacing,frequency,streg,m,bounds,refloss)
=======
      #i=Room.uniform_ray_tracer_bounded(origin,n,i+1,spacing,frequency,streg,m,bounds,refloss)
>>>>>>> 18ecc2cc0c1f654edbe4508ef89bacd728bb08ac
      f=open(filename,"a+")
      for x in Room.time:
        f.write("Run times for second source location %.8f" % x)
      f.close()
      #f.write("Estimated P value" % y)
<<<<<<< HEAD
  #mp.show()
=======
  mp.show()
>>>>>>> 18ecc2cc0c1f654edbe4508ef89bacd728bb08ac
  # TEST err=rtest.ray_tracer_test(Room, origin)
  # TEST PRINT print('error after rtest on room', err)
  exit()

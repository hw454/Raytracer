#!/usr/bin/env python3
# Hayley Wragg 2018-05-10
''' Code to Reflect a line in an edge without using Shapely '''
import numpy as np
import matplotlib.pyplot as mp
import HayleysPlotting as hp
import reflection as ref
import intersection as ins
import linefunctions as lf
#import ray_tracer_test as rtest
import geometricobjects as ob
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

  #n=10                     # number of rays emitted from source
  l=4                      # number of n's
  ave=5                     # number of runs to average over
  origin=(5,1)              # source of the signal
  i=1                       # The figure number for the room plot
  # 2.4 GHz frequency=2.4*1.0E+8      # The wave frequency in Hertz
  #frequency=28*1.0E+8 #28 GHz, ref coef is 1/0.628
  #frequency=2.3E+8 # 2.3GHz, ref coef is 1/0.628
  frequency=5.8E+8 # 5.8GHz, ref coef is 1/0.646 for 45 degree polarisation, 1/0.2512 for vertical polarisation
  powerstreg=1              # The initial signal power in db
  # The spacing is now found inside the uniform ray tracer function spacing=0.25  # Spacing in the grid spaces.
  bounds= np.array([10**-9, 10**2])               # The bounds within which the signal power is useful
  # Reflection Coefficient
  refloss1=1/0.2512
  refloss2=1/0.6457
  m1=int(math.ceil((np.log(powerstreg)-np.log(bounds[0]))/np.log(refloss1)))    # number of reflections observed
  m2=int(math.ceil((np.log(powerstreg)-np.log(bounds[0]))/np.log(refloss2)))    # number of reflections observed
  #m=3
  streg=complex(((1/8.854187817)*1E12*powerstreg)**0.5,0.0)
  print('number of reflections for first coefficient',m1)
  print('number of reflections for second coefficient',m2)
  # CONSTRUCT THE OBJECTS
  # Contain all the edges of the room
  obstacles=(wall1,wall2,wall3,wall4,Box1,Box2,Box3,Box4,sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  Room=ob.room((obstacles[0]))
  Room.roomconstruct(obstacles)
  for j in range(1,l):
      n=j*300
      print('number of rays', n)
      # First source location and first reflection coefficient
      origin=(5,1)              # source of the signal
      i,spacing,grid1=Room.uniform_ray_tracer(origin,n,ave,i,frequency,streg,m1,refloss1)
      i=i+2
      # Store the run time
      filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
      f=open(filename,"w+")
      (x,y)=Room.time
      f.write("Run times for first source location with first ref coef %.8f, %.8f" % (x,y))
      f.close()
      # Second source location and first reflection coefficient
      origin=(0,2)              # source of the signal
      i,spacing,grid1=Room.uniform_ray_tracer(origin,n,ave,i,frequency,streg,m1,refloss1)
      f=open(filename,"a+")
      for x in Room.time:
        f.write("Run times for second source location with first ref coef%.8f" % x)
      f.close()
      # Repeat for the second set of reflection coefficients
      # First source location and second reflection coefficient
      origin=(5,1)              # source of the signal
      #Attempt at spreading the initial signal strength. This is actually accounted for in C_lambda streg=stregstart/n
      i,spacing,grid1=Room.uniform_ray_tracer(origin,n,ave,i,frequency,streg,m2,refloss2)
      i=i+2
      #i,spacing=Room.uniform_ray_tracer_bounded(origin,n,i+1,frequency,streg,m,bounds,refloss)
      filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
      f=open(filename,"w+")
      for x in Room.time:
        f.write("Run times for first source location with second ref coef %.8f" % x)
      f.close()
      # Second source location and second reflection coefficient
      origin=(0,2)              # source of the signal
      i,spacing,grid1=Room.uniform_ray_tracer(origin,n,ave,i,frequency,streg,m2,refloss2)
      f=open(filename,"a+")
      for x in Room.time:
        f.write("Run times for second source location with second ref coef%.8f" % x)
      f.close()
  #mp.show()
  # TEST err=rtest.ray_tracer_test(Room, origin)
  # TEST PRINT print('error after rtest on room', err)
  exit()

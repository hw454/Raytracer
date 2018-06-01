#!/usr/bin/env python3
# Hayley Wragg 2018-05-29
''' Code to trace rays around a room using cones to account for
spreading. This version does not remove points inside an object.'''
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
  l=2                       # number of n's
  ave=5                     # number of runs to average over
  origin=(5,1)              # source of the signal
  i=1                       # The figure number for the room plot
  frequency=5.8E+8 # 5.8GHz, ref coef is 1/0.646 for 45 degree polarisation, 1/0.2512 for vertical polarisation
  powerstreg=1              # The initial signal power in db
  # The spacing is now found inside the uniform ray tracer function spacing=0.25  # Spacing in the grid spaces.
  bounds= np.array([10**-9, 10**2])               # The bounds within which the signal power is useful
  # Reflection Coefficient
  refloss=1/0.2512
  #refloss2=1/0.6457
  m=int(math.ceil(np.log(powerstreg/bounds[0])/np.log(refloss)))     # number of reflections observed
  streg=complex(((1/8.854187817)*1E12*powerstreg)**0.5,0.0)          # Initial field strength
  origin1=(5,1)              # source of the signal
  origin2=(0,2)              # source of the signal
  outsidepoint1=(4,1)              # point for testing whether another point is inside or outside an object
  outsidepoint2=(5,2)              # point for testing whether another point is inside or outside an object
  outsidepoint3=(1,3)              # point for testing whether another point is inside or outside an object
  outsidepoint4=(0,3)              # point for testing whether another point is inside or outside an object
  print('number of reflections',m)
  # CONSTRUCT THE OBJECTS
  # Contain all the edges of the room
  obstacles=(wall1,wall2,wall3,wall4,Box1,Box2,Box3,Box4,sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  box=(Box1,Box2,Box3,Box4)
  sofa=(sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  Room=ob.room((obstacles[0]))
  Room.roomconstruct(obstacles)
  Room.add_inside_objects(box)
  Room.add_inside_objects(sofa)
  for j in range(1,l):
      n=j*100
      print('number of rays', n)
      #Attempt at spreading the initial signal strength. This is actually accounted for in C_lambda streg=stregstart/n
      i,spacing,grid1=Room.uniform_ray_tracer(origin1,outsidepoint1,outsidepoint2,n,ave,i,frequency,streg,m,refloss)
      # First attempt at averaging
      #for it in range(1,ave):
        #i,spacing,grid2=Room.uniform_ray_tracer(origin,n,ave,i,frequency,streg,m,refloss)
        #grid1.grid=grid1.grid+grid2.grid
      #grid1.grid=grid1.grid/m
      #grid1.hist(i)
      #mp.figure(i)
      #mp.title('Cumulative frequency of averaged field strength')
      #mp.grid()
      #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/AveragedOnBothRNDCumsum'+str(i)+'.png', bbox_inches='tight')
      #mp.figure(i+1)
      #mp.title('Histrogram of averaged field strength')
      #mp.grid()
      #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/AveragedOnBothHistogramNoBounds'+str(i)+'.png',bbox_inches='tight')
      i=i+2
      #i,spacing=Room.uniform_ray_tracer_bounded(origin,n,i+1,frequency,streg,m,bounds,refloss)
      filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
      f=open(filename,"w+")
      (x,y)=Room.time
      f.write("Run times for first source location %.8f, %.8f" % (x,y))
      #f.write("Estimated P value" % y)
      f.close()
      #i,spacing,grid=Room.uniform_ray_tracer(origin,n,i+1,frequency,streg,m,refloss)
      i,spacing,grid1=Room.uniform_ray_tracer(origin2,outsidepoint3,outsidepoint4,n,ave,i,frequency,streg,m,refloss)
      #for it in range(1,ave):
        #i,spacing,grid2=Room.uniform_ray_tracer(origin,n,ave,i,frequency,streg,m,refloss)
        #grid1.grid=grid1.grid+grid2.grid
      #grid1.grid=grid1.grid/m
      #grid1.hist(i)
      #mp.figure(i)
      #mp.title('Cumulative frequency of averaged field strength')
      #mp.grid()
      #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/AveragedOnBothRNDCumsum'+str(i)+'.png', bbox_inches='tight')
      #mp.figure(i+1)
      #mp.title('Histrogram of averaged field strength')
      #mp.grid()
      #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/AveragedOnBothHistogramNoBounds'+str(i)+'.png',bbox_inches='tight')
      #i=i+2
      #i,spacing=Room.uniform_ray_tracer_bounded(origin,n,i+1,frequency,streg,m,bounds,refloss)
      f=open(filename,"a+")
      for x in Room.time:
        f.write("Run times for second source location %.8f" % x)
      f.close()
      #f.write("Estimated P value" % y)
  #mp.show()
  # TEST err=rtest.ray_tracer_test(Room, origin)
  # TEST PRINT print('error after rtest on room', err)
  exit()

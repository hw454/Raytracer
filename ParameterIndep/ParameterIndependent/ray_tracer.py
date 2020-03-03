#!/usr/bin/env python3
# Hayley Wragg 11-10-2018
''' The code runs a ray tracer in a defined environment and saves a output mesh
 of signatures which correspond to the power at each point in a mesh relative
 to the environment coefficients.'''

import numpy as np
import matplotlib.pyplot as mp
import Room as rom

if __name__=='__main__':
  ##---- Define the room co-ordinates----------------------------------

  # Obstacle 1
  # Obstacle 2
  # Obstacle 3
  # Walls
  # Router location
  # source=np.array([x,y,z])
  origin1=(5,1,0)              # source of the signal
  # Points which are outside objects for removing function
  outsidepoint1=(4,1,0)        # point for testing whether another point is inside or outside an object
  outsidepoint2=(5,2,0)        # point for testing whether another point is inside or outside an object
  outsidepoint3=(1,3,0)        # point for testing whether another point is inside or outside an object
  outsidepoint4=(0,3,0)        # point for testing whether another point is inside or outside an object
  # Define the walls and construct the room
  wall1=Room.Wall_segment(np.array([-1.0,-1.0,0]),np.array([ 6.0,-1.0]))
  wall2=Room.Wall_segment(np.array([-1.0,-1.0,0]),np.array([-1.0, 4.0]))
  wall3=Room.Wall_segment(np.array([-1.0, 4.0,0]),np.array([ 6.0, 4.0]))
  wall4=Room.Wall_segment(np.array([ 6.0, 4.0,0]),np.array([ 6.0,-1.0]))
  walls=list(wall1,wall2,wall3,wall4)
  # Define the object1 in the room
  Box1=Room.Wall_segment(np.array([-0.5, 0.0]),np.array([-0.5, 1.0]))
  Box2=Room.Wall_segment(np.array([-0.5, 1.0]),np.array([ 0.0, 1.0]))
  Box3=Room.Wall_segment(np.array([ 0.0, 1.0]),np.array([ 0.0, 0.0]))
  Box4=Room.Wall_segment(np.array([ 0.0, 0.0]),np.array([-0.5, 0.0]))
  box=Room.obstacle(Box1,Box2,Box3,Box4)
  # Define Object 2 in the room
  sofa1=Room.Wall_segment(np.array([ 1.0, 0.0]), np.array([ 1.0, 2.0]))
  sofa2=Room.Wall_segment(np.array([ 1.0, 2.0]), np.array([ 3.0, 2.0]))
  sofa3=Room.Wall_segment(np.array([ 3.0, 2.0]), np.array([ 3.0, 0.0]))
  sofa4=Room.Wall_segment(np.array([ 3.0, 0.0]), np.array([ 2.5, 0.0]))
  sofa5=Room.Wall_segment(np.array([ 2.5, 0.0]), np.array([ 2.5, 1.0]))
  sofa6=Room.Wall_segment(np.array([ 2.5, 1.0]), np.array([ 1.5, 1.0]))
  sofa7=Room.Wall_segment(np.array([ 1.5, 1.0]), np.array([ 1.5, 0.0]))
  sofa8=Room.Wall_segment(np.array([ 1.5, 0.0]), np.array([ 1.0, 0.0]))
  sofa=Room.obstacle(sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)

  Enviro=Room.room(walls)

  # Define the ray tracing parameters
  # DEFINE PARAMETERS FOR THE CODE
  #m=int(math.ceil(np.log(powerstreg/bounds[0])/np.log(1/measuredref)))     # number of reflections observed
  m= 4                      # number of reflections observed
  #n=10                       # number of rays emitted from source
  jmax=3                   # number of n's
  n=10                       # Initial number of rays will be 2*n
  ave=5                      # number of runs to average over
  origin=(5,1)               # source of the signal
  i=1                        # The figure number for the room plot
  origin2=(0,2)              # source of the signal

  print('Ray tracer begun for ',m,'number of reflections')
  Room=ob.room((obstacles[0]))
  Room.roomconstruct(obstacles)
  Room.add_inside_objects(box)
  Room.add_inside_objects(sofa)
  # The spacing is now found inside the uniform ray tracer function spacing=0.25  # Spacing in the grid spaces.


  ##----Parameters for the ray tracer----------------------------------

  # No. of reflections
  # Nrf=
  # No. of rays
  # Nry=
  # Mesh width
  # delta=

  ##----Construct the environment--------------------------------------
  # Room = rom.RoomConstruct(obstacles)
  # Mesh=Rays.raytracer(Room,source,Nry,Nrf)

  exit()

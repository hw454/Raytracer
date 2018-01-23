#!/usr/bin/env python3
# Hayley Wragg 2017-07-10
# Hayley wragg 2017-07-11
''' Code to Reflect a line in an edge without using Shapely '''

from math import atan2,hypot,sqrt,copysign
import numpy as np
import matplotlib.pyplot as mp
import HayleysPlotting as hp
import reflection as ref
import intersection as ins
import linefunctions as lf
import objects as ob

def roomconstruct(walls):
  ''' Takes in a set of wall segments and constructs a room object
  containing them all'''
  Room=ob.room((walls[0]))
  i=0
  for wall in walls[1:]:
      room=Room.add_wall(wall)
  return Room


def Plotray(ray):


if __name__=='__main__':
  np.set_printoptions(precision=2,threshold=1e-12,suppress=True)
  # Define the walls and construct the room
  wall1=ob.Wall_segment(np.array([-1.5,0.0]),np.array([-0.5,4.0]))
  wall2=ob.Wall_segment(np.array([0.5,0.0]),np.array([0.5,1.0]))
  wall3=ob.Wall_segment(np.array([0.20,0.0]),np.array([5.0,0.0]))
  walls=(wall1,wall2,wall3)
  # Construct a room ith the previously constructed walls
  Room=roomconstruct(walls)
  # Construct the ray
  ray=ob.Ray((0,0),(1,1))
  # Reflect the ray
  print('before reflect', ray.ray)
  ray.reflect(Room)
  print('After one reflection', ray.ray)
  ray.reflect(Room)
  print('second',ray.ray)
  mp.figure(1)
  rayline1=ray.ray[0]
  for rayline2 in ray.ray[0:-1]:
    hp.Plotedge(np.array([rayline1, rayline2]),'r')
    rayline1=rayline2
  hp.Plotline(ray.ray[-2:],Room.maxleng(),'b')
  for wall in walls:
    hp.Plotedge(np.array(wall.p),'g')
  mp.show()
  exit()

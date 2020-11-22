#!/usr/bin/env python3
# Hayley wragg 2017-07-11
''' Code to test the ray_tracer on the environment. Testing just the
first intersection for uniformly distributed rays.'''

from math import atan2
import numpy as np
import objects as ob
import math as ma
from math import sin,cos,atan2,hypot

def ray_tracer_test(Room,origin):
  pi=4*np.arctan(1)
  maxj=10
  err=0
  for j in range(-maxj,maxj):
    theta=(j*pi)/maxj
    r=Room.maxleng()
    xtil=ma.cos(theta)
    ytil=ma.sin(theta)
    x= r*xtil+origin[0]
    y= r*ytil+origin[1]
    ray=ob.Ray(origin,(x,y))
    err=ray.raytest(Room,err)
  return err

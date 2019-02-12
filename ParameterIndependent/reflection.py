
#!/usr/bin/env python3
# Keith Briggs 2017-02-02
# Hayley Wragg 2017-03-28
# Hayley Wragg 2017-04-12
# Hayley Wragg 2017-05-15
''' Code to Reflect a line in an edge without using Shapely '''

from math import atan2,hypot,sqrt,copysign
import numpy as np
import matplotlib.pyplot as mp
import HayleysPlotting as hp
import intersection as inter
from numpy import *
import linefunctions as lf

def try_reflect_ray(ray,edge):
  ''' Reflection of a ray which goes through the points source or
  previous intersect and next intersect.ray.get_origin '''
  # Find the distances which need to be translated
  trdist=-ray[1]
  direc=lf.Direction(ray)
  # Translate the points before the reflection
  o=trdist
  ray+=trdist
  edge+=trdist
  # Find the image of the ray in the edge
  Image=ray[1]+direc
  # Use the unit edge instead of the exact edge for the reflection
  unitedge=np.array([ray[1],edge[1]])
  unitedge=(1.0/lf.length(unitedge))*unitedge
  # Find the normal to the edge
  normedge=np.array([[unitedge[1][1],-unitedge[1][0]],ray[1]])
  # Find the reflection using the Image
  ReflPt=Image-2*np.dot(normedge[0],Image)*normedge[0]
  #Translate Back
  ray-=trdist
  edge-=trdist
  normedge-=trdist
  ReflPt-=trdist
  o-=trdist
  return np.array([ray[1], ReflPt]), normedge

def try_3D_reflect_ray(ray,plane):
  ''' Take a ray and find it's reflection in a plane. '''
    # Find the distances which need to be translated
  trdist=-ray[1]
  direc=lf.Direction3D(ray)
  # Translate the points before the reflection
  o=trdist
  ray+=trdist
  plane[0]+=trdist
  # Find the image of the ray in the edge
  Image=ray[1]+direc
  # Find the normal to the edge
  normedge=plane[1]
  # Find the reflection using the Image
  ReflPt=Image-2*np.dot(normedge,Image)*normedge
  #Translate Back
  ray-=trdist
  plane[0]-=trdist
  normedge-=trdist
  ReflPt-=trdist
  o-=trdist
  return np.array([ray[1], ReflPt]), normedge

def errorcheck(err,ray,ref,normedge):
  ''' Take the input ray and output ray and the normal to the edge,
  check that both vectors have the same angle to the normal'''
  # Convert to the Direction vectors
  ray=lf.Direction(ray)
  normedge=lf.Direction(normedge)
  ref=lf.Direction(ref)
  # Find the angles
  #FIXME check the cosine rule
  theta1=np.arccos(abs(np.dot(ray,normedge))/(np.linalg.norm(ray)*np.linalg.norm(normedge)))
  theta2=np.arccos(abs(np.dot(ref,normedge))/(np.linalg.norm(ref)*np.linalg.norm(normedge)))
  if (abs(theta1-theta2)>1.0E-7):
    err+=1.0
  return err


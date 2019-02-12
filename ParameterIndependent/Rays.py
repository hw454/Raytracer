#!/usr/bin/env python3
# Hayley Wragg 2018-05-10
''' Code to construct the ray-tracing objects. Constructs wall-segments,
 rays'''

from math import atan2,hypot,sqrt,copysign
import numpy as np
import reflection as ref
import intersection as ins
import linefunctions as lf
import HayleysPlotting as hp
import matplotlib.pyplot as mp
import math as ma
from math import sin,cos,atan2
import roommesh as rmes
import time as t
import random as rnd


def Raytracer(Room,source,Nrays,Nrefs):
    '''Shoot Nrays from the source location source and track them for Nrefs.
    Track which objects are hit and the angle they are hit at then output a list
    of rays which each ray having the list of points it's hit and an array
    corresponding to the object hit and the angle hit at. '''
    Frac=2*ma.pi/Nrays
    r=Room.maxleng()
    RayArray=[]
    for count in range(0,Nrays+1):
        theta=count*Fraction
        p1=r*np.array([ma.cos(theta),ma.sin(theta),0])
        RayCount=Ray(source,p1)
        #RayCount.multiref(Room,Nrefs)
        #RayArray.append(RayCount)
    return RayArray

class Ray:
  ''' A ray is a representation of the the trajectory of a reflecting line
  and it's reflections. Ray.points is an array of co-ordinates representing
  the collision points with the last term being the direction the ray ended in.
  And Ray.reflections is an array containing tuples of the angles of incidence
  and the number referring to the position of the obstacle in the obstacle list
  '''
  def __init__(origin,direction):
    s.points=np.vstack(
      (np.array(origin,   dtype=np.float),
       np.array(direction,dtype=np.float),
    ))
    s.reflections=np.vstack()
  def __str__(s):
    return 'Ray(\n'+str(s.ray)+')'
  def _get_intersection(s):
    ''' The second to last term in the np array is the starting
    co-ordinate of the travelling ray '''
    return s.ray[-2]
  def _get_direction(s):
    ''' The direction of the travelling ray is the last term in the ray
    array. '''
    return s.ray[-1]
  def _get_travellingray(s):
    '''The ray which is currently travelling. Should return the recent
    origin and direction. '''
    return [s._get_origin(), s._get_direction()]
  def wall_collision_point(s,wall_segment):
    ''' intersection of the ray with a wall_segment '''
    return ins.intersection(s._get_travellingray(),wall_segment)
  def room_collision_point(s,room):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room. Returns the intersection point and the
    wall intersected with '''
    # Retreive the Maximum length from the Room
    leng=room.maxleng()
    # Initialise the point and wall
    rwall=room.walls[0]
    rcp=s.wall_collision_point(rwall)
    # Find the intersection with all the walls and check which is the
    #closest. Verify that the intersection is not the current origin.
    for wall in room.walls[1:]:
      cp=s.wall_collision_point(wall)
      if (cp[0] is not None and (cp!=s.ray[-2]).all()):
        leng2=s.ray_length(cp)
        if (leng2<leng) :
          leng=leng2
          rcp=cp
          rwall=wall
    return rcp, rwall
  def ray_length(s,inter):
    '''The length of the ray upto the intersection '''
    o=s._get_origin()
    ray=np.array([o,inter])
    return lf.length(ray)
  def reflect(s,room):
    ''' finds the reflection of the ray inside a room'''
    cp,wall=s.room_collision_point(room)
    # Check that a collision does occur
    if cp[0] is None: return
    else:
      # Construct the incoming array
      origin=s._get_origin()
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray,n=ref.try_reflect_ray(ray,wall.p)
      # update self...
      s.ray[-1]=cp
      s.ray=np.vstack((s.ray,lf.Direction(refray)))
    return
  def multiref(s,room,m):
    ''' Takes a ray and finds the first five reflections within a room'''
    for i in range(1,m+1):
      s.reflect(room)
    # print('iteraction', i, 'ray', s.ray)
    return
  def Plotray(s,room):
    ''' Plots the ray from point to point and the final travelling ray
    the maximum length in the room '''
    rayline1=s.ray[0]
    wid=7
    for rayline2 in s.ray[0:-1]:
      wid=wid*0.5
      hp.Plotray(np.array([rayline1, rayline2]),'BlueViolet',wid)
      rayline1=rayline2
    #hp.Plotline(s.ray[-2:],room.maxleng(),'b')
    return
  def raytest(s,room,err):
    ''' Checks the reflection for errors'''
    cp,wall=s.room_collision_point(room)
    # Check that a collision does occur
    if cp[0] is None: return
    else:
      # Construct the incoming array
      origin=s._get_origin()
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray,n=ref.try_reflect_ray(ray,wall.p)
      err=ref.errorcheck(err,ray,refray,n)
      # update self...
      s.ray[-1]=cp
      s.ray=np.vstack((s.ray,lf.Direction(refray)))
      print('ray',ray, 'refray', refray, 'error', err)
    return err
  def heatmapray(s,Mesh,streg,freq,spacing,refloss,N):
    i=0
    streg=streg*(299792458/(freq*4*ma.pi))
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency,N)
      i+=1
    #Mesh.plot()
    return Mesh
  #def heatmaprayrndref(s,Mesh,streg,freq,spacing,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #phase=rnd.uniform(0,2)
      #refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.plot()
    #return Mesh
  #def heatmaprayrndsum(s,Mesh,streg,freq,spacing,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.plot()
    #return Mesh
  def heatmaprayrndboth(s,Mesh,streg,freq,spacing,refloss,N):
    i=0
    streg=streg*(299792458/(freq*4*ma.pi))
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      phase=0.0 #rnd.uniform(0,2)
      refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency,N)
      i+=1
    #Mesh.plot()
    return Mesh
  #def heatmapraybounded(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh
  #def heatmaprayboundedrndref(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #phase=rnd.uniform(0,2)
      #refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh
  #def heatmaprayboundedrndsum(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh
  #def heatmaprayboundedboth(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #phase=rnd.uniform(0,2)
      #refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh





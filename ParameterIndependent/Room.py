#!/usr/bin/env python3
# Hayley Wragg 2017-07-18
''' Code to construct the mesh of the room '''

from math import atan2,hypot,sqrt,copysign
from math import sin,cos,atan2,log
import numpy                as np
import reflection           as ref
import intersection         as ins
import linefunctions        as lf
import HayleysPlotting      as hp
import matplotlib.pyplot    as mp
import math                 as ma
import numpy.linalg         as lin
import random               as rnd
import Rays                 as ry
import time                 as t
from itertools import product

epsilon=2.22e-32

#class obstacle_segment:
  #' a line segment from p0 to p1 '
  #def __init__(s,p0,p1):
    #assert not (p0==p1).all()
    #s.p=np.vstack(
      #(np.array(p0,dtype=np.float),
       #np.array(p1,dtype=np.float),
    #))
  #def __getitem__(s,i):
    #return s.p[i]
  #def firstpoint(s):
    #return s.p[0]
  #def secondpoint(s):
    #return s.p[1]
  #def __str__(s):
    #return 'Wall_segment('+str(list(s.p))+')'

class room:
  ' A group of obstacle_segments and the time for the run'
  def __init__(s,obst):
    s.obst=obst
    RoomP=obst[0]
    for j in range(1,len(obst)):
      RoomP=np.concatenate((RoomP,obst[j]),axis=0)
    s.points=RoomP
    # Points is the array of all the co-ordinates which form the surfaces in the room
    s.Nob=len(obst)
    # Nob is the number of surfaces forming obstacles in the room.
    s.maxlength=np.zeros(4)
    # The initial maxleng, maxxlength, maxylength and maxzlength are 0, this value is changed once computed
    s.inside_points=np.array([])
    # The inside points line within obstacles and are used to detect if a ray is inside or outside.
    s.time=np.array([0.0])
    # The time taken for a computation is stored in time.
  def __get_obst__(s,i):
   ''' Returns the ith surface obstacle of the room s'''
   return s.obst[i]
  def __get_insidepoint__(s,i):
   ''' Returns the ith inside point of s '''
   return s.inside_points[i]
  def __set_obst__(s,obst0):
    ''' Adds wall to the walls in s, and the points of the wall to the
    points in s '''
    s.obst+=(obst0,)
    s.points+=obst0
    return
  def __set_insidepoint__(s,p):
    ''' Puts a point p in the inside points array '''
    s.inside_points+=p
    return
  def __str__(s):
    return 'Rooom('+str(list(s.obst))+')'
  def maxleng(s):
    ''' Finds the maximum length contained in the room '''
    # Has the maxlength in the room been found yet? If no compute it.
    if abs(s.maxlength[0])<epsilon:
      leng=0
      p1=s.points[-1]
      for p2 in s.points:
        leng2=lf.length(np.array([p1,p2]))
        if leng2>leng:
          leng=leng2
      s.maxlength[0]=leng
    # If yes then return it
    else: leng=s.maxlength[0]
    return leng
  def maxxleng(s):
    ''' Finds the maximum length contained in the room in the x plane'''
    if abs(s.maxlength[1])<epsilon:
      leng=0
      m=len(s.points)
      p1=s.points[-1][0]
      for j in range(0,m):
        p2=s.points[j][0]
        leng2=lf.length(np.array([p1,p2]))
        if leng2>leng:
          leng=leng2
      s.maxlength[1]=leng
    # If yes then return it
    else: leng=s.maxlength[1]
    return leng
  def maxyleng(s):
    ''' Finds the maximum length contained in the room in the y plane'''
    if abs(s.maxlength[2])<epsilon:
      leng=0
      m=len(s.points)
      p1=s.points[-1][0]
      for j in range(0,m):
        p2=s.points[j][0]
        leng2=lf.length(np.array([p1,p2]))
        if leng2>leng:
          leng=leng2
      s.maxlength[2]=leng
    # If yes then return it
    else: leng=s.maxlength[2]
    return leng
  def maxzleng(s):
    ''' Finds the maximum length contained in the  in the z plane '''
    if abs(s.maxlength[3])<epsilon:
      leng=0
      m=len(s.points)
      p1=s.points[-1][0]
      for j in range(0,m):
        p2=s.points[j][0]
        leng2=lf.length(np.array([p1,p2]))
        if leng2>leng:
          leng=leng2
      s.maxlength[3]=leng
    # If yes then return it
    else: leng=s.maxlength[3]
    return leng
  def ray_bounce(s,Tx,Nre,Nra):
    ''' Traces ray's uniformly emitted from an origin around a room.
    Number of rays is n, number of reflections m. Output
    Nra x Nre x Dimension array containing each ray as a squence of it's
    intersection points and the corresponding object number.'''
    start_time    =t.time()         # Start the time counter
    r             =s.maxleng()
    raylist       =np.empty([Nra+1, Nre+1,4])
    deltheta      =np.sqrt(2.0/(Nra))*ma.pi
    Nra           =int(np.sqrt(Nra/2.0)-1)*int(np.sqrt(Nra*2.0))+1
    xysteps       =int(2.0*ma.pi/deltheta)
    zsteps        =int(ma.pi/deltheta+1)
    theta1        =deltheta*np.arange(xysteps)
    theta2        =deltheta*np.arange(zsteps)
    xydirecs      =np.transpose(r*np.vstack((np.cos(theta1),np.sin(theta1))))
    z             =r*np.tensordot(np.cos(theta2),np.ones(xysteps),axes=0)
    directions    =np.zeros((Nra,4))
    directions[0] =np.array([0.0,0.0,r,0.0])
    directions[-1]=np.array([0.0,0.0,-r,0.0])
    # Form the xyz co-ordinates matrix
    #FIXME try to form this without a loop
    for j in range(1,zsteps-1):
      st=(j-1)*xysteps+1
      ed=(j)*xysteps+1
      sinalpha=np.sin(theta2[j])
      coords=np.c_[sinalpha*xydirecs,z[j]]
      directions[st:ed]=np.c_[coords,np.zeros(xysteps)]
    # Iterate through the rays find the ray reflections
    # FIXME the rays are independent of each toher so this is easily parallelisable
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      raystart.multiref(s,Nre)
      raylist[it]=raystart.points[0:-2]
    s.time=start_time-t.time()
    return raylist
  def xbounds(s):
    xarray=np.vstack(np.array([s.obst[0][0][0]]))
    for obst0 in s.obst:
      xarray=np.vstack((xarray,obst0[0][0],obst[1][0]))
    return np.array([min(xarray)[0],max(xarray)[0]])
  def ybounds(s):
    yarray=np.vstack(np.array([s.obst[0][0][1]]))
    for obst0 in s.obst:
      yarray=np.vstack((yarray,obst0[0][1],obst0[1][1]))
    return np.array([min(yarray)[0],max(yarray)[0]])
  def zbounds(s):
    yarray=np.vstack(np.array([s.obst[0][0][2]]))
    for obst0 in s.obst:
      yarray=np.vstack((yarray,obst0[0][2],obst0[1][2]))
    return np.array([min(yarray)[0],max(yarray)[0]])
  #def add_inside_objects(s,corners):
    #if s.objectcorners.shape[0]==0:
      #n=0
      #j=0
    #else:
      #n=s.objectcorners[-1][0]
      #j=n
    #for x in corners:
      #if j==0: s.inside_points=x.firstpoint()
      #else: s.inside_points=np.vstack((s.inside_points,x.firstpoint()))
      #j+=1
    #if n==0: s.objectcorners=np.array([(n,j-1)])
    #else:
      #s.objectcorners=np.vstack((s.objectcorners,np.array([(n,j-1)])))
    #return


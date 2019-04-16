#!/usr/bin/env python3
# Hayley Wragg 2017-07-18
''' Code to construct the mesh of the room '''
import numpy                as np
import reflection           as ref
import intersection         as ins
import linefunctions        as lf
import math                 as ma
import numpy.linalg         as lin
import Rays                 as ry
import time                 as t
from itertools import product


class room:
  ' A group of obstacle_segments and the time for the run'
  def __init__(s,obst,Nob):
    # obst is the array of triangles representing the surfaces of the room
    s.obst=obst
    # points is the array of the points which form all the surfaces of the room.
    RoomP=obst[0]
    for j in range(1,len(obst)):
      RoomP=np.concatenate((RoomP,obst[j]),axis=0)
    s.points=RoomP
    # Nob is the number of surfaces that form the obstacles in the room
    s.Nob=Nob
    # The inside points are the points in the room that are inside an obstacle.
    s.inside_points=np.array([])
    s.time=np.array([0.0])
  def __setobst__(s,obj):
   '''Add a surface to the rooms obstacles'''
   s.obst=np.vstack((s.obst,obj))
   for j in range(0,len(obj)):
     s.points=np.concatenate((s.points,obj[j]),axis=0)
   s.Nob+=1
   return
  def __getobst__(s,i):
   ''' Returns the ith obst of s '''
   return s.obst[i]
  def __getinsidepoint__(s,i):
   ''' Returns the ith inside point of s '''
   return s.inside_points[i]
  def __setinsidepoints__(s,p):
    '''Adds p to the list of inside points in the Room'''
    if s.insidepoints:
      s.insidepoints=np.vstack((s.insidepoints,p))
    else:
      s.insidepoints=p
    return
  def add_obst(s,obst0):
    ''' Adds wall to the walls in s, and the points of the wall to the
    points in s '''
    s.obst+=(obst0,)
    s.points+=obst0
    return
  def __str__(s):
    return 'Room('+str(list(s.obst))+')'
  def maxleng(s):
    ''' Finds the maximum length contained in the room '''
    leng=0
    p1=s.points[-1]
    for p2 in s.points:
      leng2=lf.length(np.array([p1,p2]))
      if leng2>leng:
        leng=leng2
    return leng
  def maxxleng(s):
    ''' Finds the maximum length contained in the room in the x plane'''
    leng=0
    m=len(s.points)
    p1=s.points[-1][0]
    for j in range(0,m):
      p2=s.points[j][0]
      leng2=lf.length(np.array([p1,p2]))
      if leng2>leng:
        leng=leng2
    return leng
  def maxyleng(s):
    ''' Finds the maximum length contained in the room in the y plane'''
    leng=0
    m=len(s.points)
    p1=s.points[-1][1]
    for j in range(0,m):
      p2=s.points[j][1]
      leng2=lf.length(np.array([p1,p2]))
      if leng2>leng:
        leng=leng2
    return leng
  def maxzleng(s):
    ''' Finds the maximum length contained in the  in the z plane '''
    leng=0
    m=len(s.points)
    p1=s.points[-1][2]
    for j in range(0,m):
      p2=s.points[j][2]
      leng2=lf.length(np.array([p1,p2]))
      if leng2>leng:
        leng=leng2
    return leng
  def ray_bounce(s,Tx,Nre,Nra):
    start_time=t.time()         # Start the time counter
    ''' Traces ray's uniformly emitted from an origin around a room.
    Number of rays is n, number of reflections m. Output
    Nra x Nre x Dimension array containing each ray as a squence of it's
    intersection points and the corresponding object number.'''
    r           =s.maxleng()
    raylist     =np.empty([Nra+1, Nre+1,4])
    theta       =((2.0*ma.pi)/Nra)*np.arange(Nra)
    TxMat       =np.array([Tx[0]*np.ones(Nra),Tx[1]*np.ones(Nra),Tx[2]*np.ones(Nra)])
    directions  =np.transpose(r*np.vstack((np.cos(theta),np.sin(theta),np.zeros(Nra),np.zeros(Nra),np.zeros(Nra))))#FIXME rotate in z axis too.
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      raystart.multiref(s,Nre)
      raylist[it]=raystart.points[0:-2]
    stop_time=t.time()
    s.time[0]=stop_time-start_time
    return raylist
  def Plotroom(s,origin,width):
    ''' Plots all the edges in the room '''
    mp.plot((origin),marker='x',c='r')
    for obst0 in s.obst:
      hp.Plotedge(obst0,'g',width)
    return
  # def xbounds(s):
    # xarray=np.vstack(np.array([s.obst[0][0][0]]))
    # for obst0 in s.obst:
      # xarray=np.vstack((xarray,obst0[0][0],obst[1][0]))
    # return np.array([min(xarray)[0],max(xarray)[0]])
  # def ybounds(s):
    # yarray=np.vstack(np.array([s.obst[0][0][1]]))
    # for obst0 in s.obst:
      # yarray=np.vstack((yarray,obst0[0][1],obst0[1][1]))
    # return np.array([min(yarray)[0],max(yarray)[0]])
  # def zbounds(s):
    # yarray=np.vstack(np.array([s.obst[0][0][2]]))
    # for obst0 in s.obst:
      # yarray=np.vstack((yarray,obst0[0][2],obst0[1][2]))
    # return np.array([min(yarray)[0],max(yarray)[0]])

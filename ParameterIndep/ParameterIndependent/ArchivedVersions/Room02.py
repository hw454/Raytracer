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
import Rays01               as ry
import time                 as t
from itertools import product


class obstacle_segment:
  ' a line segment from p0 to p1 '
  def __init__(s,p0,p1):
    assert not (p0==p1).all()
    s.p=np.vstack(
      (np.array(p0,dtype=np.float),
       np.array(p1,dtype=np.float),
    ))
  def __getitem__(s,i):
    return s.p[i]
  def firstpoint(s):
    return s.p[0]
  def secondpoint(s):
    return s.p[1]
  def __str__(s):
    return 'Wall_segment('+str(list(s.p))+')'

class room:
  ' A group of obstacle_segments and the time for the run'
  def __init__(s,obst,Nob):
    s.obst=obst
    RoomP=obst[0]
    for j in range(1,len(obst)):
      RoomP=np.concatenate((RoomP,obst[j]),axis=0)
    s.points=RoomP
    s.Nob=Nob
    s.inside_points=np.array([])
    s.objectcorners=np.array([obst[0:Nob-1]])
    s.boundaryedge= np.array([obst[Nob:-1]])
    s.time=np.array([0.0,0.0])
  def __getobst__(s,i):
   ''' Returns the ith wall of s '''
   return s.obst[i]
  def __getinsidepoint__(s,i):
   ''' Returns the ith inside point of s '''
   return s.inside_points[i]
  def add_obst(s,obst0):
    ''' Adds wall to the walls in s, and the points of the wall to the
    points in s '''
    s.obst+=(obst0,)
    s.points+=obst0
    return
  def __str__(s):
    return 'Rooom('+str(list(s.walls))+')'
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
    for j in range(0,m): #product(range(0,m),range(0,3)): #FIXME get x co-odrinates
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
    for j in range(0,m): #product(range(0,m),range(0,3)):
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
    for j in range(0,m): #product(range(0,m),range(0,3)):
      p2=s.points[j][2]
      leng2=lf.length(np.array([p1,p2]))
      if leng2>leng:
        leng=leng2
    return leng
  def ray_bounce(s,Tx,Nre,Nra):
    #start_time=t.time()         # Start the time counter
    ''' Traces ray's uniformly emitted from an origin around a room.
    Number of rays is n, number of reflections m. Output
    Nra x Nre x Dimension array containing each ray as a squence of it's
    intersection points and the corresponding object number.'''
    r           =s.maxleng()
    raylist     =np.empty([Nra+1, Nre+1,4])
    theta       =((2.0*ma.pi)/Nra)*np.arange(Nra)
    TxMat       =np.array([Tx[0]*np.ones(Nra),Tx[1]*np.ones(Nra),Tx[2]*np.ones(Nra)])
    directions  =np.transpose(r*np.vstack((np.cos(theta),np.sin(theta),np.zeros(Nra),np.zeros(Nra))))#FIXME rotate in z axis too.
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      raystart.multiref(s,Nre)
      raylist[it]=raystart.points[0:-2]
    #print(raylist)
    return raylist
  def Plotroom(s,origin,width):
    ''' Plots all the edges in the room '''
    mp.plot((origin),marker='x',c='r')
    for obst0 in s.obst:
      hp.Plotedge(obst0,'g',width)
    return
  def roomconstruct(s,obsts):
    ''' Takes in a set of wall segments and constructs a room object
    containing them all'''
    for obst1 in obsts[1:]:
      s.add_obst(obst1)
    return
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
  def add_inside_objects(s,corners):
    if s.objectcorners.shape[0]==0:
      n=0
      j=0
    else:
      n=s.objectcorners[-1][0]
      j=n
    for x in corners:
      if j==0: s.inside_points=x.firstpoint()
      else: s.inside_points=np.vstack((s.inside_points,x.firstpoint()))
      j+=1
    if n==0: s.objectcorners=np.array([(n,j-1)])
    else:
      s.objectcorners=np.vstack((s.objectcorners,np.array([(n,j-1)])))
    return
  def roommesh(s,spacing):
    Nx=int((s.xbounds[1]-s.xbounds[0])/spacing)
    Ny=int((s.ybounds[1]-s.ybounds[0])/spacing)
    Nz=int((s.zbounds[1]-s.zbounds[0])/spacing)
    return rmes.roommesh(s.inside_points,s.objectcorners,(s.xbounds()),(s.ybounds()),spacing)
  # def room_collision_point_with_end(s,line,space):
    # ''' The closest intersection out of the possible intersections with
    # the wall_segments in room for a line with an end point.
    # Returns the intersection point if intersections occurs'''
    # # Retreive the Maximum length from the Room
    # # Find whether there is an intersection with any of the walls.
    # cp, mu=ins.intersection_with_end(line,s.obst[0],space)
    # if cp==1: count=1
    # else: count=0
    # for OB in s.obst[1:]:
      # cp, mu=ins.intersection_with_end(line,OB,space)
      # if cp==1:
        # count+=1
    # if count % 2 ==0:
      # return 0
    # elif count % 2 ==1:
      # return 1
    # else: return 2 # This term shouldn't happen

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
import sys

epsilon=sys.float_info.epsilon

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
  def __init__(s,obst):
    s.obst=obst
    RoomP=obst[0]
    for j in range(1,len(obst)):
      RoomP=np.concatenate((RoomP,obst[j]),axis=0)
    s.points=RoomP
    # Points is the array of all the co-ordinates which form the surfaces in the room
    s.Nob=len(obst)
    # Nob is the number of surfaces forming obstacles in the room.
    s.maxlength=np.zeros((4,1))
    # Co-ordinate bounds
    s.bounds=np.array([np.min(s.points,axis=0),np.max(s.points,axis=0)])
    # The initial maxleng, maxxlength, maxylength and maxzlength are 0, this value is changed once computed
    s.inside_points=np.array([])
    # The inside points line within obstacles and are used to detect if a ray is inside or outside.
    s.time=np.array([t.time()])
    # The time taken for a computation is stored in time.
    s.meshwidth=0.0
  def __get_obst__(s,i):
   ''' Returns the ith surface obstacle of the room s'''
   return s.obst[i]
  def __get_insidepoint__(s,i):
   ''' Returns the ith inside point of s '''
   return s.inside_points[i]
  def get_meshwidth(s,Mesh):
    if abs(s.meshwidth)<epsilon:
      return s.maxlength[1]/Mesh.nx
    else:
      return s.meshwidth
  def __set_obst__(s,obst0):
    ''' Adds wall to the walls in s, and the points of the wall to the
    points in s '''
    s.obst+=(obst0,)
    s.points+=obst0
    return
  def __set_insidepoint__(s,p):
    # FIXME - Is this function needed??
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
          s.maxlength[0]=leng2
      return s.maxlength[0]
    else: return s.maxlength[0]
  def maxxleng(s):
    ''' Finds the maximum length contained in the room in the x plane'''
    if abs(s.maxlength[1])<epsilon:
      s.maxlength[1]=s.bounds[1][0]-s.bounds[0][0]
      return s.maxlength[1]
    else: return s.maxlength[1]
  def maxyleng(s):
    ''' Finds the maximum length contained in the room in the y plane'''
    if abs(s.maxlength[2])<epsilon:
      s.maxlength[2]=s.bounds[1][1]-s.bounds[0][1]
      return s.maxlength[2]
    else: return s.maxlength[2]
  def maxzleng(s):
    ''' Finds the maximum length contained in the  in the z plane '''
    if abs(s.maxlength[3])<epsilon:
      s.maxlength[3]=s.bounds[1][2]-s.bounds[0][2]
      return s.maxlength[3]
    else: return s.maxlength[3]
  def position(s,p,h):
    ''' Return the indexing position in a mesh with width h for point p
    lying in the room s. '''
    if isinstance(p[0],float): n=1
    elif isinstance(p[0],int): n=1
    else:
      n=len(p)
    if n==1:
      i,j,k=(p-s.bounds[0])/h
      return int(i),int(j),int(k)
    elif n>1:
      positions=np.array((p-np.tile(s.bounds[0],(n,1)))/h,dtype=int)
      return positions
    else:
      raise ValueError("Neither point nor array of points")
  def ray_mesh_bounce(s,Tx,Nre,Nra,directions,Mesh):
    ''' Traces ray's uniformly emitted from an origin around a room.
    Number of rays is Nra, number of reflections Nre.
    Output Mesh of sparse matrices containing the ray information'''
    start_time    =t.time()         # Start the time counter
    r             =s.maxleng()
    raylist       =np.empty([Nra+1, Nre+1,4])
    directions    =r*directions
    # Iterate through the rays find the ray reflections
    # FIXME the rays are independent of each toher so this is easily parallelisable
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      Mesh=raystart.mesh_multiref(s,Nre,Mesh,Nra,it)
      raylist[it]=raystart.points[0:-2]
    s.time=t.time()-start_time
    return Mesh
  def ray_bounce(s,Tx,Nre,Nra,directions):
    ''' Traces ray's uniformly emitted from an origin around a room.
    Number of rays is Nra, number of reflections Nre. Output
    Nra x Nre x Dimension array containing each ray as a squence of it's
    intersection points and the corresponding object number.'''
    start_time    =t.time()         # Start the time counter
    r             =s.maxleng()
    directions    =r*directions
    raylist       =np.empty([Nra+1, Nre+1,4])
    # FIXME the rays are independent of each toher so this is easily parallelisable
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      raystart.multiref(s,Nre)
      raylist[it]=raystart.points[0:-2]
    s.time=t.time()-start_time
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

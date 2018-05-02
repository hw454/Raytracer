#!/usr/bin/env python3
# Keith Briggs 2017-02-02
# Hayley Wragg 2017-03-28
# Hayley Wragg 2017-04-12
# Hayley Wragg 2017-05-15
# Hayley Wragg 2017-07-10
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

class Wall_segment:
  ' a line segment from p0 to p1 '
  def __init__(s,p0,p1):
    assert not (p0==p1).all()
    s.p=np.vstack(
      (np.array(p0,dtype=np.float),
       np.array(p1,dtype=np.float),
    ))
  def __getitem__(s,i):
    return s.p[i]
  def __str__(s):
    return 'Wall_segment('+str(list(s.p))+')'

class room:
  ' A group of wall_segment'
  def __init__(s,wall0):
    s.walls=list((wall0,))
    s.points=list(wall0)
    s.time=np.array([0.0,0.0])
  def __getwall__(s,i):
   ''' Returns the ith wall of s '''
   return s.walls[i]
  def add_wall(s,wall):
    ''' Adds wall to the walls in s, and the points of the wall to the
    points in s '''
    s.walls+=(wall,)
    s.points+=wall
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
  def Plotroom(s,origin):
    ''' Plots all the edges in the room '''
    mp.plot(origin[0],origin[1],marker='x',c='r')
    for wall in s.walls:
      hp.Plotedge(np.array(wall.p),'g')
    return
  def roomconstruct(s,walls):
    ''' Takes in a set of wall segments and constructs a room object
    containing them all'''
    for wall in walls[1:]:
      s.add_wall(wall)
    return
  def xbounds(s):
    xarray=np.vstack(np.array([s.walls[0][0][0]]))
    for wall in s.walls:
      xarray=np.vstack((xarray,wall[0][0],wall[1][0]))
    return np.array([min(xarray)[0],max(xarray)[0]])
  def ybounds(s):
    yarray=np.vstack(np.array([s.walls[0][0][1]]))
    for wall in s.walls:
      yarray=np.vstack((yarray,wall[0][1],wall[1][1]))
    return np.array([min(yarray)[0],max(yarray)[0]])
  def roommesh(s,spacing):
    return rmes.roommesh((s.xbounds()),(s.ybounds()),spacing)
  def uniform_ray_tracer(s,origin,n,i,spacing,frequency,streg,m,refloss):
    start_time=t.time()
    ''' Traces ray's uniforming emitted from an origin around a room.
    Number of rays is n, number of reflections m'''
    pi=4*np.arctan(1) # numerically calculate pi
    Mesh=s.roommesh(spacing)
    losscoef=((4.0*ma.pi*frequency)/(2.99792458*1.0E+8))**2
    start=streg/losscoef
    for j in range(0,n+1):
      theta=(2*j*pi)/n
      r=s.maxleng()
      xtil=ma.cos(theta)
      ytil=ma.sin(theta)
      x= r*xtil+origin[0]
      y= r*ytil+origin[1]
      ray=Ray(frequency,start,origin,(x,y))
      # Reflect the ray
      ray.multiref(s,m)
      mp.figure(i)
      ray.Plotray(s)
      mp.figure(i+1)
      Mesh=ray.heatmapray(Mesh,ray.streg,ray.frequency,spacing,refloss)
    end_time=(t.time() - start_time)
    s.time[0]=end_time
    print("Time to compute unbounded--- %s seconds ---" % end_time )
    mp.figure(i)
    mp.title('Ray paths')
    s.Plotroom(origin)
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/Rays'+str(i)+'.png',bbox_inches='tight')
    mp.figure(i+1)
    mp.title('Heatmap')
    s.Plotroom(origin)
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/Heatmap'+str(i)+'.png',bbox_inches='tight')
    Mesh.hist(i+2)
    mp.figure(i+2)
    mp.title('Cumulative Frequency of signal power')
    mp.grid()
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/Cumsum'+str(i)+'.png', bbox_inches='tight')
    mp.figure(i+3)
    mp.title('Histrogram of signal power')
    mp.grid()
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/HistogramNoBounds'+str(i)+'.png',bbox_inches='tight')
    mp.figure(i+4)
    s.Plotroom(origin)
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/Room.jpg',bbox_inches='tight')
    return i+4
  def uniform_ray_tracer_bounded(s,origin,n,i,spacing,frequency,streg,m,bounds,refloss):
    ''' Traces ray's uniforming emitted from an origin around a room.
    Number of rays is n, number of reflections m'''
    start_time=t.time()
    pi=4*np.arctan(1) # numerically calculate pi
    Mesh=s.roommesh(spacing)
    losscoef=((4.0*ma.pi*frequency)/(2.99792458*1.0E+8))**2
    start=streg/losscoef
    for j in range(0,n+1):
      theta=(2*j*pi)/n
      r=s.maxleng()
      xtil=ma.cos(theta)
      ytil=ma.sin(theta)
      x= r*xtil+origin[0]
      y= r*ytil+origin[1]
      ray=Ray(frequency,start,origin,(x,y))
      # Reflect the ray
      ray.multiref(s,m)
      #mp.figure(i)
      #ray.Plotray(s)
      mp.figure(i+1)
      Mesh=ray.heatmapraybounded(Mesh,ray.streg,ray.frequency,spacing,bounds,refloss)
    end_time=(t.time() - start_time)
    s.time[1]=end_time
    print("Time to compute bounded--- %s seconds ---" % end_time)
    #mp.figure(i)
    #mp.title('Ray paths')
    #mp.savefig('../../ImagesOfSignalStrength/FiguresNew/Rays.jpg',bbox_inches='tight')
    mp.figure(i+3)
    s.Plotroom(origin)
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/temproom'+str(i)+'.png',bbox_inches='tight')#This section is the same as the unbounded
    #s.Plotroom(origin)
    mp.figure(i+1)
    #mp.title('Heatmap Bounded')
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/HeatmapBounded'+str(i)+'.png',bbox_inches='tight')
    s.Plotroom(origin)
    Mesh.histbounded(i+2)
    mp.figure(i+2)
    mp.title('Cumulative Frequency of signal power')
    mp.grid()
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/CumsumBounded'+str(i)+'.png',bbox_inches='tight')
    mp.figure(i+3)
    mp.title('Histogram of bounded signal power')
    mp.grid()
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/HistogramBounds'+str(i)+'.png',bbox_inches='tight')
    return i+4

class Ray:
  ''' represents a ray by a collection of line segments followed by
      an origin and direction.  A Ray will grow when reflected replacing
       the previous direction by the recently found intersection and
       inserting the new direction onto the end of the ray.
  '''
  def __init__(s,freq,start,origin,direction):
    s.ray=np.vstack(
      (np.array(origin,   dtype=np.float),
       np.array(direction,dtype=np.float),
    ))
    s.frequency=freq #, dtype=np.float
    s.streg=start    #, dtype=np.float
  def __str__(s):
    return 'Ray(\n'+str(s.ray)+')'
  def _get_origin(s):
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
  def heatmapray(s,Mesh,streg,freq,spacing,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      #refloss=2
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    Mesh.plot()
    return Mesh
  def heatmapraybounded(s,Mesh,streg,freq,spacing,bounds,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    Mesh.bound(bounds)
    Mesh.plot()
    return Mesh





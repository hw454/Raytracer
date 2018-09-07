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
  def __getitem__(s,i):
   return s.walls[i]
  def add_wall(s,wall):
    s.walls+=(wall,)
    s.points+=wall
  def __str__(s):
    return 'Rooom('+str(list(s.walls))+')'
  def maxleng(s):
    leng=0
    p1=s.points[-1]
    for p2 in s.points:
      leng2=lf.length(np.array([p1,p2]))
      if (leng2>leng) :
        leng=leng2
    return leng



class Ray:
  ''' represents a ray by a collection of line segments followed by
      an origin and direction.  A Ray will grow when reflected, but will
      always have an even number of rows.
  '''
  def __init__(s,origin,direction):
    s.ray=np.vstack(
      (np.array(origin,   dtype=np.float),
       np.array(direction,dtype=np.float),
    ))
  def __str__(s):
    return 'Ray(\n'+str(s.ray)+')'
  def get_origin(s):
    # The second to last term in the np array is the starting
    #co-ordinate of the travelling ray
    return s.ray[-2]
  def get_direction(s):
    # The direction of the travelling ray is the last term in the np
    #array.
    return s.ray[-1]
  def get_travellingray(s):
    # The ray which is currently travelling. Should return the recent
    #origin and direction.
    return [s.get_origin(), s.get_direction()]
  def wall_collision_point(s,wall_segment):
    # intersection of the ray with a wall_segment
    return ins.intersection(s.get_travellingray(),wall_segment)
  def room_collision_point(s,room):
    # The closest intersection out of the possible intersections.
    leng=room.maxleng()
    rwall=room.walls[0]
    rcp=s.wall_collision_point(rwall)
    for wall in room.walls[1:]:
      cp=s.wall_collision_point(wall)
      if cp[0] is not None:
        leng2=s.ray_length(cp)
        if (leng2<leng) :
          leng=leng2
          rcp=cp
          rwall=wall
    return rcp, rwall
  def ray_length(s,inter):
    # The length of the ray upto the intersection
    o=s.get_origin()
    ray=np.array([o,inter])
    return lf.length(ray)
  def reflect(s,room):
    cp,wall=s.room_collision_point(room)
    # Check that a collision does occur
    if cp is None: return None
    else:
      # Construct the incoming array
      origin=s.get_origin()
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray,n=ref.try_reflect_ray(ray,wall.p)
      # update self...
      s.ray[-1]=cp
      s.ray=np.vstack((s.ray,lf.Direction(refray)))
    return



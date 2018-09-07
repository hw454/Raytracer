#!/usr/bin/env python3
# Keith Briggs 2017-02-02

from math import atan2,hypot,sqrt,copysign
import numpy as np
import matplotlib as mp

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
    return s.ray[-2]
  def get_direction(s):
    return s.ray[-1]
  def collision_point(s,wall_segment):
    # Ray-line segment intersection test in 2D
    # http://bit.ly/1CoxdrG
    point0=wall_segment[0]
    point1=wall_segment[1]
    origin=s.get_origin()
    v1=origin-point0
    v2=point1-point0
    direction=s.get_direction()
    direction/=np.linalg.norm(direction)
    v3=np.array([-direction[1],direction[0],])
    v2dotv3=np.dot(v2,v3)
    t1=np.cross(v2,v1)/v2dotv3
    if t1>=0.0:
      t2=np.dot(v1,v3)/v2dotv3
      if 0.0<=t2<=1.0:
        return origin+t1*direction
    return None
  def reflect(s,wall_segment):
    cp=s.collision_point(wall_segment)
    if cp is None: return None
    print('wall_segment=',wall_segment)
    print('cp=',cp)
    origin=s.get_origin()
    dx=cp[0]-origin[0]
    dy=cp[1]-origin[1]
    angle=atan2(dy,dx) # FIXME
    reflected_ray=np.array([cp,(angle,angle)]) # FIXME
    print('angle=',angle)
    # update self...
    s.ray[-1]=cp
    s.ray=np.vstack((s.ray,reflected_ray))

def go():
  ray=Ray((0,0),(1,1))
  print('ray=',ray)
  point0=np.array([0.5,0.0])
  point1=np.array([0.5,1.0])
  ws=Wall_segment(point0,point1)
  print('wall_segment=',ws)
  print('collision_point=',ray.collision_point(ws))
  ray.reflect(ws)
  print('ray after reflection=',ray)
  ray.reflect(ws)
  print('ray after another reflection=',ray)
  point0=np.array([-1.5,0.0])
  point1=np.array([-1.5,1.0])
  ws=Wall_segment(point0,point1)
  #print('wall_segment=',ws)
  #print('collision_point=',ray.collision_point(ws))

def try_reflect_ray(sp,cp,p0):
  # solve((1+c1^2)*q=b0^2+c1^2*b1^2+2*b0*c0*c1*b1,c1);
  # with c0 +or- 1
  # ray "a" goes from sp to cp
  # hits wall segment b=(p0,cp) at cp
  # output is sp,ep of reflected ray
  a=(cp[0]-sp[0],cp[1]-sp[1])
  b=(cp[0]-p0[0],cp[1]-p0[1])
  dot=a[0]*b[0]+a[1]*b[1]
  cross=a[0]*b[1]-a[1]*b[0]
  print('dot=',dot,'cross=',cross)
  q=(dot/hypot(a[0],a[1]))**2
  c0=copysign(1.0,-dot*cross) # FIXME is this right?
  disc=-q**2+(b[0]**2+b[1]**2)*q+(b[0]*b[1])**2*(c0**2-1.0)
  c1=(-sqrt(disc)+b[0]*b[1]*c0)/(q-b[1]**2) # FIXME sign of sqrt?
  return np.array([(cp[0],cp[1]),(cp[0]+c0,cp[1]+c1)])

if __name__=='__main__':
  np.set_printoptions(precision=2,threshold=1e-12,suppress=True)
  sp=np.array([0.0,0.0])
  cp=np.array([1.0,1.0])
  p0=np.array([0.0,1.0])
  mp.pl0t(try_reflect_ray(sp,cp,p0))
  p0=np.array([2.0,1.0])
  print(try_reflect_ray(sp,cp,p0))
  p0=np.array([1.0,0.0])
  print(try_reflect_ray(sp,cp,p0))
  p0=np.array([1.0,2.0])
  print(try_reflect_ray(sp,cp,p0))
  p0=np.array([2.0,0.0])
  print(try_reflect_ray(sp,cp,p0))
  p0=np.array([0.0,3.0])
  print(try_reflect_ray(sp,cp,p0))
  p0=np.array([2.0,3.0])
  print(try_reflect_ray(sp,cp,p0))
  exit()
  go()

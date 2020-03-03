#!/usr/bin/env python3
# Keith Briggs 2017-02-02
# Hayley Wragg 2017-03-28

from math import atan2,hypot,sqrt,copysign
import numpy as np
import matplotlib.pyplot as mp

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
  def ray_length(s,inter):
	  o=get_origin(s)
	  return ((inter[0]-o[0])**2+(inter[1]-o[1])**2)**0.5
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

def Gradient(line):
  '''  Gradient using co-ordinates input is a line (pair of co-ordinates) '''
  if (line[0][0]-line[1][0]==0):
    return
  else:
    m=(line[0][1]-line[1][1])/(line[0][0]-line[1][0])
  return m
  
def try_reflect_ray(o,inter,edge0):
  ''' Reflection of a ray which goes through the points o and inter and intersects an edge at the point edge '''
  ox=np.linalg.norm(np.array([(o[0],inter[1]),(inter[0],inter[1])]))
  oy=np.linalg.norm(np.array([(inter[0],o[1]),(inter[0],inter[1])]))
  Dx=o[0]-ox
  if m:
    if m > 0:
      Dy=o[1]+oy
    elif m < 0:
      Dy=o[1]-oy
    else:                                  # m=0 i.e the line is of the form y=
      if (o[0]-inter[0])>0:                # line is going right
        Dy=o[1]-oy
      elif (o[0]-inter[0])<0:              # line is going left
        Dy=o[1]+oy
  else:                                    # m has not be returned. i.e The line if of the form x=
    if (inter[1]-o[1])>0:                  # line is going up
      Dy=o[1]-oy
    elif (inter[1]-o[1])<0:
      Dy=o[1]+oy                # line is going down
    else:
      Dy=0
  Image=(5*inter[0]-4*o[0], 5*inter[1]-4*o[1])
  o=(o[0]-Dx,o[1]-Dy)
  inter=(inter[0]-Dx, inter[1]-Dy)
  edge0=(edge0[0]-Dx,edge0[1]-Dy)
  edge=(edge0,inter)
  if line_length(edge0,inter)==0 :
    unitedge=(0,0)
  else: 
    unitedge=((inter[0]-edge[0])/np.linalg.norm(edge), (inter[1]-edge[1] )/np.linalg.norm(edge))
    normedge=(-unitedge[1],unitedge[0])
  k=2*dot(normedge,Image)
  RefPt=(Image[0]-k*normedge[0], image[1]-k*normedge[1])
  mbar=Gradient(inter, ReflPt)
  #Translate Back
  inter=(inter[0]+Dx, inter[1]+Dy)
  RefPt=(ReflPt[0]+Dx, ReflPt[1]+Dy)
  if mbar:
    ReflPt=(RefPt[0]+310,ReflPt[1]+310*mbar)
  else:
     #mbar is None #means that the line goes straight up
     ReflPt=(ReflPt[0],ReflPt[1]+310)
  #FIXME this has not been finished and should be moved inside the class structures
  return np.array([inter, ReflPt])

def intersection(line,edge):
  ''' Locates the point inter as the point of intersection between a line  and an edge '''
  # line p0=(x0,y0), p1=(x1,y1), y=mx+c m=Gradient(p0,p1), c=y0-m*x0
  # Calculate the gradients
  m1=Gradient(line)
  m2=Gradient(edge)
  inter=np.array([0.0,0.0])
  # Determine if either line is constant in x
  if (m1):
    c1=line[0][1]-m1*line[0][0]
    if (m2):
      c2=edge[0][1]-m2*edge[0][0]
      if(abs(m2-m1)>0):
        # Gradients are different and infinite lines will intersect at some point
        print(c1,c2,m1,m2) 
        print(edge)
        inter[0]=(c1-c2)/(m2-m1)  
        inter[1]=m1*inter[0]+c1
      else:
		# lines are parallel
        return
    else:
      # edge is constant in x
      inter[0]=edge[0][0]
      inter[1]=inter[0]*m1+c1
  else :
	 # line is constant
    if (m2):
      inter[0]=line[0][0]
      inter[1]=inter[0]*m2+c2
    else:
      # line and edge are constant in x
      return

  return (inter)
  
def plot(line,c):
  mp.plot((line[0][0],line[1][0]),(line[0][1],line[1][1]),c)
  return

if __name__=='__main__':
  np.set_printoptions(precision=2,threshold=1e-12,suppress=True)
  sp=np.array([0.0,0.0])
  cp=np.array([1.0,1.0])
  l1=np.array([sp,cp])
  p0=np.array([0.0,2.0])
  p1=np.array([2.0,0.0])
  e1=np.array([p0,p1])
  plot(l1,'r')
  plot(e1, 'b')
  print(intersection(l1,e1))
  mp.show()
  exit()
 

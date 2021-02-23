
#!/usr/bin/env python3
# Hayley Wragg 2017-05-19
''' Module Containing the Intersection and it's test. '''

import numpy as np
from numpy import linalg as la
import linefunctions as lf
import logging
import sys
import itertools
import math as ma

epsilon=sys.float_info.epsilon
dbg=0
if dbg:
  logon=1
else:
  logon=np.load('Parameters/logon.npy')

def Project2D(po,triangle):
  T0=triangle[0]
  T1=triangle[1]
  T2=triangle[2]
  # Output po as a 2D co-ordinate in the triangle plane with T0 as the origin
  direc1=(T1-T0)/(la.norm(T1-T0))
  norm=np.cross(T1-T0, T2-T0)
  norm=norm/(la.norm(norm))
  direc2=np.cross(direc1,norm)
  # Check the vectors direc1, norm, and direc2 satisfy the right conditions.
  if abs(np.dot(norm,direc1))<epsilon and abs(np.dot(norm,direc2))<epsilon: # and abs(np.dot(direc1,direc2))<epsilon:
    coef=np.array([np.dot(po,direc1),np.dot(po,direc2),1])
    return coef
  else:
    print('n.d1',abs(np.dot(norm,direc1)),'n.d2',abs(np.dot(norm,direc2)))
    raise CalculateError("The created vectors, norm, direc1, and direc2 aren't all normal")
    return None
#FIXME write a test function for inside check
# test inside and outside and on edge
#FIXME check for close to edge
#FIXME test for iterating through two triangles
def TriangleFalseCheck(triangle):
  check=0
  for j in range(0,2):
    c=triangle[j]
    x=triangle[j+1]
    if c[0]==x[0] and c[1]==x[1] and c[2]==x[2]:
      check=1
      return check
  return check

def InsideCheck(Point,triangle):
  # Since all points are in the same plane, translate them all into 2D
  # Project2D outputs (x,y,1)
  Point=Project2D(Point,       triangle)
  T0   =Project2D(triangle[0], triangle)
  T1   =Project2D(triangle[1], triangle)
  T2   =Project2D(triangle[2], triangle)
  # Use barycentric co-ordinates to check if inter is within the triangle
  TriMat=np.array([T0,T1,T2]).T
  coefs=np.linalg.solve(TriMat,Point)
  # If all the coefficients are bigger than 0 and less than 1 then the point is inside the triangle
  return abs(np.sum(coefs))-1<=epsilon and all(c>=-epsilon for c in coefs)

def intersection(line,room,nob):
    ''' find the intersection of a line and a plane. The line is
    represented as a point and a direction and the plane is three points which lie on the plane.'''
    triangle=room.obst[nob-1]
    tricheck=TriangleFalseCheck(triangle)
    if logon:
      logging.info('Ray direction (%f,%f,%f)'%(line[1][0],line[1][1],line[1][2]))
    if tricheck:
      print("Stopped on the surface intersection")
      print("Triangle:", triangle)
      raise Error("The surfaces is not defined properly")
      return None
    elif not tricheck:
        # The triangle lies on a plane with normal norm and goes through the point p0.
        # The line goes through the point l0 in the direction direc
        norm=room.norms[nob-1]
        direc   =0.1*line[1]
        p0      =room.obst[nob-1][0]
        l0      =line[0]
        dot=direc@norm
        if  abs(dot)<=epsilon:
          if logon:
            logging.info('Ray is parallel to surface')
            logging.info('Surface norm (%f,%f,%f)'%(norm[0],norm[1],norm[2]))
            logging.info('dot %f'%dot)
            logging.info('nob %d'%nob)
          return np.array([ma.nan,ma.nan,ma.nan])
        elif abs(dot)>epsilon:
          # The line is not parallel with the plane and there is therefore an intersection.
          lam=((p0-l0)@norm)/dot
          if abs((p0-l0)@norm)<epsilon:
            p0=room.obst[nob-1][1]
            lam=((p0-l0)@norm)/dot
            if abs((p0-l0)@norm)<epsilon:
              p0=room.obst[nob-1][2]
              lam=((p0-l0)@norm)/dot
          if lam<epsilon:
            # The intersection point is in the opposite direction to the ray
            # print('negative direction',lam)
            # Not just negative considered as there may be an epsilon difference to the point itself
            if logon:
              logging.info('lam %f'%lam)
              logging.info('nob %d'%nob)
              logging.info('Triangle point (%f,%f,%f)'%(p0[0],p0[1],p0[2]))
              logging.info('Ray start (%f,%f,%f)'%(l0[0],l0[1],l0[2]))
              logging.info('Triangle norm (%f,%f,%f)'%(norm[0],norm[1],norm[2]))
              logging.info('direc dot norm %f'%dot)
              logging.info('Top frac (%f,%f,%f)'%((p0-l0)[0],(p0-l0)[1],(p0-l0)[2]))
              logging.info('Triangle (%f,%f,%f),(%f,%f,%f),(%f,%f,%f)'%(triangle[0][0],triangle[0][1],triangle[0][2],triangle[1][0],triangle[1][1],triangle[1][2],triangle[2][0],triangle[2][1],triangle[2][2]))
            return np.array([ma.nan,ma.nan,ma.nan])
          else:
            # Compute the intersection point with the plane
            inter=l0+lam*direc
            if InsideCheck(inter,triangle):
              # The point is inside the triangle
              if logon:
                logging.info('nob %d'%nob)
                logging.info('Intersection found (%f,%f,%f)'%(inter[0],inter[1],inter[2]))
              return inter
            else:
              # The point is outside the triangle
              # print('outside surface')
              if logon:
                logging.info('Point outside ob (%f,%f,%f)'%(inter[0],inter[1],inter[2]))
                logging.info('nob %d'%nob)
              return np.array([ma.nan,ma.nan,ma.nan])
        #elif (parcheck<=epsilon and parcheck>=-epsilon):
          #FIXME deal with diffraction here.
          # print('parallel to surface', parcheck)
          # The line is contained in the plane or parallel right output
        #  return np.array([ma.nan,ma.nan,ma.nan])
        else:
          print('Before error, direction ',direc,' Normal ',norm,' Parallel check ',parcheck)
          logging.error('neither intersect or parallel to plane')
          return np.array([ma.nan,ma.nan,ma.nan])
    else:
      print("Triangle: ", triangle)
      raise Error("Triangle neither exists or doesn't exist")

def test():
    l1=np.array([[0.0,0.0,0.0],[1.0,1.0,1.0]])
    triangle=np.array([[0.5,3.0,3.0],[0.5,0.0,3.0],[0.5,0.0,0.0]])
    inter=intersection(l1,triangle)
    print(inter)
    return

def test2():
    l1=np.array([[0.0,0.0,0.0],[0.0,0.0,1.0]])
    triangle=np.array([[0.5,3.0,1.0],[0.5,0.0,1.0],[0.5,0.0,1.0]])
    inter=intersection(l1,triangle)
    return

def test3():
    l1=np.array([[0.0,0.0,0.0],[0.0,0.0,1.0]])
    triangle=np.array([[0.5,3.0,1.0],[0.5,0.0,1.0],[0.0,0.3,1.0]])
    inter=intersection(l1,triangle)
    return

def test4():
    ''' Test that an intersection is found with one ray in a closed domain'''
    # Start of the ray
    origin=np.array([ 0.0    , 0.0    , 0.0    ])
    # Direction of the ray
    direct=np.array([ 1.0    , 1.0    , 1.0    ])
    # Points defining the triangles which enclose the domain
    point0=np.array([ 1.0/3.0, 0.0    ,-1.0/3.0])
    point1=np.array([ 0.0    , 1.0/3.0,-1.0/3.0])
    point2=np.array([ 0.0    , 0.0    , 1.0/3.0])
    point3=np.array([-1.0/3.0,-1.0/3.0,-1.0/3.0])
    # Construct the triangles
    triangle1=np.array([point0,point1,point3])
    triangle2=np.array([point0,point1,point2])
    triangle3=np.array([point1,point2,point3])
    triangle4=np.array([point0,point2,point3])
    Triangles=np.array([triangle1,triangle2,triangle3,triangle4])
    # Construct the rays
    l1=np.array([origin,direct])
    # Find the intersections
    count =0
    for T in Triangles:
      inter=intersection(l1,T)
      if inter[0] is not None:
        count+=1
    if count==1:
      return 1
    elif count==0:
      print("No intersection found in closed domain")
      return 0
    elif count>1:
      print("More than one intersection found for one ray")
      return 0
    else:
      print("Should not be negative intersections")
      return 0

def testinputray(ray):
    ''' Test that an intersection is found with one ray in a closed domain'''
    origin=ray[0]
    direct=ray[1]
    # Points defining the triangles which enclose the domain
    point0=np.array([ 1.0/3.0, 0.0    ,-1.0/3.0])
    point1=np.array([ 0.0    , 1.0/3.0,-1.0/3.0])
    point2=np.array([ 0.0    , 0.0    , 1.0/3.0])
    point3=np.array([-1.0/3.0,-1.0/3.0,-1.0/3.0])
    # Construct the triangles
    triangle1=np.array([point0,point1,point3])
    triangle2=np.array([point0,point1,point2])
    triangle3=np.array([point1,point2,point3])
    triangle4=np.array([point0,point2,point3])
    Triangles=np.array([triangle1,triangle2,triangle3,triangle4])
    # Construct the rays
    l1=np.array([origin,direct])
    # Find the intersections
    count =0
    for T in Triangles:
      inter=intersection(l1,T)
      if inter[0] is not None:
        interreturn=inter
        count+=1
    if count==1:
      return interreturn
    elif count==0:
      print("No intersection found in closed domain")
      return inter
    elif count>1:
      print("More than one intersection found for one ray", inter)
      return interreturn
    else:
      print("Should not be negative intersections")
      return inter

def testmanyrays():
    # Start of the ray
    #origin=np.array([ 1.0/6.0 , 1.0/6.0, 1.0/6.0])
    origin=np.array([0.0,0.0,0.0])
    # Direction of the ray
    Nra           =20
    r             =3.0
    deltheta      =(-2+np.sqrt(2.0*(Nra)))*(ma.pi/(Nra-2))
    xysteps       =int(2.0*ma.pi/deltheta)
    zsteps        =int(ma.pi/deltheta-2)
    Nra           =xysteps*zsteps+2
    theta1        =deltheta*np.arange(xysteps)
    theta2        =deltheta*np.arange(1,zsteps+1)
    xydirecs      =np.transpose(r*np.vstack((np.cos(theta1),np.sin(theta1))))
    z             =r*np.tensordot(np.cos(theta2),np.ones(xysteps),axes=0)
    directions    =np.zeros((Nra,3))
    directions[0] =np.array([0.0,0.0,r])
    directions[-1]=np.array([0.0,0.0,-r])
    for j in range(1,zsteps+1):
      st=(j-1)*xysteps+1
      ed=(j)*xysteps+1
      sinalpha=np.sin(theta2[j-1])
      coords=np.c_[sinalpha*xydirecs,z[j-1]]
      directions[st:ed]=np.c_[coords]
    count=0
    for j in range(0,zsteps*xysteps):
      l1=np.array([origin,directions[j]])
      inter=testinputray(l1)
      if inter[0] is not None:
        count+=1
    if count==zsteps*xysteps:
      return 1
    elif count<zsteps*xysteps:
      print("Not enough intersections found in closed domain")
      print("count: ",count)
      return 0
    elif count>steps*xysteps:
      print("Too many intersections found")
      print("count: ", count)
      return 0
    else:
      print("Should not be zero or negative intersections")
      print("count: ", count)
      return 0


if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  result=testmanyrays()
  print(result)

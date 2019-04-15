
#!/usr/bin/env python3
# Hayley Wragg 2017-05-19
''' Module Containing the Intersection and it's test. '''

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as mp
import linefunctions as lf
import HayleysPlotting as hp
import sys
import itertools
import math as ma

epsilon=2.22e-16

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
    coef=np.array([np.dot(po-T0,direc1),np.dot(po-T0,direc2),1])
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
  if (np.sum(coefs)<=(1+epsilon)) and (np.sum(coefs)>=(1-epsilon)) and all(c>=-epsilon for c in coefs):
      return 1
  else:
      return 0
#FIXME archive the 2D function which is no longer needed.
def intersection2D(line,edge):
  ''' Locates the point inter as the point of intersection between
  a line  and an edge. line is an origin and direction, edge is 2
  points '''
  # line p0=(x0,y0), d1, p0=mu1 d1+c1
  # edge p2=(x2, y2), p3=(x3,y3), p2=lam1d2+c2, p3=lam3d2+c2
  # Intersection at mu3 d1+ c1= lam2 d2 +c1
  ## Calculate the directions
  d1=line[1]
  d2=lf.Direction(edge)
  ## Find the c's such that each line is of the form d*lam+c
  c1=line[0]                   # p=mu*d1+c1
  c2=edge[0]-edge[0][1]*d2     # p1=lamd2+c2
  if(abs(np.linalg.det(np.column_stack((line[1],d2))))>0):
    # Directions are different and infinite lines will intersect at some
    # point
    # NEXT3D to expand to 3D the edge will be a plane and the line
    #NEXT3Dshould still intersect at some point.
    # NEXT3D the line should be written as v=d(np.array([x,m1x, m2x]))
    #NEXT3D+np.array([0,c0,c1]))
    # NEXT3D and the plane (P-P0).n=0, then d=((P0-l0).n)/l.n, sub into
    #v to get the intersection
    cbar=c2-c1
    lam2=np.linalg.det(np.column_stack((cbar,d1)))/np.linalg.det(np.column_stack((d1,d2)))
    inter=lam2*d2+c2
    lam3=lf.dot((edge[1]-c2),d2)/lf.dot(d2,d2)
    lam1=lf.dot((edge[0]-c2),d2)/lf.dot(d2,d2)
    if (abs(lam2-lam1)>abs(lam3-lam1) or abs(lam3-lam2)>abs(lam3-lam1)) :
    # the intersection point does not lie withint the boundaries of the
    #edges.
     # print('outside bounds', lam1, lam2, lam3)
      return [None, None]
    else:
      # Check that the intersection point is in the direction of the ray.
      mu2=np.linalg.det(np.column_stack((d2,cbar)))/np.linalg.det(np.column_stack((d2,d1)))
      if (mu2<0):
          return [None,None]
      else:
        return (inter)
  else:
    # line and edge are parallel, there is no intersection or they lie
    #on each other
    return [None,None]

def intersection_with_end(line,edge,delta):
  ''' Locates whether there is an intersection between
  a line  and an edge. line is an origin and an end point, edge is 2
  points. Returns 1 for an intersection and 0 for none '''
  # line p0=(x0,y0), d1, p0=mu1 d1+c1
  # edge p2=(x2, y2), p3=(x3,y3), p2=lam1d2+c2, p3=lam3d2+c2
  # Intersection at mu3 d1+ c1= lam2 d2 +c1
  ## Calculate the directions
  d1=lf.Direction(line)
  d2=lf.Direction(edge)
  ## Find the c's such that each line is of the form d*lam+c
  c1=line[0]                 # p=mu*d1+c1
  c2=edge[0]     # p1=lamd2+c2
  if(abs(np.linalg.det(np.column_stack((d1,d2))))>0):
    # Directions are different and infinite lines will intersect at some
    # point
    # NEXT3D to expand to 3D the edge will be a plane and the line
    #NEXT3Dshould still intersect at some point.
    # NEXT3D the line should be written as v=d(np.array([x,m1x, m2x]))
    #NEXT3D+np.array([0,c0,c1]))
    # NEXT3D and the plane (P-P0).n=0, then d=((P0-l0).n)/l.n, sub into
    #v to get the intersection
    cbar=c2-c1
    #lam2=np.linalg.det(np.column_stack((cbar,d1)))/np.linalg.det(np.column_stack((d1,d2)))
    n1=np.array([d1[1],-d1[0]])
    if lf.dot(d2,n1)!=0: lam2=-lf.dot(cbar,n1)/np.dot(d2,n1)
    else: return [0, 0.0]
    inter=lam2*d2+c2
    lam3=lf.dot((edge[1]-c2),d2)/lf.dot(d2,d2)
    lam1=lf.dot((edge[0]-c2),d2)/lf.dot(d2,d2)
    mu1=lf.dot((line[0]-c1),d1)/lf.dot(d1,d1)
    mu2=lf.dot((inter-c1),d1)/lf.dot(d1,d1) # inter=mu2*d1+c1
    mu3=lf.dot((line[1]-c1),d1)/lf.dot(d1,d1) # line[1]=mu3*d1+c1
    space=0.5*delta/np.sqrt(lf.dot(d1,d1))
    if (abs(lam2-lam3)<abs(lam3-lam1)) and (abs(lam2-lam1)<abs(lam3-lam1)):
    # the intersection point does not lie within the boundaries of the
    #edges.
     # print('outside bounds', lam1, lam2, lam3)
      if (abs(mu2-mu3)<abs(mu3-mu1)) and (abs(mu2-mu1)<abs(mu3-mu1)):
        # the intersection point lies within the boundaries of the line.
        # Check the point isn't on the edge
        if abs(mu3-mu2)<0.5*space:
          return [0,mu2]
        return [1,mu2]
      else:
        return [0, mu2]
    else:
      return [0, mu2]
  else:
    # line and edge are parallel, there is no intersection or they lie
    #on each other
    return [0, 0.0]
#FIXME test intersection function.
# test negative and positive direction.
# Parallel, perpendicular and angled hit at the surface.
def intersection(line,triangle):
    ''' find the intersection of a line and a plane. The line is
    represented as a point and a direction and the plane is three points which lie on the plane.'''
    tricheck=TriangleFalseCheck(triangle)
    if tricheck:
      print("Stopped on the surface intersection")
      print("Triangle:", triangle)
      raise Error("The surfaces is not defined properly")
      return None
    elif not tricheck:
        edge1=triangle[1]-triangle[0]
        edge2=triangle[1]-triangle[2]
        # The triangle lies on a plane with normal norm and goes through the point p0.
        # The line goes through the point l0 in the direction direc
        norm    =np.cross(edge1,edge2)
        norm    =norm/(la.norm(norm))
        direc   =line[1]
        p0      =triangle[0]
        l0      =line[0]
        parcheck=np.inner(direc,norm)
        if (parcheck>epsilon or parcheck<-epsilon):
          # The line is not parallel with the plane and there is therefore an intersection.
          lam=np.inner(p0-l0,norm)/np.inner(direc,norm)
          if lam<-epsilon:
            # The intersection point is in the opposite direction to the ray
            # print('negative direction',lam)
            return [None, None,None]
          else:
            # Computer the intersection point with the plane
            inter=l0+lam*direc
            check=InsideCheck(inter,triangle)
            if check:
              # The point is inside the triangle
              return inter
            else:
              # The point is outside the triangle
              # print('outside surface')
              return [None,None,None]
        elif (parcheck<=epsilon and parcheck>=-epsilon):
          #FIXME deal with diffraction here.
          # print('parallel to surface', parcheck)
          # The line is contained in the plane or parallel right output
          return [None,None,None]
        else:
            print('Before error, direction ',direc,' Normal ',norm,' Parallel check ',parcheck)
            raise Error('neither intersect or parallel to plane')
        return None
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
        count+=1
    if count==1:
      return inter
    elif count==0:
      print("No intersection found in closed domain")
      return inter
    elif count>1:
      print("More than one intersection found for one ray")
      return inter
    else:
      print("Should not be negative intersections")
      return inter

def testmanyrays():
    # Start of the ray
    origin=np.array([ 1.0/6.0 , 1.0/6.0, 1.0/6.0])
    # Direction of the ray
    Nra           =20
    r             =1.0
    deltheta      =(np.sqrt(2.0*(Nra)-3)+1)*(ma.pi/(Nra-2))
    xysteps       =int(2.0*ma.pi/deltheta)
    zsteps        =int(ma.pi/deltheta+1)
    Nra           =xysteps*zsteps+2
    theta1        =deltheta*np.arange(xysteps)
    theta2        =deltheta*np.arange(zsteps)
    xydirecs      =np.transpose(r*np.vstack((np.cos(theta1),np.sin(theta1))))
    z             =r*np.tensordot(np.cos(theta2),np.ones(xysteps),axes=0)
    directions    =np.zeros((Nra,3))
    directions[0] =np.array([0.0,0.0,r])
    directions[-1]=np.array([0.0,0.0,-r])
    for j in range(1,zsteps):
      st=(j-1)*xysteps+1
      ed=(j)*xysteps+1
      sinalpha=np.sin(theta2[j])
      coords=np.c_[sinalpha*xydirecs,z[j]]
      directions[st:ed]=np.c_[coords]
    print(directions)
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

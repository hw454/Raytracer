
#!/usr/bin/env python3
# Hayley Wragg 2017-05-19
''' Module Containing the Intersection and it's test. '''

import numpy as np
import matplotlib.pyplot as mp
import linefunctions as lf
import HayleysPlotting as hp


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

def intersection(line,triangle):
    ''' find the intersection of a line and a plane. The line is
    represented as a point and a direction and the plane is three points which lie on the plane.'''
    print('Triangle',triangle)
    edges=np.matrix([triangle.T])
    print('edges',edges)
    norm=np.cross(edge1,edge2)
    direc=line[1]
    p0=triangle[0]
    l0=line[0]
    if np.inner(direc,norm)!=0:
      lam=np.inner(p0-l0,norm)/np.inner(direc,norm)
      #FIXME check if it's inside the triangle
      return l0+lam*direc
    elif np.inner(direc,norm)==0:
      # The line is contained in the plane or parallele right output
      return [None,None]
    else: raise Error('neither intersect or parallel to plane')
    return None

def test3D():
    l1=np.array([[0.0,0.0,0.0],[1.0,1.0,1.0]])
    plane=np.array([[0.0,1.0,1.0],[0.0,0.0,1.0]])
    inter=intersection3D(l1,plane)
    print('Intersection3D test should be 0', np.inner((inter-plane[0]),plane[1]))
    return


def test():
  ''' Test for the intersection function '''
  l1=np.array([[0.0,0.0],[1.0,1.0]])
  e1=np.array([[0.0,0.5],[3.0,0.5]])
  e2=np.array([[0.5,0.0],[0.5,3.0]])
  e3=np.array([[0.0,1.0],[2.0,-1.0]])
  inter=intersection(l1,e1)
  if inter[0]:
    if (abs(inter[0]-0.5)<1.0E-7 and abs(inter[1]-0.5)<1.0E-7):
     inter=intersection(l1,e2)
     if inter[0]:
       if (abs(inter[0]-0.5)<1.0E-7 and abs(inter[1]-0.5)<1.0E-7):
        inter=intersection(l1,e3)
        if inter[0]:
          if (abs(inter[0]-0.5)<1.0E-7 and abs(inter[1]-0.5)<1.0E-7):
            l1=np.array([[0.0,0.0],[1.0,2.0]])
            inter=intersection(l1,e1)
            if inter[0]:
              if (abs(inter[0]-0.25)<1.0E-7 and abs(inter[1]-0.5)<1.0E-7):
                inter=intersection(l1,e2)
                if inter[0]:
                  if (abs(inter[0]-0.5)<1.0E-7 and abs(inter[1]-1.0)<1.0E-7):
                    inter=intersection(l1,e3)
                    if inter[0]:
                      if (abs(inter[0]-3**-1)<1.0E-5 and abs(inter[1]-2*3.0**-1)<1.0E-5):
                       return True
  else:
    return False



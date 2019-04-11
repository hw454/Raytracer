
#!/usr/bin/env python3
# Hayley Wragg 2017-05-19
''' Module Containing the Intersection and it's test. '''

import numpy as np
import matplotlib.pyplot as mp
import linefunctions as lf
import HayleysPlotting as hp


def intersection(line,edge):
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

def intersection3D(line,plane):
    ''' find the intersection of a line and a plane. The line is
    represented as a origin and normal and the plane is represented as
    a direction as a normal. '''
    if np.inner(line[1],plane[1])!=0:
      lam=-np.inner(line[0]-plane[0],plane[1])/np.inner(line[1],plane[1])
      inter=line[0]+lam*line[1]
    else:
      # The line is contained in the plane
      return [None, None]
    return inter

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



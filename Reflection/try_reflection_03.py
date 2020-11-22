
#!/usr/bin/env python3

# Second Practice of using Shapely. Using the variable types line and polygon and plotting them.
import numpy as np
import math
from math import sin,cos,atan2,hypot
import matplotlib.pyplot as p
#from shapely.geometry.polygon import LinearRing, box, 
from shapely.geometry import Polygon, LineString, Point
from pprint import pprint
from numpy import *

def coords_to_xy_lists(coords):
  # [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)] -> ([0.0,0.5,1.0],[0.0,1.0,0.0])
  return ([x for x,y in coords],[y for x,y in coords])

def edge_cut_by_point_in_polygon(pt,poly):
  # compute which edge (if any) of poly contains pt
  for xy0,xy1 in zip(poly.exterior.coords[:-1],poly.exterior.coords[1:]):
    if LineString((xy0,xy1)).intersects(pt):
      return (xy0,xy1)

def ClosestPointOnLine(edge,a):
  #Project the point a onto the line edge and return the result as a point.
  p=Point(a)
  c=edge.coords[0]
  b=edge.coords[1]
  a = np.asarray(a)
  b = np.asarray(b)
  c = np.asarray(c)
  ba = b-a
  bc = b-c
  bc_unit=bc/((np.dot(bc,bc))**2)
  result = c+np.dot(ba,bc)/np.dot(bc,bc) * bc_unit
  return Point(result)
  
def reflection_in_edge(a,edge):
  # LineString([(x0,y0),(x1,y1),]) Polygon([(0, 0), (1, 1), (1, 0)]) -> LineString([(x0,y0),(x1,y1),(x2,y2)]) 
  x=(edge.coords[0][0],edge.coords[1][0]) 
  y=(edge.coords[0][1],edge.coords[1][1])
  closest=ClosestPointOnLine(edge,a.coords[0])
  inter=edge.intersection(a)
  proj=(inter.coords[0][0]-closest.coords[0][0],inter.coords[0][1]-closest.coords[0][1])
  #The vector from the projected point to the intersection point
  opp=(a.coords[0][0]-closest.coords[0][0],a.coords[0][1]-closest.coords[0][1])
  #The vector from the start of the line to the projected point on the edge
  ReflPt=(closest.coords[0][0]+2*proj[0]+opp[0],closest.coords[0][1]+2*proj[1]+opp[1])
  p.plot(ReflPt[0],ReflPt[1],'.')
  return LineString([inter,(ReflPt[0],ReflPt[1])])

  
def test():
  a=LineString([(-2,0.5),(7,0.5),]) 
  polygon = Polygon([(0, 0), (1, 1), (1, 0)]) # Creates a polygon with exterior points (0,0), (1,1), (1,0)
  print('The start of the line is',a.coords[0]) 
  inter=polygon.intersection(a) # Assigns x to the intercepts of the line a and the Polygon polygon
  print('The line intersects the polygon at ', inter)
  x,y=coords_to_xy_lists(inter.coords) # lists the x,y coords of the intersection
  c=Point(x[0],y[0]) # c is th initial interseption
  print('The line intersects the polygon at',c)
  edge=edge_cut_by_point_in_polygon(c,polygon) # edge gives the set of pairs which give the edge
  l=LineString([edge[0],edge[1]]) # l converts line into a LineString
  closest=ClosestPointOnLine(l,a.coords[0])
  print('The closest point on the polygon to the start point is ', closest)
  reflection=reflection_in_edge(a,l)
  print('Reflection is ', reflection)
  p.plot(closest.coords[0][0],closest.coords[0][1],'x')
  p.plot(a.coords[0][0],a.coords[0][1], '*')
  p.plot(*coords_to_xy_lists(reflection.coords))
  p.plot(*coords_to_xy_lists(a.coords))   
  p.plot(*coords_to_xy_lists(polygon.exterior.coords))
  p.plot(x,y) # Plots the intersection of the line and the polygon
  p.show()

if __name__=='__main__':
  test()


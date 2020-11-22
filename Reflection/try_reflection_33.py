
#!/usr/bin/env python3

# Second Practice of using Shapely. Using the variable types line and polygon and plotting them.
import numpy as np
import math
from math import sin,cos,atan2,hypot
import matplotlib.pyplot as p
#from shapely.geometry.polygon import LinearRing, box, 
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
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
  c=edge.coords[1]
  b=edge.coords[0]
  a = np.asarray(a)
  b = np.asarray(b)
  c = np.asarray(c)
  ba = b-a
  bc = b-c
  bc_unit=bc/((np.dot(bc,bc))**2)
  result = b-bc_unit*np.dot(ba,bc)/np.dot(bc,bc)
  return Point(result)
  
def reflection_in_edge(a,edge,inter):
  # LineString([(x0,y0),(x1,y1),]) Polygon([(0, 0), (1, 1), (1, 0)]) -> LineString([(x0,y0),(x1,y1),(x2,y2)]) 
  x=(edge.coords[0][0],edge.coords[1][0]) 
  y=(edge.coords[0][1],edge.coords[1][1])
  closest=ClosestPointOnLine(edge,a.coords[0])
  proj=(inter.coords[0][0]-closest.coords[0][0],inter.coords[0][1]-closest.coords[0][1])
  #The vector from the projected point to the intersection point
  ReflPt=(a.coords[0][0]+2*proj[0],a.coords[0][1]+2*proj[1])
  #ReflPt=(closest.coords[0][0]+2*proj[0]+opp[0],closest.coords[0][1]+2*proj[1]+opp[1])
  p.plot(ReflPt[0],ReflPt[1],'.')
  return LineString([inter,(ReflPt[0],ReflPt[1])])
  
def intersection_find_b(o,a,p):
  #if p.contains(a):
    #print('The ray is contained in a object')
    #return;
    #if p.geoms[1].contains(a):
    #print('The ray is contained in a object')
    #return;
    #if p.geoms[2].contains(a):
    #print('The ray is contained in a object')
    #return;
  #else:
    inter=p.intersection(a) # Assigns x to the intercepts of the line a and the Polygon polygon1
    edge=inter.geoms[0]
    x,y=coords_to_xy_lists(edge.coords) # lists the x,y coords of the intersection
    inter=Point(x[0],y[0])
    p1=p.geoms[0]
    p2=p.geoms[1]
    if p1.intersects(edge):
      p=p1
    if p2.intersects(edge):
      p=p2
    else:
        print('There is an error')
        return;
    return [inter,p,edge]
def intersection_find_a(o,a,p):
  if p.contains(a):
    print('The ray is contained in a object')
    return;
  else:
    inter=p.intersection(a) 
    # Assigns x to the intercepts of the line a and the Polygon polygon1
    print(inter)
    interfirst=inter.geoms[0]
    interfirstPT=Point(interfirst.coords[0][0],interfirst.coords[0][1])
    end=Point(a.coords[1][0],a.coords[1][1])
    l=LineString([interfirstPT,end])
    #x,y=coords_to_xy_lists(edge.coords) # lists the x,y coords of the intersection
    #inter=Point(x[0],y[0])
    p1=p.geoms[0]
    p2=p.geoms[1]
    p3=p.geoms[2]
    if p1.intersects(l):
      p=p1
    else:
      if p2.intersects(l):
        p=p2
      if p3.intersects(l):
        p=p3
      else:
        print('There is an error or no intersections')
        return;
    edge=edge_cut_by_point_in_polygon(interfirstPT,p)
    edge=LineString([edge[0],edge[1]])
    return [interfirstPT,p,edge]
  
def test():
  #Define the original ray and the setting
  a=LineString([(-2,0.5),(7,0.5),]) 
  polygon1 = Polygon([(0, 0), (1, 1), (1, 0)]) # Creates a polygon with exterior points (0,0), (1,1), (1,0)
  polygon2 = Polygon([(1, 1), (2, 0), (1, 2)]) 
  polygon3 = Polygon([(-1,0.75),(-1,4),(-1.5,4)])
  originlist=coords_to_xy_lists(a.coords)
  origin=Point(originlist[0][0],originlist[1][0])
  polygons=MultiPolygon([polygon1,polygon2,polygon3]);
  #x=(1,2,3)
  #for x in range(1,4):
  [inter,polygon,l]=intersection_find_a(origin,a,polygons)
  print('The line intersects the polygon at ', inter)
  x,y=coords_to_xy_lists(inter.coords) # lists the x,y coords of the intersection
  closest=ClosestPointOnLine(l,a.coords[0])
  print('The closest point on the polygon to the start point is ', closest)
  norma=(a.coords[1][0]-(a.coords[1][0]-a.coords[0][0])/a.length,a.coords[1][1]-(a.coords[1][1]-a.coords[0][1])/a.length)
  a=Point(norma)
  reflection=reflection_in_edge(a,l,inter)
  print('Reflection is ', reflection)
  ray=LineString([origin,inter])
  p.plot(closest.coords[0][0],closest.coords[0][1],'x')
  p.plot(*coords_to_xy_lists(origin.coords), '*')
  p.plot(*coords_to_xy_lists(reflection.coords))
  p.plot(*coords_to_xy_lists(ray.coords))   
  a=LineString([reflection.coords[0],reflection.coords[1]])
  origin=Point(a.coords[0])

  p.plot(*coords_to_xy_lists(polygon1.exterior.coords))
  p.plot(*coords_to_xy_lists(polygon2.exterior.coords))
  p.plot(*coords_to_xy_lists(polygon3.exterior.coords))
  p.plot(x,y) # Plots the intersection of the line and the polygon
  p.show()

if __name__=='__main__':
  test()


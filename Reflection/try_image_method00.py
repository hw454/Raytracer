

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

def Gradient(edge):
  print(edge)
  x,y=coords_to_xy_lists(edge.coords)
  if y[0]-y[1]==0:
    return;
  else:
    m=(x[0]-x[1])/(y[0]-y[1])
  return m;
def edge_cut_by_point_in_polygon(interobj,poly,o):
  # compute which edge (if any) of poly contains pt
  d1=poly.distance(o)
  for xy0,xy1 in zip(poly.exterior.coords[:-1],poly.exterior.coords[1:]):
    if LineString((xy0,xy1)).intersects(interobj):
      edge=LineString((xy0,xy1))
      d2=edge.distance(o)
      if d2==d1:
        p=(xy0,xy1)
        return p;
      elif d2<d1:
        p=(xy0,xy1)
        return p;


def ImagePoint(o,inter):
  #Project the pimage of the initial ray behind the object.
  result=(2*inter.coords[0][0]-o.coords[0][0],2*inter.coords[0][1]-o.coords[0][1])
  return Point(result)

def reflection_in_edge(o,edge,inter):
  ox=LineString([(o.coords[0][0],inter.coords[0][1]),(inter.coords[0][0],inter.coords[0][1])])
  oy=LineString([(inter.coords[0][0],o.coords[0][1],),(inter.coords[0][0],inter.coords[0][1])])
  Dx=ox.length
  Dy=oy.length
  I=ImagePoint(o,inter)
  print(I)
  print(Dy)
  print(Dx)
  #Find the Gradient of the edge and determine how to +- Dx and Dy
  ReflPt=Point(2*I.coords[0][0]-2*Dx,2*I.coords[0][1]+2*Dy)	  
  print(ReflPt)  
  return LineString([inter,ReflPt])
  
def intersection_find_a(o,a,p):
  if p.contains(a):
    print('The ray is contained in a object')
    return;
  else:
    inter=p.intersection(a) 
    # Assigns x to the intercepts of the line a and the Polygon polygon1
    interfirst=inter.geoms[0]
    interfirstPT=Point(interfirst.coords[0][0],interfirst.coords[0][1])
    interfirst=LineString([interfirstPT,(interfirst.coords[1][0],interfirst.coords[1][1])])
    end=Point(a.coords[1][0],a.coords[1][1])
    l=LineString([interfirstPT,end])
    #x,y=coords_to_xy_lists(edge.coords) # lists the x,y coords of the intersection
    #inter=Point(x[0],y[0])
    p1=p.geoms[0]
    p2=p.geoms[1]
    p3=p.geoms[2]
    if p1.intersects(interfirst):
      p=p1
    else:
      if p2.intersects(interfirst):
        p=p2
      if p3.intersects(interfirst):
        p=p3
      else:
        print('There is an error or no intersections')
        return;
    edge=edge_cut_by_point_in_polygon(interfirst,p,o)
    if edge==None:
      print('There is no edge')
      return;
    elif edge is not None:
      edge=LineString([edge[0],edge[1]])
    else:
      print('There has been an error, edge can not be none and not none')
      return;
    return [interfirstPT,p,edge]
  
def test():
  #Define the original ray and the setting
  a=LineString([(-2,0.3),(7,2),]) 
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
  #closest=ClosestPointOnLine(l,a.coords[0])
  #print('The closest point on the polygon to the start point is ', closest)
  #norma=(a.coords[1][0]-(a.coords[1][0]-a.coords[0][0])/length.a,a.coords[1][1]-(a.coords[1][1]-a.coords[0][1])/length.a)
  #o=Point(norma)
  reflection=reflection_in_edge(origin,l,inter)
  p.plot(reflection.coords[1][0],reflection.coords[1][1],'.')
  ray=LineString([origin,inter])
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


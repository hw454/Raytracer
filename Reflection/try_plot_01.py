#!/usr/bin/env python3

# Second Practice of using Shapely. Using the variable types line and polygon and plotting them.

from math import sin,cos,atan2
import matplotlib.pyplot as p
#from shapely.geometry.polygon import LinearRing, box, 
from shapely.geometry import Polygon, LineString, Point
from pprint import pprint

def coords_to_xy_lists(coords):
  # [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)] -> ([0.0,0.5,1.0],[0.0,1.0,0.0])
  return ([x for x,y in coords],[y for x,y in coords])


def intersection_of_point_and_polygon(a,poly):
  # Point(0.5,0.5), Polygon([(0, 0), (1, 1), (1, 0)]) -> True
  print('The initial point of intersection is ',a[0][0], a[1][0])
  print('The exterior of the polygon is ',poly.exterior)
  if Point(a[0][0], a[1][0]).intersects(poly.exterior):
    print(a[0][0], a[1][0], ' lies on ', poly.exterior)
  else: 
    print('The initial intercept is not on the boundary of the polygon')
  return Point(a[0][0], a[1][0]).intersects(poly.exterior)

def edge_cut_by_point_in_polygon(pt,poly):
  # compute which edge (if any) of poly contains pt
  for xy0,xy1 in zip(poly.exterior.coords[:-1],poly.exterior.coords[1:])
    if LineString((xy0,xy1)).intersects(pt):
      return (xy0,xy1)

def test():
  a=LineString([(0,0),(0.5,1),(1,0)]) # Creates a line through (0,0), (0.5,1) and (1,0)  
  a=LineString([(-1       ,0.5),(5,0.5),]) 
  polygon = Polygon([(0, 0), (1, 1), (1, 0)]) # Creates a polygon with exterior points (0,0), (1,1), (1,0)
  inter=polygon.intersection(a) # Assigns x to the intercepts of the line a and the Polygon polygon
  print(inter,type(inter)) # Gives the Geometry collection of intercepts
  p.plot(*coords_to_xy_lists(a.coords)) # Plots the line a
  p.plot(*coords_to_xy_lists(polygon.exterior.coords)) # Plots the exterior of the polygon
  x,y=coords_to_xy_lists(inter.coords) # lists the x,y coords of the intersection
  c=x,y
  print(polygon.interiors)
  #intersection_of_point_and_polygon(c, polygon)
  p.plot(x,y) # Plots the intersection
  p.show()

if __name__=='__main__':
  test()


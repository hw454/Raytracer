#!/usr/bin/env python3

# Second Practice of using Shapely. Using the variable types line and polygon and plotting them.

from math import sin,cos,atan2,hypot
import matplotlib.pyplot as p
#from shapely.geometry.polygon import LinearRing, box, 
from shapely.geometry import Polygon, LineString, Point
from pprint import pprint

def coords_to_xy_lists(coords):
  # [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)] -> ([0.0,0.5,1.0],[0.0,1.0,0.0])
  return ([x for x,y in coords],[y for x,y in coords])

def edge_cut_by_point_in_polygon(pt,poly):
  # compute which edge (if any) of poly contains pt
  for xy0,xy1 in zip(poly.exterior.coords[:-1],poly.exterior.coords[1:]):
    if LineString((xy0,xy1)).intersects(pt):
      return (xy0,xy1)


def reflection_in_edge(line,pt):
  # LineString([(x0,y0),(x1,y1),]) Polygon([(0, 0), (1, 1), (1, 0)]) -> LineString([(x0,y0),(x1,y1),(x2,y2)]) 
  x,y=coords_to_xy_lists(line.coords)
  d=pt.distance(line)
  return LineString([(x[0],y[0]),(pt[0],pt[1]),(x2,y2)])
  
def test():
  a=LineString([(-1,0.5),(5,0.5),]) 
  polygon = Polygon([(0, 0), (1, 1), (1, 0)]) # Creates a polygon with exterior points (0,0), (1,1), (1,0)
  inter=polygon.intersection(a) # Assigns x to the intercepts of the line a and the Polygon polygon
  x,y=coords_to_xy_lists(inter.coords) # lists the x,y coords of the intersection
  c=Point(x[0],y[0]) # c is th initial interseption
  line=edge_cut_by_point_in_polygon(c,polygon) # line gives the set of pairs which give the edge
  l=LineString([line[0],line[1]]) # l converts line into a LineString
  print(a.coords[0])
  d=Point(a.coords[0]).distance(l) # Shortest Distance from the start of the line to the polygon
  h=hypot(a.coords[0][0]-x[0],a.coords[0][1]-y[0])
  print(h)
  #b=reflection_in_edge(a,c)
  p.plot(*coords_to_xy_lists(a.coords))   
  p.plot(*coords_to_xy_lists(polygon.exterior.coords))
  p.plot(x,y) # Plots the intersection
  p.show()



if __name__=='__main__':
  test()


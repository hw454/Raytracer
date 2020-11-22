#!/usr/bin/env python3

from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def coords_to_xy_lists(c):
  return ([x for x,y in c],[y for x,y in c])

def test():
  p0=Polygon([(0,0),(1,1),(1,0)])
  p1=Polygon([(0,0.5),(1.5,1.5),(1.5,0.5)])
  plt.plot(*coords_to_xy_lists(p0.exterior.coords))
  plt.plot(*coords_to_xy_lists(p1.exterior.coords))
  x=p0.intersection(p1)
  print(tuple(x.exterior.coords))
  x,y=coords_to_xy_lists(x.exterior.coords)
  print(x,y)
  plt.plot(x,y)
  plt.show()

if __name__=='__main__':
  test()

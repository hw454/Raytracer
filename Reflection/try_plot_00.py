#!/usr/bin/env python3

# First practice of using shapely, reviewing variable types and intersections and variable measurements.


import matplotlib.pyplot as p
#from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon,box, LineString,Point
from pprint import pprint

print( Point(0,0).geom_type) # Gives the type for Point(0,0) which is Point
print( Point(0,0).distance(Point(1,1)) ) # Distance between Point(0,0) and Point(1,1)
a= LineString ([(0,0), (0.5,1),(1,0)]) # Creates a line through (0,0), (0.5,1) and (1,0)
polygon = Polygon([(0, 0), (1, 1), (1, 0)]) # creates a polygon with exterior points (0,0), (1,1), (1,0)
b = box(0.0, 0.0, 1.0, 1.0) # Creates a rectangle with corners (0,0), (1,0), (1,1), (0,1)
x=a.intersection(polygon) # Assigns x to the intercepts of the line a and the Polygon polygon
print(x) # Gives the Geometry collection of intercepts

ring = LinearRing([(0, 0), (1, 1), (1, 0)]) # Creates a linear ring through the coords

print (ring.area) # Rings have zero area
print(a.area) # Lines have 0 area
print(a.length) # length of the line
print (ring.length) # Rings have length



#!/usr/bin/env python3
# Hayley Wragg 2017-05-26
''' Module Containing simple functions for lines '''

import numpy as np
import matplotlib.pyplot as mp


def Direction(line):
  '''  Direction using co-ordinates input is a line
  (pair of co-ordinates) '''
  if (length(line)>0):
    return (line[1]-line[0])/length(line)
  else:
    return np.zeros(len(line[1]))

def Direction3D(line):
  '''  Direction using co-ordinates input is a line
  (pair of co-ordinates) '''
  if (length3D(line)>0):
    return (line[1]-line[0])/length3D(line)
  else:
    return np.array([0.0,0.0,0.0])

def length(line):
  ''' line is given as a pair of two co-ordinates. Output the length of
  the line '''
  length=np.linalg.norm(np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]]))
  return length

def angle(line1,line2):
  l1=length(line1)
  l2=length(line2)
  return np.arccos(np.dot(line1[1]-line1[0],line2[1]-line2[0])/(l1*l2))

def length3D(line):
  ''' line is given as a pair of two co-ordinates. Output the length of
  the line '''
  length=np.linalg.norm(np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]  ,line[1][2]- line[0][2]]))
  return length

def dot(p1, p2):
  return p1[0]*p2[0]+p1[1]*p2[1]


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
    return np.array([0.0,0.0])

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
  n=len(line[0])
  print(line)
  if n==3:
    length=np.linalg.norm(np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]]))
    return length
  elif n==2:
    length=np.linalg.norm(np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]  ,line[1][2]- line[0][2]]))
  elif: n==1:
    length=np.linalg.norm(np.array([line[1]-line[0]]))
  else:
    raise ValueError('The line is neither 3D nor 2D')
  return

def length3D(line):
  ''' line is given as a pair of two co-ordinates. Output the length of
  the line '''
  length=np.linalg.norm(np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]  ,line[1][2]- line[0][2]]))
  return length

def dot(p1, p2):
  return p1[0]*p2[0]+p1[1]*p2[1]

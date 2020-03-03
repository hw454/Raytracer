
#!/usr/bin/env python3
# Hayley Wragg 2017-05-26
''' Module Containing simple functions for lines '''

import numpy as np
import matplotlib.pyplot as mp


def Direction(line):
  '''  Direction using co-ordinates input is a line
  (pair of co-ordinates) '''
  if isinstance(line[0],float): n=1
  elif isinstance(line[0],int): n=1
  else:
    n=len(line[0])
  if (length(line)>0):
    return (line[1]-line[0])/length(line)
  else:
    return np.zeros(n)

def coordlistdistance(A):
  ''' A= nx3 array. Return nx1 array of the L2 norms of each row'''
  Out=np.sum(np.abs(A)**2,axis=-1)**(1./2)
  return Out

def coordlistdot(A,B):
  '''A=nx3 array, B=nx3 array. Return nx1 array of the dot product of the
  rows of A and B '''
  Out=np.sum(A*B,axis=-1)
  return Out

def length(line):
  ''' line is given as a pair of two co-ordinates. Output the length of
  the line '''
  length=np.linalg.norm(np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]]))
  return length

def length3D(line):
  ''' line is given as a pair of two co-ordinates. Output the length of
  the line '''
  length=np.linalg.norm(np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]  ,line[1][2]- line[0][2]]))
  return length

def dot(p1, p2):
  return p1[0]*p2[0]+p1[1]*p2[1]

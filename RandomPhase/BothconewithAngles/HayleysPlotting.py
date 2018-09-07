#!/usr/bin/env python3
# Hayley Wragg 2017-05-19

''' Functions for Plotting lines and points '''

import numpy as np
import matplotlib.pyplot as mp

def Plotline(line,l,c):
  ''' takes a 2D np array and plots the line in colour c. the line is
  given by an origin and direction. l is it's length'''
  line[1]=line[1]*l+line[0]
  mp.plot([line[0][0],line[1][0]], [line[0][1], line[1][1]], color=c)
  return

def Plotray(edge,c,wid):
  ''' takes a 2D np array and plots the line in colour c. the line is
  given by an origin and direction. l is it's length'''
  x=np.linspace(edge[0][0], edge[1][0], 100)
  y=np.linspace(edge[0][1], edge[1][1], 100)
  mp.plot(x,y,linewidth=wid, color=c)
  return

def Plotedge(edge,c,width):
  ''' takes a 2D np array and plots the line in colour c. the line is
  given by an origin and direction. l is it's length'''
  mp.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=c, linewidth=width)
  return

def Plotpoint(point):
  ''' takes a point and plots an x in it's position '''
  mp.plot(point[0],point[1],'x')
  return



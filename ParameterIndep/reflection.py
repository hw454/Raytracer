
#!/usr/bin/env python3
# Keith Briggs 2017-02-02
# Hayley Wragg 2017-03-28
# Hayley Wragg 2017-04-12
# Hayley Wragg 2017-05-15
''' Code to Reflect a line in an edge without using Shapely '''

from math import atan2,hypot,sqrt,copysign
import numpy as np
#import matplotlib.pyplot as mp
import HayleysPlotting as hp
import intersection as inter
import linefunctions as lf
#from mayavi.core.api import Engine
#from mayavi.sources.vtk_file_reader import VTKFileReader
#from mayavi.modules.surface import Surface
#from mayavi import mlab
#from pyface.api import GUI
import sys
import math as ma

epsilon=sys.float_info.epsilon

def try_reflect_ray(ray,triangle):
  ''' Reflection of a ray with a triangle'''
  # Find the distances which need to be translated
  trdist=-ray[1] # Make the intersection the origin
  direc=lf.Direction(ray)
  # Translate the points before the reflection
  ray[0]+=trdist
  ray[1]+=trdist
  edge1=triangle[0]-triangle[1]
  edge2=triangle[0]-triangle[2]
  edge3=triangle[2]-triangle[1]
  normedge=np.cross(edge1,edge2)
  normedge=normedge/la.norm(normedge)
  # Find the image of the ray in the edge
  # The image point is the point if the ray was to continue going through the surface
  Image=ray[1]+direc
  normcoef=np.dot(Image-ray[1],normedge)/np.dot(normedge,normedge)
  normedge=normcoef*normedge
  # Find the reflection using the Image
  ReflPt=Image-2*normedge
  #Translate Back
  ray[0]-=trdist
  ray[1]-=trdist
  normedge-=trdist
  ReflPt-=trdist
  return np.array([ray[1], ReflPt])

# FIXME The plotting for the tests should go elsewhere
def test():
  triangle=np.array([(0.0,0.0,1.0),(3.0,0.0,0.0),(0.0,0.0,0.0)])
  x=np.array([triangle[0][0],triangle[1][0],triangle[2][0],triangle[0][0]])
  y=np.array([triangle[0][1],triangle[1][1],triangle[2][1],triangle[0][1]])
  z=np.array([triangle[0][2],triangle[1][2],triangle[2][2],triangle[0][2]])
  #mlab.plot3d(x,y,z,color= (1, 1, 1))
  ray=np.array([(-1.0,-1.0,0.0),(3,3,2.25)])
  x=np.array([ray[0][0],ray[0][0]+ray[1][0]])
  y=np.array([ray[0][1],ray[0][1]+ray[1][1]])
  z=np.array([ray[0][2],ray[0][2]+ray[1][2]])
  #mlab.plot3d(x,y,z,color= (0, 1, 1))
  ray[1]=inter.intersection(ray,triangle)
  rayout=try_reflect_ray(ray,triangle)
  x=np.array([ray[0][0],rayout[0][0],rayout[1][0]])
  y=np.array([ray[0][1],rayout[0][1],rayout[1][1]])
  z=np.array([ray[0][2],rayout[0][2],rayout[1][2]])
  #mlab.plot3d(x,y,z,color= (0, 0, 1))
  #gui = GUI()
  #gui.start_event_loop()
  return 0

#FIXME test the refangle calculation
def refangle(line,obst):
  '''Find the reflection angle for the line reflection on the surface obst'''
  edge1=obst[0]-line[0]
  edge2=obst[2]-line[0]
  norm=np.cross(edge1,edge2)
  linedge=line[1]-line[0]
  angle=np.arccos((np.dot(norm,linedge)/(la.norm(norm)*la.norm(line))))
  return angle

def test3():
  '''angle test'''
  line=np.array([(0.0,5.0,0.0),(0.0,0.0,3.0)]) #FIXME
  surface=np.array([(-1.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0)])
  angle=refangle(line,surface)
  print(angle)
  return 0

def test2():
  triangle =np.array([(0.0,0.0,0.0),(3.0, 0.0,0.0),(1.5,1.5,0.0)])
  triangle2=np.array([(0.0,0.0,0.0),(3.0, 0.0,0.0),(1.0,1.0,3.0)])
  triangle3=np.array([(0.0,0.0,0.0),(1.5, 1.5,0.0),(1.0,1.0,3.0)])
  triangle4=np.array([(1.5,1.5,0.0),(1.0, 1.0,3.0),(3.0,0.0,0.0)])
  Trilist=10*np.array([triangle,triangle2,triangle3,triangle4])
  for l in range(0,4):
    x=np.array([Trilist[l][0][0],Trilist[l][1][0],Trilist[l][2][0],Trilist[l][0][0]])
    y=np.array([Trilist[l][0][1],Trilist[l][1][1],Trilist[l][2][1],Trilist[l][0][1]])
    z=np.array([Trilist[l][0][2],Trilist[l][1][2],Trilist[l][2][2],Trilist[l][0][2]])
    #mlab.plot3d(x,y,z,color= (1, 1, 1))
  ray=np.array([(1.5,1.0,0.25),(3,3,2.25)])
  r =10.0
  Nra=20
  theta=((2.0*ma.pi)/Nra)*np.arange(Nra)
  Tx=ray[0]
  TxMat       =np.array([Tx[0]*np.ones(Nra),Tx[1]*np.ones(Nra),Tx[2]*np.ones(Nra)])
  directions  =np.transpose(TxMat+r*np.array([np.cos(theta),np.sin(theta),np.zeros(Nra)])) #FIXME rotate in z axis too.
  for j in range(0,Nra):
    ray=np.array([Tx,directions[j]])
    x=np.array([ray[0][0],ray[0][0]+ray[1][0]])
    y=np.array([ray[0][1],ray[0][1]+ray[1][1]])
    z=np.array([ray[0][2],ray[0][2]+ray[1][2]])
    #mlab.plot3d(x,y,z,color= (0, 1, 1))
    for l in range(0,4):
      ray=np.array([Tx,directions[j]])
      ray[1]=inter.intersection(ray,Trilist[l])
      if np.isnan(ray[1][0]):
        pass
      else:
        rayout=try_reflect_ray(ray,Trilist[l])
        x=np.array([ray[0][0],rayout[0][0],rayout[1][0]])
        y=np.array([ray[0][1],rayout[0][1],rayout[1][1]])
        z=np.array([ray[0][2],rayout[0][2],rayout[1][2]])
        #mlab.plot3d(x,y,z,color= (0, 0, 1))
  #gui = GUI()
  #gui.start_event_loop()
  return 0

def try_3D_reflect_ray(ray,plane):
  ''' Take a ray and find it's reflection in a plane. '''
  # Find the distances which need to be translated
  trdist=-ray[1]
  direc=lf.Direction3D(ray)
  # Translate the points before the reflection
  o=trdist
  ray+=trdist
  plane[0]+=trdist
  # Find the image of the ray in the edge
  Image=ray[1]+direc
  # Find the normal to the edge
  normedge=plane[1]
  # Find the reflection using the Image
  ReflPt=Image-2*np.dot(normedge,Image)*normedge
  #Translate Back
  ray-=trdist
  plane[0]-=trdist
  normedge-=trdist
  ReflPt-=trdist
  o-=trdist
  return np.array([ray[1], ReflPt])

def errorcheck(err,ray,ref,normedge):
  ''' Take the input ray and output ray and the normal to the edge,
  check that both vectors have the same angle to the normal'''
  # Convert to the Direction vectors
  ray=lf.Direction(ray)
  normedge=lf.Direction(normedge)
  ref=lf.Direction(ref)
  # Find the angles
  #FIXME check the cosine rule
  theta1=np.arccos(abs(np.dot(ray,normedge))/(np.linalg.norm(ray)*np.linalg.norm(normedge)))
  theta2=np.arccos(abs(np.dot(ref,normedge))/(np.linalg.norm(ref)*np.linalg.norm(normedge)))
  if (abs(theta1-theta2)>1.0E-7):
    err+=1.0
  return err

if __name__=='__main__':
  test3()
  print('Running  on python version')
  print(sys.version)
  exit()


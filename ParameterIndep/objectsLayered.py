
#!/usr/bin/env python3
# Keith Briggs 2017-02-02
# Hayley Wragg 2017-03-28
# Hayley Wragg 2017-04-12
# Hayley Wragg 2017-05-15
# Hayley Wragg 2017-07-10
''' Code to construct the ray-tracing objects. Constructs wall-segments,
 rays'''


from math import atan2,sin,cos,hypot,sqrt,copysign
import numpy as np
import reflection as ref
import intersection as ins
import linefunctions as lf
import HayleysPlotting as hp
import matplotlib.pyplot as mp
import math as ma
from math import sin,cos,atan2
import cmath as cma
import roommesh as rmes
import time as t
import random as rnd

class Wall_segment:
  ' a line segment from p0 to p1 '
  def __init__(s,p0,p1):
    assert not (p0==p1).all()
    s.p=np.vstack(
      (np.array(p0,dtype=np.float),
       np.array(p1,dtype=np.float),
    ))
  def __getitem__(s,i):
    return s.p[i]
  def __str__(s):
    return 'Wall_segment('+str(list(s.p))+')'

class room:
  ' A group of wall_segment'
  def __init__(s,walls,heights):
    s.objects=list((()))
    s.heights=list(heights)
    s.points=list()
    s.time=np.array([0.0,0.0])
  def __getwall__(s,i):
   ''' Returns the ith wall of s '''
   return s.objects[0][i]
  def add_pts(s,obj,i):
    ''' Adds the edge to the objects in s, and the points of to the
    points in s at layer i'''
   # s.objects[i]+=((obj),)
    s.points[i]+=(obj,)
    return
  def __str__(s):
    return 'Rooom('+str(list(s.objects[0]))+')'
  def maxleng(s):
    ''' Finds the maximum length contained in the room '''
    leng=0
    wallpoints=s.points[0]
    p1=wallpoints[-1]
    for p2 in wallpoints[0:]:
      leng2=lf.length(np.array([p1,p2]))
      if leng2>leng:
        leng=leng2
    return leng
  def Plotroom(s,origin):
    ''' Plots all the edges in the room '''
    mp.plot(origin[0],origin[1],marker='x',c='r')
    for edge in s.objects[-1]:
      hp.Plotedge(np.array(edge.p),c='r')
    return
  def roomconstruct(s,Obstaclelayers):
    ''' Takes in a set of wall segments and constructs a room object
    containing them all'''
    i=0
    for level in Obstaclelayers[0:]:
      s.objects+=(level,)
      layer=list()
      for obj in level:
        layer.extend(obj)
      s.points.append(layer)
      i+=1
    return
  def xbounds(s):
    walls=s.objects[0]
    xarray=np.vstack(np.array([walls[0][0][0]]))
    for wall in walls:
      xarray=np.vstack((xarray,wall[0][0],wall[1][0]))
    return np.array([min(xarray)[0],max(xarray)[0]])
  def ybounds(s):
    walls=s.objects[0]
    yarray=np.vstack(np.array([walls[0][0][1]]))
    for wall in walls:
      yarray=np.vstack((yarray,wall[0][1],wall[1][1]))
    return np.array([min(yarray)[0],max(yarray)[0]])
  def roommesh(s,spacing):
    return rmes.roommesh((s.xbounds()),(s.ybounds()),spacing)
  def uniform_ray_tracer(s,origin,n,fig,spacing,frequency,streg,m,refloss):
    start_time=t.time()
    ''' Traces ray's uniforming emitted from an origin around a room.
    Number of rays is n, number of reflections m'''
    pi=4*np.arctan(1) # numerically calculate pi
    Mesh=s.roommesh(spacing)
    i=1
    losscoef=((4.0*ma.pi*frequency)/(2.99792458*1.0E+8))**2
    start=streg/losscoef
    for h in s.heights:
        if h == s.heights[-1]:
          break
        for j in range(0,n+1):
          wallstart=start*(s.heights[i]-h)/s.heights[-1]
          theta=(2*j*pi)/n
          r=s.maxleng()
          xtil=ma.cos(theta)
          ytil=ma.sin(theta)
          x= r*xtil+origin[0]
          y= r*ytil+origin[1]
          ray=Ray(frequency,wallstart,origin,(x,y))
          # Reflect the ray
          ray.multiref(s.objects[-(i-1)],r,m)
          mp.figure(fig)
          ray.Plotray(s)
          mp.figure(fig+1)
          Mesh=ray.heatmapray(Mesh,ray.streg,ray.frequency,spacing,refloss)
        i+=1
    Mesh.hist(fig+2)
    end_time=(t.time() - start_time)
    s.time[0]=end_time
    print("Time to compute unbounded--- %s seconds ---" % end_time )
    mp.figure(fig)
    mp.title('Ray paths')
    s.Plotroom(origin)
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/RaysLayered'+str(i)+'.jpg',bbox_inches='tight')
    mp.figure(fig+1)
    s.Plotroom(origin)
    mp.title('Heatmap')
    mp.savefig('../../ImagesOfSignalStrength/FiguresNew/HeatmapLayered'+str(i)+'.jpg',bbox_inches='tight')
    #mp.figure(i+4)
    #s.Plotroom(origin)
    #mp.savefig('../../ImagesOfSignalStrength/FiguresNew/Room.jpg',bbox_inches='tight')
    return
  def uniform_ray_tracer_bounded(s,origin,n,fig,spacing,frequency,streg,m,bounds,refloss):
    ''' Traces ray's uniforming emitted from an origin around a room.
    Number of rays is n, number of reflections m'''
    start_time=t.time()
    pi=4*np.arctan(1) # numerically calculate pi
    Mesh=s.roommesh(spacing)
    i=1
    losscoef=((4.0*ma.pi*frequency)/(2.99792458*1.0E+8))**2
    start=streg/losscoef
    for h in s.heights:
      if h == s.heights[-1]:
        break
      for j in range(0,n+1):
        wallstart=start*(s.heights[i]-h)/s.heights[-1]
        theta=(2*j*pi)/n
        r=s.maxleng()
        xtil=ma.cos(theta)
        ytil=ma.sin(theta)
        x= r*xtil+origin[0]
        y= r*ytil+origin[1]
        ray=Ray(frequency,start,origin,(x,y))
        # Reflect the ray
        ray.multiref(s.objects[1-i],m,r)
        mp.figure(fig)
        ray.Plotray(s)
      i+=1
    end_time=(t.time() - start_time)
    s.time[1]=end_time
    print("Time to compute bounded--- %s seconds ---" % end_time)
    mp.figure(fig)
    s.Plotroom(origin)
    mp.title('Ray paths')
    mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefRaysLayered.png',bbox_inches='tight')
    #mp.figure(i+16)
    #s.Plotroom(origin)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/temproomLayered'+str(fig)+'.png',bbox_inches='tight')
    #s.Plotroom(origin)
    #mp.figure(i+1)
    #s.Plotroom(origin)
    #mp.colorbar()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/NoRNDHeatmapBoundsLayered'+str(fig)+'.png',bbox_inches='tight')
    #mp.figure(i+2)
    #s.Plotroom(origin)
    #mp.colorbar()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefHeatmapBoundsLayered'+str(fig)+'.png',bbox_inches='tight')
    #mp.figure(i+3)
    #s.Plotroom(origin)
    #mp.colorbar()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumHeatmapBoundsLayered'+str(fig)+'.png',bbox_inches='tight')
    mp.figure(fig+1)
    s.Plotroom(origin)
    cbar=mp.colorbar()
    cbar.set_label('Field strength in dBm', rotation=270)
    mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothHeatmapBoundsLayered'+str(fig)+'.png',bbox_inches='tight')
    #Mesh0.histbounded(i+5)
    #mp.figure(i+5)
    #mp.title('Cumulative Frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/NoRNDRNDCumsumBoundsLayered'+str(fig)+'.png', bbox_inches='tight')
    #mp.figure(i+6)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/NoRNDHistogramBoundsLayered'+str(fig)+'.png',bbox_inches='tight')
    #Mesh1.histbounded(i+7)
    #mp.figure(i+7)
    #mp.title('Cumulative Frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefRNDCumsumBoundsLayered'+str(fig)+'.png', bbox_inches='tight')
    #mp.figure(i+8)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefHistogramBounds'+str(fig)+'.png',bbox_inches='tight')
    #Mesh2.histbounded(i+9)
    #mp.figure(i+9)
    #mp.title('Cumulative Frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumRNDCumsumBoundsLayered'+str(fig)+'.png', bbox_inches='tight')
    #mp.figure(i+10)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumHistogramBoundsLayered'+str(fig)+'.png',bbox_inches='tight')
    Mesh3.histbounded(fig+2)
    mp.figure(fig+2)
    mp.title('Cumulative frequency of field strength')
    mp.grid()
    mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothRNDCumsumBoundsLayered'+str(fig)+'.png', bbox_inches='tight')
    mp.figure(fig+3)
    mp.title('Histrogram of field strength')
    mp.grid()
    mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothHistogramBoundsLayered'+str(fig)+'.png',bbox_inches='tight')
    #mp.figure(i+14)
    #RefSumDiff=Mesh1.meshdiff(Mesh2)
    #mp.title('Residual between phase change on reflection and change on sumation')
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/ResidualRefSumboundedLayered.png',bbox_inches='tight')
    #mp.figure(i+15)
    #RefSumDiff=Mesh0.meshdiff(Mesh3)
    #mp.title('Residual between No phase change and phase change on reflection and change on sumation')
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/ResidualNoPhasePhaseboundedLayered.png',bbox_inches='tight')
    return fig+4, spacing

class Ray:
  ''' represents a ray by a collection of line segments followed by
      an origin and direction.  A Ray will grow when reflected replacing
       the previous direction by the recently found intersection and
       inserting the new direction onto the end of the ray.
  '''
  def __init__(s,freq,start,origin,direction):
    s.ray=np.vstack(
      (np.array(origin,   dtype=np.float),
       np.array(direction,dtype=np.float),
    ))
    s.frequency=freq #, dtype=np.float
    s.streg=start    #, dtype=np.float
  def __str__(s):
    return 'Ray(\n'+str(s.ray)+')'
  def _get_origin(s):
    ''' The second to last term in the np array is the starting
    co-ordinate of the travelling ray '''
    return s.ray[-2]
  def _get_direction(s):
    ''' The direction of the travelling ray is the last term in the ray
    array. '''
    return s.ray[-1]
  def _get_travellingray(s):
    '''The ray which is currently travelling. Should return the recent
    origin and direction. '''
    return [s._get_origin(), s._get_direction()]
  def wall_collision_point(s,wall_segment):
    ''' intersection of the ray with a wall_segment '''
    return ins.intersection(s._get_travellingray(),wall_segment)
  def room_collision_point(s,roomedges,leng):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room. Returns the intersection point and the
    wall intersected with '''
    # Initialise the point and wall
    rwall=roomedges[0]
    rcp=s.wall_collision_point(rwall)
    # Find the intersection with all the walls and check which is the
    #closest. Verify that the intersection is not the current origin.
    for wall in roomedges[1:]:
      cp=s.wall_collision_point(wall)
      if (cp[0] is not None and (cp!=s.ray[-2]).all()):
        leng2=s.ray_length(cp)
        if (leng2<leng) :
          leng=leng2
          rcp=cp
          rwall=wall
    return rcp, rwall
  def ray_length(s,inter):
    '''The length of the ray upto the intersection '''
    o=s._get_origin()
    ray=np.array([o,inter])
    return lf.length(ray)
  def reflect(s,roomedges,leng):
    ''' finds the reflection of the ray inside a room'''
    cp,wall=s.room_collision_point(roomedges,leng)
    # Check that a collision does occur
    if cp[0] is None: return
    else:
      # Construct the incoming array
      origin=s._get_origin()
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray,n=ref.try_reflect_ray(ray,wall.p)
      # update self...
      s.ray[-1]=cp
      s.ray=np.vstack((s.ray,lf.Direction(refray)))
    return
  def multiref(s,room,m,r):
    ''' Takes a ray and finds the first five reflections within a room'''
    for i in range(1,m+1):
      s.reflect(room,r)
    # print('iteraction', i, 'ray', s.ray)
    return
  def Plotray(s,room):
    ''' Plots the ray from point to point and the final travelling ray
    the maximum length in the room '''
    rayline1=s.ray[0]
    wid=7
    for rayline2 in s.ray[0:-1]:
      wid=wid*0.5
      hp.Plotray(np.array([rayline1, rayline2]),'BlueViolet',wid)
      rayline1=rayline2
    #hp.Plotline(s.ray[-2:],room.maxleng(),'b')
    return
  def raytest(s,room,err):
    ''' Checks the reflection for errors'''
    cp,wall=s.room_collision_point(room)
    # Check that a collision does occur
    if cp[0] is None: return
    else:
      # Construct the incoming array
      origin=s._get_origin()
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray,n=ref.try_reflect_ray(ray,wall.p)
      err=ref.errorcheck(err,ray,refray,n)
      # update self...
      s.ray[-1]=cp
      s.ray=np.vstack((s.ray,lf.Direction(refray)))
      print('ray',ray, 'refray', refray, 'error', err)
    return err
  def heatmapray(s,Mesh,streg,freq,spacing,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    #Mesh.plot()
    return Mesh
  def heatmaprayrndref(s,Mesh,streg,freq,spacing,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      phase=rnd.uniform(0,2)
      refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    #Mesh.plot()
    return Mesh
  def heatmaprayrndsum(s,Mesh,streg,freq,spacing,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    #Mesh.plot()
    return Mesh
  def heatmaprayrndboth(s,Mesh,streg,freq,spacing,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-2]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      phase=rnd.uniform(0,2)
      refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    #Mesh.plot()
      #theta=rnd.uniform(0,2*ma.pi)
      #phase=cma.exp(ma.pi*complex(0,1)*theta)
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    Mesh.plot()
    return Mesh
  def heatmaprayboundedrndref(s,Mesh,streg,freq,spacing,bounds,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      phase=rnd.uniform(0,2)
      refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    Mesh.bound(bounds)
    #Mesh.plot()
    return Mesh
  def heatmaprayboundedrndsum(s,Mesh,streg,freq,spacing,bounds,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    Mesh.bound(bounds)
    #Mesh.plot()
    return Mesh
  def heatmaprayboundedboth(s,Mesh,streg,freq,spacing,bounds,refloss):
    i=0
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      phase=rnd.uniform(0,2)
      refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      i+=1
    Mesh.bound(bounds)
    #Mesh.plot()
    return Mesh





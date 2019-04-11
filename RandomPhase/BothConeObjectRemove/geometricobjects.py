#!/usr/bin/env python3
# Hayley Wragg 2018-05-10
''' Code to construct the ray-tracing objects. Constructs wall-segments,
 rays'''

from math import atan2,hypot,sqrt,copysign
import numpy as np
import reflection as ref
import intersection as ins
import linefunctions as lf
import HayleysPlotting as hp
import matplotlib.pyplot as mp
import math as ma
from math import sin,cos,atan2
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
  def firstpoint(s):
    return s.p[0]
  def secondpoint(s):
    return s.p[1]
  def __str__(s):
    return 'Wall_segment('+str(list(s.p))+')'

class room:
  ' A group of wall_segments and the time for the run'
  def __init__(s,wall0):
    s.walls=list((wall0,))
    s.points=list(wall0)
    s.inside_points=np.array([])
    s.objectcorners=np.array([])
    s.time=np.array([0.0,0.0])
  def __getwall__(s,i):
   ''' Returns the ith wall of s '''
   return s.walls[i]
  def __getinsidepoint__(s,i,j):
   ''' Returns the ith wall of s '''
   return s.inside_points[i][j]
  def add_wall(s,wall):
    ''' Adds wall to the walls in s, and the points of the wall to the
    points in s '''
    s.walls+=(wall,)
    s.points+=wall
    return
  def __str__(s):
    return 'Rooom('+str(list(s.walls))+')'
  def maxleng(s):
    ''' Finds the maximum length contained in the room '''
    leng=0
    p1=s.points[-1]
    for p2 in s.points:
      leng2=lf.length(np.array([p1,p2]))
      if leng2>leng:
        leng=leng2
    return leng
  def Plotroom(s,origin,width):
    ''' Plots all the edges in the room '''
    mp.plot(origin[0],origin[1],marker='x',c='r')
    for wall in s.walls:
      hp.Plotedge(np.array(wall.p),'g',width)
    return
  def roomconstruct(s,walls):
    ''' Takes in a set of wall segments and constructs a room object
    containing them all'''
    for wall in walls[1:]:
      s.add_wall(wall)
    return
  def xbounds(s):
    xarray=np.vstack(np.array([s.walls[0][0][0]]))
    for wall in s.walls:
      xarray=np.vstack((xarray,wall[0][0],wall[1][0]))
    return np.array([min(xarray)[0],max(xarray)[0]])
  def ybounds(s):
    yarray=np.vstack(np.array([s.walls[0][0][1]]))
    for wall in s.walls:
      yarray=np.vstack((yarray,wall[0][1],wall[1][1]))
    return np.array([min(yarray)[0],max(yarray)[0]])
  def add_inside_objects(s,corners):
    if s.objectcorners.shape[0]==0:
      n=0
      j=0
    else:
      n=s.objectcorners[-1][0]
      j=n
    for x in corners:
      if j==0: s.inside_points=x.firstpoint()
      else: s.inside_points=np.vstack((s.inside_points,x.firstpoint()))
      j+=1
    if n==0: s.objectcorners=np.array([(n,j-1)])
    else:
      s.objectcorners=np.vstack((s.objectcorners,np.array([(n,j-1)])))
    return
  def roommesh(s,spacing):
     #FIXME the room needs the object bounds
    return rmes.roommesh(s.inside_points,s.objectcorners,(s.xbounds()),(s.ybounds()),spacing)
  def room_collision_point_with_end(s,line,space):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room for a line with an end point.
    Returns the intersection point if intersections occurs'''
    # Retreive the Maximum length from the Room
    # Find whether there is an intersection with any of the walls.
    cp, mu=ins.intersection_with_end(line,s.walls[0],space)
    if cp==1: count=1
    else: count=0
    for wall in s.walls[1:]:
      cp, mu=ins.intersection_with_end(line,wall,space)
      if cp==1:
        count+=1
    if count % 2 ==0:
      return 0
    elif count % 2 ==1:
      return 1
    else: return 2 # This term shouldn't happen
  def uniform_ray_tracer(s,origin,outsidepoint1,outsidepoint2,n,ave,i,frequency,start,m,refloss):
    start_time=t.time()         # Start the time counter
    ''' Traces ray's uniformly emitted from an origin around a room.
    Number of rays is n, number of reflections m. Then calculates loss
    along each ray, this is output as a mesh of values where rays which
    go through the same square have their values added together.'''
    #pi=4*np.arctan(1) # numerically calculate pi
    r=s.maxleng()
    spacing=0.25#*ma.sin((2*ma.pi)/n)*(r*np.sqrt(2))
    #spacing=0.05
    Mesh0=s.roommesh(spacing) # Mesh for the case with no phase
    Mesh1=s.roommesh(spacing) # Mesh for the initial case with phase
    Mesh2=s.roommesh(spacing) # Mesh for the averaged case with phase
    #Mesh3=s.roommesh(spacing)
    #Mesh4=s.roommesh(spacing)
    k=int((origin[0]- Mesh0.__xmin__())/spacing)
    l=int((Mesh0.__ymax__()-origin[1])/spacing)
    Mesh0.grid[l][k]+=start
    Mesh1.grid[l][k]+=start
    Mesh2.grid[l][k]+=start
    #Mesh3.grid[l][k]+=start
    #Mesh4.grid[l][k]+=start
    #start=start/n
    #FIXME
    for it in range(0,n+1):
      theta=(2*it*ma.pi)/n
      xtil=ma.cos(theta)
      ytil=ma.sin(theta)
      x= r*xtil+origin[0]
      y= r*ytil+origin[1]
      ray=Ray(frequency,start,origin,(x,y))
      # Reflect the ray
      ray.multiref(s,m)
      mp.figure(i)
      ray.Plotray(s)
      Mesh0=ray.heatmapray(Mesh0,ray.streg,ray.frequency,spacing,refloss,n)
      #mp.figure(i+2)
      #mp.title('Heatmap phase change on ref.')
      #Mesh1=ray.heatmaprayrndref(Mesh1,ray.streg,ray.frequency,spacing,refloss)
      #mp.figure(i+3)
      #mp.title('Heatmap phase change on sum.')
      #Mesh2=ray.heatmaprayrndsum(Mesh2,ray.streg,ray.frequency,spacing,refloss)
      Mesh1=ray.heatmaprayrndboth(Mesh1,ray.streg,ray.frequency,spacing,refloss,n)
    mp.figure(i+1)
    PowerMeshNoRND,ext1=Mesh0.powergrid(s,origin,outsidepoint1,outsidepoint2)
    mp.figure(i+2)
    PowerMeshRND,ext2=Mesh1.powergrid(s,origin,outsidepoint1,outsidepoint2)
    end_time=(t.time() - start_time)
    s.time[0]=end_time
    print("Time to compute unbounded--- %s seconds ---" % end_time )
    mp.figure(i)
    #mp.title('Ray paths')
    s.Plotroom(origin,1.0)
    mp.savefig('../../../../NewFigures/Rays'+str(i)+'.png',bbox_inches='tight')
    #mp.figure(i+2)
    ##s.Plotroom(origin)
    #cbar=mp.colorbar()
    #cbar.set_label('Power in dBm', rotation=270)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefHeatmap'+str(i)+'.png',bbox_inches='tight')
    #mp.figure(i+3)
    ##s.Plotroom(origin)
    #cbar=mp.colorbar()
    #cbar.set_label('Power in dBm', rotation=270)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumHeatmap'+str(i)+'.png',bbox_inches='tight')
    #mp.figure(i+2)
    #s.Plotroom(origin,1/(spacing))
    #c0=Mesh0.hist(i+5)
    #mp.figure(i+5)
    #mp.title('Cumulative frequency of power')
    #mp.grid()
    #mp.savefig('../../../../NewFigures/ConeNoPhaseCumsum'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+6)
    #mp.title('Histogram of power')
    #mp.grid()
    #mp.savefig('../../../../NewFigures/ConeNoPhaseHistogram'+str(i)+'.png',bbox_inches='tight')
    #c1=Mesh1.hist(i+7)
    #mp.figure(i+7)
    #mp.title('Cumulative frequency of power')
    #mp.grid()
    #mp.savefig('../../../../NewFigures/AveragedCumsumwithPhase'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+8)
    #mp.title('Histogram of power')
    #mp.grid()
    #mp.savefig('../../../../NewFigures/ConeAveragedHistogramwithPhase'+str(i)+'.png',bbox_inches='tight')
    #c2=Mesh2.hist(i+9)
    #mp.figure(i+9)
    #mp.title('Cumulative frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumRNDCumsum'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+10)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumHistogramNoBounds'+str(i)+'.png',bbox_inches='tight')
    #c3=Mesh4.hist(i+11)
    #mp.figure(i+11)
    #mp.title('Cumulative frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothRNDCumsum'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+12)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothHistogramNoBounds'+str(i)+'.png',bbox_inches='tight')
    ##mp.figure(i+13)
    #s.Plotroom(origin)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefRoom.png',bbox_inches='tight')
    #mp.figure(i+14)
    #RefSumDiff=Mesh1.meshdiff(Mesh2)
    #mp.title('Residual- Phase change on ref. and change on sum.')
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/ResidualRefSumUnbounded.png',bbox_inches='tight')
    #mp.figure(i+9)
    #RefSumDiff=Mesh0.meshdiff(Mesh1)
    #mp.title('Residual- No phase change and phase change')
    #mp.savefig('../../../../NewFigures/ConeResidualNoPhasePhasepng',bbox_inches='tight')
    return i+16,spacing,PowerMeshNoRND,PowerMeshRND
  #def uniform_ray_tracer_bounded(s,origin,n,i,frequency,start,m,bounds,refloss):
    #''' Traces ray's uniforming emitted from an origin around a room.
    #Number of rays is n, number of reflections m'''
    #start_time=t.time()
    #pi=4*np.arctan(1) # numerically calculate pi
    #r=s.maxleng()
    ##spacing=2*pi*r/n
    #spacing=ma.sin((2*pi)/n)*(r*np.sqrt(2))
    #Mesh0=s.roommesh(spacing)
    #Mesh1=s.roommesh(spacing)
    #Mesh2=s.roommesh(spacing)
    #Mesh3=s.roommesh(spacing)
    #k=int((origin[0]- Mesh0.__xmin__())/spacing)
    #l=int((Mesh0.__ymax__()-origin[1])/spacing)
    #Mesh0.grid[l][k]+=start
    #Mesh1.grid[l][k]+=start
    #Mesh2.grid[l][k]+=start
    #Mesh3.grid[l][k]+=start
    #for j in range(0,n+1):
      #theta=(2*j*pi)/n
      #xtil=ma.cos(theta)
      #ytil=ma.sin(theta)
      #x= r*xtil+origin[0]
      #y= r*ytil+origin[1]
      #ray=Ray(frequency,start,origin,(x,y))
      ## Reflect the ray
      #ray.multiref(s,m)
      #mp.figure(i)
      ##ray.Plotray(s)
      #mp.figure(i+1)
      #mp.title('Heatmap bounded no phase change')
      #Mesh0=ray.heatmapraybounded(Mesh0,ray.streg,ray.frequency,spacing,bounds,refloss)
      #mp.figure(i+2)
      #mp.title('Heatmap bounded phase change on ref.')
      #Mesh1=ray.heatmaprayboundedrndref(Mesh1,ray.streg,ray.frequency,spacing,bounds,refloss)
      #mp.figure(i+3)
      #mp.title('Heatmap bounded phase change on sum.')
      #Mesh2=ray.heatmaprayboundedrndsum(Mesh2,ray.streg,ray.frequency,spacing,bounds,refloss)
      #mp.figure(i+4)
      #mp.title('Heatmap bounded phase change on ref. and sum.')
      #Mesh3=ray.heatmaprayboundedboth(Mesh3,ray.streg,ray.frequency,spacing,bounds,refloss)
    #end_time=(t.time() - start_time)
    #s.time[1]=end_time
    #print("Time to compute bounded--- %s seconds ---" % end_time)
    #mp.figure(i)
    #mp.title('Ray paths')
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefRays.png',bbox_inches='tight')
    #mp.figure(i+16)
    ##s.Plotroom(origin)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/temproom'+str(i)+'.png',bbox_inches='tight')
    ##s.Plotroom(origin)
    #mp.figure(i+1)
    ##s.Plotroom(origin)
    #cbar=mp.colorbar()
    #cbar.set_label('Field strength in dBm', rotation=270)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/NoRNDHeatmapBounds'+str(i)+'.png',bbox_inches='tight')
    #mp.figure(i+2)
    ##s.Plotroom(origin)
    #cbar=mp.colorbar()
    #cbar.set_label('Field strength in dBm', rotation=270)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefHeatmapBounds'+str(i)+'.png',bbox_inches='tight')
    #mp.figure(i+3)
    ##s.Plotroom(origin)
    #cbar=mp.colorbar()
    #cbar.set_label('Field strength in dBm', rotation=270)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumHeatmapBounds'+str(i)+'.png',bbox_inches='tight')
    #mp.figure(i+4)
    ##s.Plotroom(origin)
    #cbar=mp.colorbar()
    #cbar.set_label('Field strength in dBm', rotation=270)
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothHeatmapBounds'+str(i)+'.png',bbox_inches='tight')
    #c0=Mesh0.histbounded(i+5)
    #mp.figure(i+5)
    #mp.title('Cumulative frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/NoRNDRNDCumsumBounds'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+6)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/NoRNDHistogramBounds'+str(i)+'.png',bbox_inches='tight')
    #c1=Mesh1.histbounded(i+7)
    #mp.figure(i+7)
    #mp.title('Cumulative frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefRNDCumsumBounds'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+8)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnRefHistogramBounds'+str(i)+'.png',bbox_inches='tight')
    #c2=Mesh2.histbounded(i+9)
    #mp.figure(i+9)
    #mp.title('Cumulative frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumRNDCumsumBounds'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+10)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnSumHistogramBounds'+str(i)+'.png',bbox_inches='tight')
    #c3=Mesh3.histbounded(i+11)
    #mp.figure(i+11)
    #mp.title('Cumulative frequency of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothRNDCumsumBounds'+str(i)+'.png', bbox_inches='tight')
    #mp.figure(i+12)
    #mp.title('Histrogram of field strength')
    #mp.grid()
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/OnBothHistogramBounds'+str(i)+'.png',bbox_inches='tight')
    #mp.figure(i+14)
    #RefSumDiff=Mesh1.meshdiff(Mesh2)
    #mp.title('Residual- Phase change on ref and change on sum')
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/ResidualRefSumbounded.png',bbox_inches='tight')
    #mp.figure(i+15)
    #RefSumDiff=Mesh0.meshdiff(Mesh3)
    #mp.title('Residual- No phase change and phase change')
    #mp.savefig('../../../../ImagesOfSignalStrength/FiguresNew/RandomPhase/ResidualNoPhasePhasebounded.png',bbox_inches='tight')
    #return i+16, spacing,Mesh3

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
  def room_collision_point(s,room):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room. Returns the intersection point and the
    wall intersected with '''
    # Retreive the Maximum length from the Room
    leng=room.maxleng()
    # Initialise the point and wall
    rwall=room.walls[0]
    rcp=s.wall_collision_point(rwall)
    # Find the intersection with all the walls and check which is the
    #closest. Verify that the intersection is not the current origin.
    for wall in room.walls[1:]:
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
  def reflect(s,room):
    ''' finds the reflection of the ray inside a room'''
    cp,wall=s.room_collision_point(room)
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
  def multiref(s,room,m):
    ''' Takes a ray and finds the first five reflections within a room'''
    for i in range(1,m+1):
      s.reflect(room)
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
  def heatmapray(s,Mesh,streg,freq,spacing,refloss,N):
    i=0
    streg=streg*(299792458/(freq*4*ma.pi))
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency,N)
      i+=1
    #Mesh.plot()
    return Mesh
  #def heatmaprayrndref(s,Mesh,streg,freq,spacing,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #phase=rnd.uniform(0,2)
      #refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.plot()
    #return Mesh
  #def heatmaprayrndsum(s,Mesh,streg,freq,spacing,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.plot()
    #return Mesh
  def heatmaprayrndboth(s,Mesh,streg,freq,spacing,refloss,N):
    i=0
    streg=streg*(299792458/(freq*4*ma.pi))
    iterconsts=np.array([streg,1.0])
    for r in s.ray[:-3]:
      #In db
      #refloss=10*np.log10(2)
      #streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      #In Watts
      phase=0.0 #rnd.uniform(0,2)
      refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      iterconsts[0]=iterconsts[0]/refloss
      iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency,N)
      i+=1
    #Mesh.plot()
    return Mesh
  #def heatmapraybounded(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh
  #def heatmaprayboundedrndref(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #phase=rnd.uniform(0,2)
      #refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singleray(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh
  #def heatmaprayboundedrndsum(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh
  #def heatmaprayboundedboth(s,Mesh,streg,freq,spacing,bounds,refloss):
    #i=0
    #streg=streg*(299792458/(freq*4*ma.pi))
    #iterconsts=np.array([streg,1.0])
    #for r in s.ray[:-3]:
      ##In db
      ##refloss=10*np.log10(2)
      ##streg=Mesh.singleray(np.array([r,s.ray[i+1]]),streg-refloss,s.frequency)
      ##In Watts
      #phase=rnd.uniform(0,2)
      #refloss=refloss*np.exp(ma.pi*phase*complex(0,1))
      #iterconsts[0]=iterconsts[0]/refloss
      #iterconsts=Mesh.singlerayrndsum(np.array([r,s.ray[i+1]]),iterconsts,s.frequency)
      #i+=1
    #Mesh.bound(bounds)
    #Mesh.plot()
    #return Mesh





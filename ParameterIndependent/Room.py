#!/usr/bin/env python3
# Hayley Wragg 2017-07-18
''' Code to construct the mesh of the room '''

from math import atan2,hypot,sqrt,copysign
from math import sin,cos,atan2,log,isnan
import numpy                as np
from numpy import zeros, load, cross, ones,concatenate, amin,amax,append,vstack
from numpy import sin, cos, tan, arcsin, arccos, allclose,absolute,arange
from numpy import min as nmin
from numpy import max as nmax
from numpy import array as numparr
import reflection           as ref
import intersection         as ins
import linefunctions        as lf
import math                 as ma
from numpy.linalg import norm as leng
import random               as rnd
import Rays                 as ry
import DictionarySparseMatrix as DS
import time                 as t
from itertools import product
import sys
import logging
import pdb
epsilon=sys.float_info.epsilon
xcheck=2
ycheck=4
zcheck=9
dbg=0
if dbg:
  logon=1
else:
  logon=load('Parameters/logon.npy')

class room:
  ''' A room is where the obstacle co-ordinates are contained.

  :param obst: is a Nobx3x[3x1] array, where Nob is the number of \
    obstacles.

  obst[j] is a 3x[3x1] array which is 3, 3D co-ordinates \
  which form a triangle.

  This array of triangles forms the obstacles in the room.

  Attributes of room:
    * s.obst=obst
    * .points[3*j]=obst[j][0]
    * s.maxlength is a 4x1 array initialised as empty. Once assigned \
    this is the maximum length in theroom and in the x, y, and z axis.
    *  s.bounds is a 3x2 array \
    :math:`s.bounds= [ [minx, miny, minz], [maxx,maxy,maxz]]`
    * s.inside_points is an initial empty array. Points which are known \
    to be inside obstacles are added to this array later.
    * s.meshwidth is initialised as zero but is stored once asked for \
    using get_meshwidth.

  '''
  ## Constructor
  # @param obst is a Nobx3x[3x1] array, where Nob is the number of
  # obstacles. obst[j] is a 3x[3x1] array which is 3, 3D co-ordinates
  # which form a triangle.
  # This array of triangles forms the obstacles in the room.
  # - s.obst=obst
  # - \f$ s.points[3*j]=obst[j][0] \f$
  # - s.maxlength is a 4x1 array initialised as empty. Once assigned this
  # is the maximum length in theroom and in the x, y, and z axis.
  # - s.bounds is a 3x2 array
  # \f$ s.bounds= [ [minx, miny, minz], [maxx,maxy,maxz]] \f$
  # - s.inside_points is an initial empty array. Points which are known
  # to be inside obstacles are added to this array later.
  # - s.meshwidth is initialised as zero but is stored once asked for
  # using get_meshwidth.
  def __init__(s,obst,Ntri=0):
    s.obst=obst
    RoomP=obst[0]
    for j in range(1,len(obst)):
      RoomP=concatenate((RoomP,obst[j]),axis=0)
    s.points=RoomP
    # Points is the array of all the co-ordinates which form the surfaces in the room
    s.Nob=len(obst)
    s.norms=zeros((s.Nob,3),dtype=float)
    for j in range(s.Nob):
      s.norms[j]=cross(s.obst[j][0]-s.obst[j][1],s.obst[j][0]-s.obst[j][2])
      s.norms[j]/=leng(s.norms[j])
    if isinstance(Ntri,type(0)):
      s.Ntri=ones(s.Nob,dtype=int)
    else:
      s.Ntri=Ntri.astype(int)
    s.Nsur=len(s.Ntri)
    # Nob is the number of surfaces forming obstacles in the room.
    s.maxlength=zeros(4)
    s.bounds=numparr([nmin(s.points,axis=0),nmax(s.points,axis=0)])
    s.inside_points=numparr([]).astype(float)
    # The inside points line within obstacles and are used to detect if a ray is inside or outside.
    # The time taken for a computation is stored in time.
    s.meshwidth=0.0
    # The length of a cell in the room. Before scaling.
    s.MaxInter=int(s.Nob)
    # The maximum number of intersections a single ray can have in a given direction.
  ## Get the 'ith' obstacle
  # @param i the indiex for the term being returned.
  # @return s.points[i]
  def __get_obst__(s,i):
   ''' Returns the ith surface obstacle of the room s'''
   return s.obst[i]
  ## Get the 'ith' inside point
  # @param i the indiex for the term being returned.
  # @return s.inside_points[i]
  def __get_insidepoint__(s,i):
   ''' Returns the ith inside point of s '''
   return s.inside_points[i]
  ## Get the meshwidth.
  # @param Mesh the Nx*Ny*Nz*na*nb array which will be used to store
  # reflection angles and ray distances.
  # If the meshwidth has not yet been assigned then calculate it using
  # the number of terms in the x direction (Nx) in the mesh and the
  # maximum length in the x direction (s.maxlength[1]).
  # \f$ meshwidth=s.maxlength[1]/Nx \f$
  # If the meshwidth has already been found then it is returned without recalculating.
  # @return s.points[i]
  def get_meshwidth(s,Mesh):
    if abs(s.meshwidth)<epsilon:
      if type(Mesh) is np.ndarray:
        s.meshwidth=s.maxlength[1]/Mesh.shape[0]
      else: s.meshwidth=s.maxlength[1]/Mesh.Nx
    return s.meshwidth
  ## Add a new obst to the room.
  # @param obst0 is added to s.obst
  # @param obst0[0],obst0[1],obst0[2] are added to s.points.
  # @return nothing
  def __set_obst__(s,obst0):
    s.obst+=(obst0,)
    s.points+=obst0
    s.Nob+=1
    s.norms=append(s.norms,cross(obst0[0]-obst0[1],obst0[0]-obst0[2]))
    s.norms[-1]/=leng(s.norms[1])
    return
  def __set_MaxInter__(s,m):
    s.MaxInter=m
    return
  ##  Add the point p to the list of inside points for the room.
  # @param p=[x,y,z]
  # \if inside_points was empty then assign it to be p.
  # \f$s.insidepoints=[[x,y,z]]\f$ \endif
  # \else stack the new point underneath the previous array.
  # \f$ s.insidepoints=[[x0,y0,z0],[x1,y1,z1],...,[xn,yn,zn],[x,y,z]]\f$,
  # \f$s.insidepoints[j]=[xj,yj,zj]\f$ \endelse
  # @return nothing
  def __set_insidepoint__(s,p):
    if len(s.inside_points)<1:
      s.inside_points=numparr([p])
    else:
      s.inside_points=vstack((s.inside_points,p))
    return
  ## The string representation of the room s is the string of the
  # obstacle co-ordinates.
  def __str__(s):
    return 'Room('+str(list(s.obst))+')'
  ## Get the maximum length in the room or axis.
  # @param a the axis or room. a=0 maximum length in room, a=1 for
  # x-axis, a=2 for y-axis a=3 for z-axis.
  # .
  # \par
  # If the maxlength[a] hasn't been found yet find it by comparing the
  # length between points in s.points.
  # @return s.maxlength[a]
  def check_innerpoint(s,p):
    '''Check if the point p is one of the interior points of the room.
    Return True is interior False if not'''
    return any(allclose(p2,p,rtol=epsilon) for p2 in s.inside_points)
  def CheckTxInner(s,Tx):
    direc =numparr([1,0,0])
    direc2=numparr([0,1,0])
    direc3=numparr([0,0,1])
    ray =numparr([Tx,direc])
    ray2=numparr([Tx,direc2])
    ray3=numparr([Tx,direc3])
    count =0
    count2=0
    intercheck =numparr([-1,-1,-1])
    intercheck2=numparr([-1,-1,-1])
    for ob in range(s.Nob):
      inter=ins.intersection(ray,s,ob)
      #print(inter,isnan(inter[0]),intercheck,(inter==intercheck).all(),inter.all()==intercheck.all())
      inter2=ins.intersection(ray2,s,ob)
      if not isnan(inter[0]):
        if not (inter==intercheck).all():
          count+=1
          intercheck=inter
      if not isnan(inter2[0]):
        if not (inter2==intercheck2).all():
          count2+=1
          intercheck2=inter2
    if count%2==count2%2:
      return count%2
    else:
      count3=0
      intercheck3=numparr([-1,-1,-1])
      for ob in range(s.Nob):
        inter3=ins.intersection(ray3,s,ob)
        if not isnan(inter3[0]):
          if not (inter3==intercheck3).all():
            count3+=1
            intercheck3=inter3
      if count%2+count2%2+count3%2==2:
        return 1
      else: return 0
  def maxleng(s,a=0):
    ''' Get the maximum length in the room or axis.

    :param a: the axis or room. a=0 maximum length in room, a=1 for \
    x-axis, a=2 for y-axis a=3 for z-axis.

    If the maxlength[a] hasn't been found yet find it by comparing the \
    between points in s.points.

    :return: s.maxlength[a]

    '''
    # Has the maxlength in the room been found yet? If no compute it.
    if abs(s.maxlength[a])<epsilon:
      leng1=0
      if a==0:
        for p1,p2 in product(s.points,s.points):
          leng2=leng(p1-p2)
          if leng2>leng1:
            s.maxlength[a]=leng2
            leng1=leng2
      else:
        s.maxlength[a]=s.bounds[1][a-1]-s.bounds[0][a-1]
      return s.maxlength[a]
    else: return s.maxlength[a]
  ## maxxleng this function is no longer needed but won't be removed
  # until all programs calling it have been corrected.
  # @return s.maxleng(1)
  def maxxleng(s):
    return s.maxleng(1)
  ## maxyleng this function is no longer needed but won't be removed
  # until all programs calling it have been corrected.
  # @return s.maxleng(2)
  def maxyleng(s):
    return s.maxleng(2)
  ## maxzleng this function is no longer needed but won't be removed
  # until all programs calling it have been corrected.
  # @return s.maxlength(3)
  def maxzleng(s):
    return s.maxleng(3)
  ## Find the indexing position in a mesh with width h for point p
  #  lying in the room s.
  # @param p =[x,y,z] the co-ordinate of the point p or an array of
  # points p=[[x0,y0,z0],...,[xn,yn,zn]]
  # @param h is the meshwidth, once assigned this matches s.meshwidth
  # If p is one point, out=(p-[minx,miny,minz])//h,
  # If p is an array of points,
  # out=[(p0-[minx,miny,minz])//h,...,(pn-[minx,miny,minz])//h]
  # @return out
  def position(s,p,h):
    ''' Find the indexing position in a mesh with width h for point p \
    lying in the room s.

    :param p: =[x,y,z] the co-ordinate of the point p or an array \
    of :math:`points p=[[x0,y0,z0],...,[xn,yn,zn]]`
    :param h: is the meshwidth, once assigned this matches s.meshwidth

    If p is one point,

    .. code::

       out=(p-[minx,miny,minz])//h,

    If p is an array of points,

    .. code::

       out=[(p0-[minx,miny,minz])//h,...,(pn-[minx,miny,minz])//h]

    :return: out

    '''
    if isinstance(p[0],float): n=1
    elif isinstance(p[0],int): n=1
    else:
      n=len(p)
    if n==1:
      i,j,k=(p-s.bounds[0])//h
      return int(i),int(j),int(k)
    elif n>1:
      positions=numparr([(px-s.bounds[0])//h for px in p]).astype(int)#np.array((p-np.tile(s.bounds[0],(n,1)))//h,dtype=int)
      return positions
    else:
      raise ValueError("Neither point nor array of points")
  ## Find the co-ordinate of the point at the centre of the element.
  # @param h the meshwdith. Once assigned this matches s.meshwidth
  # @param i the first index or an array corresponding to the first
  # index for multiple points.
  # @param j the second index or an array corresponding to the second
  # index for multiple points.
  # @param k the third index or an array corresponding to the third
  # index for multiple points.
  # .
  # \par
  # \if there is only 1 i, 1 j, and 1 k,
  # \f$ p=[minx,miny,minz] +h*[i+0.5,j+0.5,k+0.5] \f$ \endif
  # \elseif there's arrays for i,j, and k,
  # \f$ p=[[minx,miny,minz] +h*[i0+0.5,j0+0.5,k0+0.5],...,
  # [minx,miny,minz] +h*[in+0.5,jn+0.5,kn+0.5 ]\f$ \endelseif
  # @return p
  def surf_from_ob(s,nob):
    nsur=1
    for j in range(s.Nsur):
      if nob-sum(s.Ntri[0:j])>0:
        nsur=int(j+1)
    return nsur
  def nob_from_sur(s,nsur):
    return sum(s.Ntri[0:nsur])
  def coordinate(s,h,i,j,k):
    ''' Find the co-ordinate of the point at the centre of the element.

    :param h: the meshwdith. Once assigned this matches s.meshwidth

    :param i: the first index or an array corresponding to the first \
    index for multiple points.

    :param j: the second index or an array corresponding to the second \
    index for multiple points.

    :param k: the third index or an array corresponding to the third \
    index for multiple points.

    If there is only 1 i, 1 j, and 1 k,

    .. code::

       p=[minx,miny,minz] +h*[i+0.5,j+0.5,k+0.5]

    ElseIf there's arrays for i,j, and k,

    .. code::

       p=[[minx,miny,minz] +h*[i0+0.5,j0+0.5,k0+0.5],...,
         [minx,miny,minz] +h*[in+0.5,jn+0.5,kn+0.5 ]

    :return: p

    '''
    if isinstance(i,(float,int,np.complex128,np.int64)): n=1
    else:
      n=len(i)
    if n==1:
      coord=numparr([s.bounds[0]+h*numparr([i+0.5,j+0.5,k+0.5])])
      return coord
    elif n>1:
      Addarray=s.bounds[0]+h*numparr([0.5,0.5,0.5])#np.tile(s.bounds[0]+h*np.array([0.5,0.5,0.5]),(n,1))
      coord=numparr([h*numparr([ix,jx,kx])+Addarray for (ix,jx,kx) in zip(i,j,k)])#np.array((h*np.c_[i,j,k]+Addarray),dtype=float)
      #coord=coord.T
      return coord
    else:
      raise ValueError("Neither point nor array of points")
 ## Traces ray's uniformly emitted from an origin around a room.
 # @param Tx the co-ordinate of the transmitter location
 # @param Nra Number of rays
 # @param Nre number of reflections
 # @param directions Nra*3 array of the initial direction for each ray.
 # @param Mesh a Nx*Ny*Nz*na*nb array (actually a dictionary of sparse
 # matrices using class DS but built to have similar structure to an array).
 # na=int(Nsur*Nre+1), nb=int((Nre)*(Nra)+1)
 # .
 # \par The rays are reflected Nre times with the obstacles s.obst. The
 # points of intersection with the obstacles are stored in raylist. This
 # is done using the mesh_multiref function.
 # \par As each intersection is found the mesh_multiref function
 # forms the line segment between intersection points and the
 # corresponding ray cone. All mesh elements in the ray cone store the
 # reflection angles and the distance along the ray cone from the source
 # to the centre of each mesh element. This is stored in Mesh.
 # See function mesh_multiref for more details on the reflections and
 # storage.
 # @return raylist, Mesh
  def ray_mesh_bounce(s,Tx,directions,Mesh,programterms):
    ''' Traces ray's uniformly emitted from an origin around a room.

    :param Tx: the co-ordinate of the transmitter location
    :param Nra: Number of rays
    :param Nre: number of reflections
    :param directions: Nra*3 array of the initial direction for \
    each ray.
    :param Mesh: a Nx*Ny*Nz*na*nb array (actually a dictionary of \
    sparse matrices using class DS but built to have similar \
    structure to an array).

    .. code::

       na=int(Nsur*Nre+1), nb=int((Nre)*(Nra)+1)

    The rays are reflected Nre times with the obstacles s.obst. \
    The points of intersection with the obstacles are stored in z
    raylist. This is done using the mesh_multiref function.
    As each intersection is found the mesh_multiref function \
    forms the line segment between intersection points and the \
    corresponding ray cone. All mesh elements in the ray cone store \
    the reflection angles and the distance along the ray cone from the \
    source to the centre of each mesh element. This is stored in Mesh.
    See :py:func:`Rays.mesh_multiref` for more details on the \
    reflections and storage.

    :return: raylist, Mesh

    '''
    Nra,Nre       =programterms[0:2].astype(int)
    r             =s.maxleng()
    raylist       =zeros([Nra+1, Nre+1,4])
    directions    =r*directions
    # Iterate through the rays find the ray reflections
    # FIXME rays are independent of each other so this is parallelisable
    #j=int(Nra/2)
    start     =append(Tx,[0])
    for it in range(0,Nra):
      raystart  =ry.Ray(start, directions[it])
      Mesh=raystart.mesh_multiref(s,Mesh,it,programterms)
      raylist[it]=raystart.points[0:-2]
    if dbg:
      assert Mesh.check_nonzero_col(Nre,s.Nsur)
      #logging.error('There is a column with too many terms')
      #raise ValueError('There is a column with too many terms')
    #logging.info('Raypoints')
    #logging.info(str(raylist))
    return raylist, Mesh
  def ray_mesh_power_bounce(s,Tx,directions,Grid,Znobrat,refindex,Antpar,Gt,Pol,programterms,loghandle=str()):
    ''' Traces ray's uniformly emitted from an origin around a room.

    :param Tx: the co-ordinate of the transmitter location
    :param Nra: Number of rays
    :param Nre: number of reflections
    :param directions: Nra*3 array of the initial direction for \
    each ray.
    :param Grid: a Nx*Ny*Nz array which will contain power values
    :param Znobrat: The array with the ratio of the impedance of an \
    obstacle over the impedance of air.
    :param refindex: Array with the refractive indices of an obstacle.
    :param Antpar: array with antenna parameters - scaled wavenumber, wavelength, lengthscale.
    :param Gt: transmitter gains.
    :param Pol: polarisation of the antenna, first term indicates the strength in ther perp direction, second in the parallel.
    :param deltheta: The maximum angle spacing between neighbouring rays.
    :param loghandle: handle for logging information during run time.


    The rays are reflected Nre times with the obstacles s.obst. \
    The points of intersection with the obstacles are stored in z
    raylist. This is done using the mesh_multiref function.
    As each intersection is found the mesh_multiref function \
    forms the line segment between intersection points and the \
    corresponding ray cone. All mesh elements in the ray cone store \
    the power. This is stored in Grid.
    See :py:func:`Rays.mesh_multiref` for more details on the \
    reflections and storage.

    :return: raylist, Grid

    '''
    Nra,Nre=programterms[0:2].astype(int)
    #start_time    =t.time()         # Start the time counter
    raylist       =zeros([Nra+1, Nre+1,4]) # Initialise the ray reflection points
    # Iterate through the rays find the ray reflections
    for it in range(Nra):
      Dir       =directions[it]
      start     =append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      Grid=raystart.mesh_power_multiref(s,Grid,it,Znobrat,refindex,Antpar,Pol,programterms,loghandle)
      raylist[it]=raystart.points[0:-2]
    Nx=Grid.shape[0]
    Ny=Grid.shape[1]
    Nz=Grid.shape[2]
    P=zeros((Nx,Ny,Nz),dtype=np.longdouble)
    P=absolute(Grid[:,:,:,0])**2+absolute(Grid[:,:,:,1])**2
    P=DS.Watts_to_db(P)
    return raylist, P
  def ray_bounce(s,Tx,Nre,Nra,directions):
    ''' Trace ray's uniformly emitted from an origin around a room.

    :param Nra: Number of rays
    :param Nre: number of reflections Nre
    :param directions: A Nra*3 array of the initial directions \
    for each ray.

    The multiref function is used to find the Nre reflections for \
    the Nra rays with the obstacles s.obst.

    :math:`raylist=[[p00,p01,...,p0Nre],[p10,...,p1Nre],...,[pNra0,...,pNraNre]]`

    :rtype: An array of the ray points.

    :return: raylist

    '''
    start_time    =t.time()         # Start the time counter
    r             =s.maxleng()
    directions    =r*directions
    raylist       =empty([Nra+1, Nre+1,4])
    # FIXME the rays are independent of each toher so this is easily parallelisable
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      raystart.multiref(s,Nre)
      raylist[it]=raystart.points[0:-2]
    return raylist
  ## Takes in n array of obsts and adds to the room
  # @param obsts=[obst0,obst1,...,obstn] the array of obstacles
  # .
  # \par The obstacles are add to s.obst and points are added to
  # s.points using add_obst.
  # \par The number of obstacles Nob is increased by n.
  # @return nothing
  def roomobstadd(s,obsts):
    for obst1 in obsts[1:]:
      s.add_obst(obst1)
      s.Nob+=1
    return
  def nsurffromnob(s,Nob):
    ttot=0
    nsur=0
    for t in s.Ntri:
      ttot+=t
      if ttot<Nob:
        nsur+=1
      else:
        break
    return nsur


def FindInnerPoints(Room,Mesh):
    h=Room.get_meshwidth(Mesh)
    Nx=Mesh.Nx
    Ny=Mesh.Ny
    Nz=Mesh.Nz
    CentreX=numparr([h*(Nx*0.5)+h*0.5,-h*0.5,h*(Nz*0.5)+h*0.5])
    CentreY=numparr([-h*0.5,h*(Ny*0.5)+h*0.5,h*(Nz*0.5)+h*0.5])
    CentreZ=numparr([h*(Nz*0.5)+h*0.5,h*(Nz*0.5)+h*0.5,h*(Nz*0.5)+h*0.5])
    for i,j,k in product(range(Nx),range(Ny),range(Nz)):
      p=Room.coordinate(h,i,j,k)[0]
      #p=numparr([x,y,z])
      d1=CentreX-p
      d2=CentreY-p
      d3=CentreZ-p
      ray1=numparr([p,d1])
      rayleng1=leng([d1])
      ray2=numparr([p,d2])
      rayleng2=leng([d2])
      ray3=numparr([p,d3])
      rayleng3=leng([d3])
      count1=0
      count2=0
      count3=0
      surfacenumbers1=numparr([])
      surfacenumbers2=numparr([])
      surfacenumbers3=numparr([])
      for ob in range(Room.Nob):
        Tri=Room.obst[ob]
        inter1=ins.intersection(ray1,Room,ob)
        inter2=ins.intersection(ray2,Room,ob)
        inter3=ins.intersection(ray3,Room,ob)
        nsur=DS.Correct_ObNumbers(numparr([ob]),Room.Ntri)[0]
        if ins.InsideCheck(inter1,Tri) and not isnan(inter1[0]):
          repeatpoint1=0
          if len(surfacenumbers1)!=0:
            surcheck=(abs(nsur-surfacenumbers1)<epsilon).any()
            if surcheck:
              repeatpoint1=1
          if leng(inter1-p)<rayleng1 and not repeatpoint1:
            count1+=1
            surfacenumbers1=append(surfacenumbers1,nsur)
        if ins.InsideCheck(inter2,Tri) and not isnan(inter2[0]):
          repeatpoint2=0
          if len(surfacenumbers2)!=0:
            surcheck=(abs(nsur-surfacenumbers2)<epsilon).any()
            if surcheck:
              repeatpoint2=1
          if leng(inter2-p)<rayleng2 and not repeatpoint2:
            count2+=1
            surfacenumbers2=append(surfacenumbers2,nsur)
        if ins.InsideCheck(inter3,Tri) and not isnan(inter3[0]):
          repeatpoint3=0
          if len(surfacenumbers3)!=0:
            surcheck=(abs(nsur-surfacenumbers3)<epsilon).any()
            if surcheck:
              repeatpoint3=1
          if leng(inter3-p)<rayleng3 and not repeatpoint3:
            count3+=1
            surfacenumbers3=append(surfacenumbers3,nsur)
      if count1%2+count2%2+count3%2<2:
        Room.__set_insidepoint__(p)
    return 0

def TestTxCheck():
  ##----The lengths are non-dimensionalised---------------------------
  OuterBoundary =load('Parameters/OuterBoundary.npy').astype(float)  # The Obstacles forming the outer boundary of the room
  NtriOb        =load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Oblist=OuterBoundary

  Room=room(Oblist,NtriOb)
  Nob=Room.Nob
  Nsur=Room.Nsur
  for x,y,z in product(arange(0,1,0.1),arange(0,1,0.1),arange(0,1,0.1)):
    Tx=numparr([x,y,z])
    print('Tx',Tx)
    print(Room.CheckTxInner(Tx))
  return 0

if __name__=='__main__':
  TestTxCheck()
  exit()


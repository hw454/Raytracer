#!/usr/bin/env python3
# Hayley Wragg 2018-05-29
import numpy as np
import vtk
import sys
import os
import math as ma
import Room as rom
import Rays as ra
from mayavi.core.api import Engine
from mayavi.sources.vtk_file_reader import VTKFileReader
from mayavi.modules.surface import Surface
from mayavi import mlab
from pyface.api import GUI
import scipy.sparse.linalg
from itertools import product
import matplotlib.pyplot as mp
import num2words as nw

epsilon=sys.float_info.epsilon

def PlotRays(plottype=str()):
    ##Plot the obstacles and the room first

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L    =np.load('Parameters/Raytracing.npy')[0:3]
    Nre=int(Nre)

    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')
    Tx            =np.load('Parameters/Origin.npy')
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')
    LOS           =np.load('Parameters/LOS.npy')
    PerfRef       =np.load('Parameters/PerfRef.npy')
    Nsur          =np.load('Parameters/Nsur.npy')
    Nrs           =np.load('Parameters/Nrs.npy')
    InnerOb       =np.load('Parameters/InnerOb.npy')
    numjobs       =np.load('Parameters/Numjobs.npy')
    numjobs=500
    Nob           =len(Oblist)
    #Nob2          =len(OuterBoundary)
    Room          =Oblist#=np.concatenate((Oblist,OuterBoundary),axis=0)
    Oblist        =Room
    RoomP=Oblist[0]
    if LOS:
      LOSstr='LOS'
    else:
      if Nre>2:
        if Nrs<Nsur:
          LOSstr=nw.num2words(Nrs)+''
        else:
          LOSstr='Multi'
      else:
        LOSstr='Single'
    if InnerOb:
      Box='Box'
    else:
      Box='NoBox'
    foldtype=LOSstr+Box
    for j in range(1,Nob):
      RoomP=np.concatenate((RoomP,Oblist[j]),axis=0)
    for j in range(0,RoomP.shape[0],3):
      mlab.figure(0)
      x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
      y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
      z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
      mlab.plot3d(x,y,z,tube_radius=0.01)
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
        os.makedirs('./ConeFigures/'+Box)
      if not os.path.exists('./ConeFigures/'+Box):
        os.makedirs('./ConeFigures/'+Box)
    mlab.savefig('ConeFigures/'+Box+'/Room.jpg',size=(1000,1000))
    mlab.clf()
    mlab.close(all=True)
    for job in range(numjobs):
      Txstr='Parameters/Origin_job%03d.npy'%job
      if os.path.isfile(Txstr):
        Tx=np.load(Txstr)
        if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
          loca='Centre'
        else:
          loca='OffCentre'
        foldtype=LOSstr+Box+loca
      else:
        continue
      for i,nre in product(range(nra),range(Nre+1)):
        ##---Retrieve the Ray points ---------------------------------------
        RayPstr='./Mesh/'+foldtype+'/RayMeshPoints%03dRefs%03d_tx%03d.npy'%(Nra[i],nre,job)
        if os.path.isfile(RayPstr):
          print('Plotting ray points')
          print(RayPstr)
          data_matrix=np.load(RayPstr)
          mlab.clf()
          mlab.close(all=True)
          mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.25)
        else:
          continue
        for j in range(0,int(Nra[i])):
          for l in range(0,int(nre)+1):
            if not any(ma.isnan(d) for d in data_matrix[j][l]):
              if l==0:
                x=np.array([data_matrix[j][0][0]])
                y=np.array([data_matrix[j][0][1]])
                z=np.array([data_matrix[j][0][2]])
              else:
                xp=np.array([data_matrix[j][l][0]])
                yp=np.array([data_matrix[j][l][1]])
                zp=np.array([data_matrix[j][l][2]])
                x=np.append(x,xp)
                y=np.append(y,yp)
                z=np.append(z,zp)
          mlab.plot3d(x,y,z,color= (0, 1, 1),opacity=0.5)
        for k in range(0,RoomP.shape[0],3):
          x=np.array([RoomP[k][0],RoomP[k+1][0],RoomP[k+2][0],RoomP[k][0]])
          y=np.array([RoomP[k][1],RoomP[k+1][1],RoomP[k+2][1],RoomP[k][1]])
          z=np.array([RoomP[k][2],RoomP[k+1][2],RoomP[k+2][2],RoomP[k][2]])
          mlab.plot3d(x,y,z,opacity=0.5)
        if not os.path.exists('./ConeFigures'):
          os.makedirs('./ConeFigures')
          os.makedirs('./ConeFigures/'+foldtype)
        if not os.path.exists('./ConeFigures/'+foldtype):
          os.makedirs('./ConeFigures/'+foldtype)
        mlab.savefig('ConeFigures/'+foldtype+'/Room%03dRefs%03d_tx%03d.jpg'%(Nra[i],nre,job),size=(1000,1000))
        print('ConeFigures/'+foldtype+'/Room%03dRefs%03d_tx%03d.jpg'%(Nra[i],nre,job))
        mlab.clf()
        mlab.close(all=True)
    return

def PlotCones(plottype):
    '''Plot the cone calculations.'''

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')
    # Take Tx to be 0,0,0
    delangle     =np.load('Parameters/delangle.npy')
    mulfac=1

    for i in range(0,nra):
      directionname='Parameters/Directions%d.npy'%i
      data_matrix   =np.load(directionname)         # Matrix of ray directions
      mlab.points3d(0,0,0,scale_factor=0.1)
      zsteps=int(np.pi/delangle[i])
      nre=0
      dist=1 #L
      Ncon=ra.no_cones(h,dist,delangle[i],0,nre)
      anglevec=np.linspace(0.0,2*ma.pi,num=int(Ncon), endpoint=False) # Create an array of all the angles
      Norm=np.zeros((Ncon,3),dtype=np.float) # Initialise the matrix of normals
      Cones=np.zeros((int(Nra[i])*Ncon,2,3))
      for j in range(0,int(Nra[i])):#int(Nrao)):
        cx,cy,cz,cs=mulfac*data_matrix[j]/np.linalg.norm(data_matrix[j])
        d=data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3])                  # Normalise the direction of the ray
        if abs(d[2])>0 and Ncon>0:
         Norm[0]=np.array([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
         Norm[0]/=np.linalg.norm(Norm[0]) # Normalise the vector
         yax=np.cross(Norm[0],d)            # Compute another vector in the plane for the axis.
         yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        elif Ncon>0:
         Norm[0]=np.array([0,0,1])        # If d_z is 0 then this vector is always in the plane.
         yax=np.cross(Norm[0],d)            # Compute another vector in the plane to form co-ordinate axis.
         yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        if Ncon>0:
          Norm=np.outer(np.cos(anglevec),Norm[0])+np.outer(np.sin(anglevec),yax) # Use the outer product to multiple the axis
        for k in range(0,Ncon):
          Cones[j*Ncon+k][0]=np.array([cx,cy,cz])
          delth=ra.angle_space(delangle[i],nre)
          beta=ra.beta_leng(dist,delth,0)
          Cones[j*Ncon+k][1]=(mulfac*data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3]))+beta*Norm[k]
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
        os.makedirs('./ConeFigures/'+plottype)
      if not os.path.exists('./ConeFigures/'+plottype):
        os.makedirs('./ConeFigures/'+plottype)
      for j in range(0,int(Nra[i])):
        x=0
        y=0
        z=0
        x2,y2,z2=mulfac*data_matrix[j][0:3]
        x=np.append(x,[x2])
        y=np.append(y,[y2])
        z=np.append(z,[z2])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
      mlab.savefig('ConeFigures/'+plottype+'/Rays%d.jpg'%Nra[i],size=(1000,1000))
      N=int(zsteps)
      count=0
      for l in range(0,N):
        if count>(Nra[i]*Ncon):
          break
        #j=int(Nre)*k
        if l==0 or l==N-1:
          for k in range(0, int(Ncon)):
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xp=np.append(x,[Cones[j][1][0]])
            yp=np.append(y,[Cones[j][1][1]])
            zp=np.append(z,[Cones[j][1][2]])
            mlab.plot3d(xp,yp,zp,color= (1,0,1))
            count+=1
        else:
          j=count
          c=np.arcsin(Cones[j][0][2])/delangle[i]
          if c>int(zsteps/2):
            c=zsteps-c
          mid=(np.cos(delangle[i])-np.sin((c)*delangle[i])**2)/(np.cos(delangle[i]*(c))**2)
          if abs(mid)>1:
            xyk=0
            count+=1
          else:
            bot=ma.acos(mid)
            xyk=int(2*np.pi/bot)
            if xyk<=1:
              xyk=1
          for k in range(0,int(xyk*Ncon)):
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xp=np.append(x,[Cones[j][1][0]])
            yp=np.append(y,[Cones[j][1][1]])
            zp=np.append(z,[Cones[j][1][2]])
            mlab.plot3d(xp,yp,zp,color= (1,0,1))
            count+=1
      filename='ConeFigures/'+plottype+'/Cone%d.jpg'%Nra[i]
      mlab.savefig(filename,size=(1000,1000))
      mlab.clf()
      mlab.close(all=True)
    #mlab.show()
    return

def PlotConesGains(plottype):
    '''Plot the cone calculations.'''

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
    # Take Tx to be 0,0,0
    delangle     =np.load('Parameters/delangle.npy')
    mulfac=1
    cophi=0
    siphi=0
    for i in range(0,nra):
      directionname='Parameters/Directions%03d.npy'%Nra[i]
      data_matrix   =np.load(directionname)         # Matrix of ray directions
      ray_matrix=np.zeros((Nra[i],3))
      mlab.points3d(0,0,0,scale_factor=0.1)
      zsteps=int(np.pi/delangle[i])
      nre=0
      dist=1 #L
      Ncon=ra.no_cones(h,dist,delangle[i],0,nre)
      anglevec=np.linspace(0.0,2*ma.pi,num=int(Ncon), endpoint=False) # Create an array of all the angles
      Norm=np.zeros((Ncon,3),dtype=np.float) # Initialise the matrix of normals
      Cones=np.zeros((int(Nra[i])*Ncon,2,3))
      for j in range(0,int(Nra[i])):#int(Nrao)):
        d=data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3])
        r=np.linalg.norm(d)
        if d[0]!=0:
          theta=np.arctan(d[1]/d[0])
        else:
          theta=np.pi/2
        if d[0]<0 and d[1]>0:
          theta=np.pi-abs(theta)
        elif d[0]<0 and d[1]<0:
          theta=np.pi+abs(theta)
        elif d[0]>0 and d[1]<0:
          theta=2*np.pi-abs(theta)
        phi=np.arccos(d[2]/r)
        if np.pi/2 <= theta <= 3*np.pi/2:
          cothe=np.cos(theta)
          sithe=np.sin(theta)
        else:
          cothe=0.5*np.cos(5*theta)
          sithe=0.5*np.sin(5*theta)
        if np.pi/4 <= phi <= 3*np.pi/4:
          cophi=np.cos(2*phi+np.pi/2)
          siphi=np.sin(2*phi+np.pi/2)
        #print(mulfac,theta,phi,data_matrix[j])
        #cx,cy,cz,cs=mulfac*data_matrix[j]/np.linalg.norm(data_matrix[j])
        mulfac=(2*(cothe**2)+siphi**2)**0.5
        ray_matrix[j]=mulfac*d
        cx,cy,cz=ray_matrix[j][0:3]
        #d=data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3])                  # Normalise the direction of the ray
        if abs(d[2])>0 and Ncon>0:
         Norm[0]=np.array([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
         Norm[0]/=np.linalg.norm(Norm[0]) # Normalise the vector
         yax=np.cross(Norm[0],d)            # Compute another vector in the plane for the axis.
         yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        elif Ncon>0:
         Norm[0]=np.array([0,0,1])        # If d_z is 0 then this vector is always in the plane.
         yax=np.cross(Norm[0],d)            # Compute another vector in the plane to form co-ordinate axis.
         yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        if Ncon>0:
          Norm=np.outer(np.cos(anglevec),Norm[0])+np.outer(np.sin(anglevec),yax) # Use the outer product to multiple the axis
        delth=ra.angle_space(delangle[i],nre)
        beta=ra.beta_leng(dist*mulfac,delth,0)
        #print(beta,j)
        for k in range(0,Ncon):
          Cones[j*Ncon+k][0]=np.array([cx,cy,cz])
          Cones[j*Ncon+k][1]=(mulfac*data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3]))+beta*Norm[k]
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
        os.makedirs('./ConeFigures/'+plottype)
      if not os.path.exists('./ConeFigures/'+plottype):
        os.makedirs('./ConeFigures/'+plottype)
      for j in range(0,int(Nra[i])):
        x=0
        y=0
        z=0
        x2,y2,z2=ray_matrix[j][0:3]
        x=np.append(x,[x2])
        y=np.append(y,[y2])
        z=np.append(z,[z2])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
      mlab.savefig('ConeFigures/'+plottype+'/RaysGains%d.jpg'%Nra[i],size=(1000,1000))
      N=int(zsteps)
      count=0
      for j in range(0,Nra[i]*Ncon):
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xp=np.append(x,[Cones[j][1][0]])
            yp=np.append(y,[Cones[j][1][1]])
            zp=np.append(z,[Cones[j][1][2]])
            mlab.plot3d(xp,yp,zp,color= (1,0,1))
            xp=np.append(0,[Cones[j][1][0]])
            yp=np.append(0,[Cones[j][1][1]])
            zp=np.append(0,[Cones[j][1][2]])
            mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.25)
            #count+=1
      mlab.show()
      filename='ConeFigures/'+plottype+'/ConesWithGains%03d.jpg'%Nra[i]
      mlab.savefig(filename,size=(1000,1000))
      #mlab.clf()
      #mlab.close(all=True)
    #mlab.show()
    return

def PlotPolarTheoryGains(plottype):
    '''Plot the cone calculations.'''
    mulfac=1
    steps=1000
    theta_matrix=np.linspace(0,2*np.pi,steps)
    phi_matrix  =np.linspace(0,np.pi,steps)
    polarhoz_matrix=np.zeros((steps,2))
    polarvert_matrix=np.zeros((steps,2))
    for j in range(0,steps):#int(Nrao)):
      theta=theta_matrix[j]
      phi  =phi_matrix[j]
      if np.pi/2 <= theta <= 3*np.pi/2:
        cothe=np.cos(theta)
        sithe=np.sin(theta)
      #elif 2*np.pi/3 < theta <= np.pi :
       # mulfacthe=0.25+0.25*(np.pi-theta)*3/np.pi
      #elif 0 <= theta < np.pi/3:
       # mulfacthe=0.25+0.25*(theta)*3/np.pi
      #elif 5*np.pi/3 < theta <= 2*np.pi :
       # mulfacthe=0.25*(2*np.pi-theta)/np.pi
      else:
        cothe=0.5*np.cos(5*theta)
        sithe=0.5*np.sin(5*theta)
      if np.pi/4 <= phi <= 3*np.pi/4:
        cophi=np.cos(2*phi+np.pi/2)
        siphi=np.sin(2*phi+np.pi/2)
      #elif 0 <= phi < np.pi/4:
       # mulfacphi=0.25*( phi)*4/np.pi
      #elif 3*np.pi/4 < phi <= np.pi:
      #  mulfacphi=0.25*(np.pi-phi)*4/np.pi
      else:
        cophi=0.5*np.cos(12*phi)
        siphi=0.5*np.sin(12*phi)
      radhoz=abs(cothe)
      radver=abs(siphi)
      polarhoz_matrix[j]=theta,radhoz
      polarvert_matrix[j]=phi,radver
    if not os.path.exists('./ConeFigures'):
      os.makedirs('./ConeFigures')
    mp.figure(1)
    #print(polarhoz_matrix)
    #polarhoz_matrix=np.sort(polarhoz_matrix,1)
    mp.polar(polarhoz_matrix[:,0],polarhoz_matrix[:,1])
    filename='ConeFigures/GainsPatternHozTheory.jpg'
    mp.title('Angle to the horizontal plane and antenna gain')
    mp.savefig(filename)
    mp.figure(2)
    mp.polar(polarvert_matrix[:,0],polarvert_matrix[:,1])
    filename='ConeFigures/GainsPatternVertTheory.jpg'
    mp.title('Angle to the z-axis and antenna gain')
    mp.savefig(filename)
    mp.show()
    mp.clf()
    return

def PlotPolarGains(plottype):
    '''Plot the cone calculations.'''

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
    # Take Tx to be 0,0,0
    delangle     =np.load('Parameters/delangle.npy')
    mulfac=1
    cothe=0
    sithe=0
    cophi=0
    siphi=0
    for i in range(0,nra):
      directionname='Parameters/Directions%03d.npy'%Nra[i]
      data_matrix   =np.load(directionname)         # Matrix of ray directions
      nre=0
      dist=1 #L
      Ncon=ra.no_cones(h,dist,delangle[i],0,nre)
      anglevec=np.linspace(0.0,2*ma.pi,num=int(Ncon), endpoint=False) # Create an array of all the angles
      Norm=np.zeros((Ncon,3),dtype=np.float) # Initialise the matrix of normals
      Cones=np.zeros((int(Nra[i])*Ncon,2,3))
      radhoz_matrix=np.zeros((Nra[i]*(Ncon+1),1))
      radvert_matrix=np.zeros((Nra[i]*(Ncon+1),1))
      theta_matrix=np.zeros((Nra[i]*(Ncon+1),1))
      phi_matrix=np.zeros((Nra[i]*(Ncon+1),1))
      polarhoz_matrix=np.zeros((Nra[i]*(Ncon+1),2))
      polarvert_matrix=np.zeros((Nra[i]*(Ncon+1),2))
      for j in range(0,int(Nra[i])):#int(Nrao)):
        d=data_matrix[j]
        r=np.linalg.norm(d)
        if d[0]!=0:
          theta=np.arctan(d[1]/d[0])
        else:
          theta=np.pi/2
        if d[0]<0 and d[1]>0:
          theta=np.pi-abs(theta)
        elif d[0]<0 and d[1]<0:
          theta=np.pi+abs(theta)
        elif d[0]>0 and d[1]<0:
          theta=2*np.pi-abs(theta)
        phi=np.arccos(d[2]/r)
        if np.pi/2 <= theta <= 3*np.pi/2:
          cothe=np.cos(theta)
          sithe=np.sin(theta)
        else:
          cothe=0.5*np.cos(5*theta)
          sithe=0.5*np.sin(5*theta)
        if np.pi/4 <= phi <= 3*np.pi/4:
          cophi=np.cos(2*phi+np.pi/2)
          siphi=np.sin(2*phi+np.pi/2)
        cx,cy,cz,cs=data_matrix[j]/np.linalg.norm(data_matrix[j])
        #cx*=cophi*cothe
        #cy*=cophi*sithe
        #cz*=sithe
        theta_matrix[j*Ncon]=theta
        phi_matrix[j*Ncon]=phi
        radhoz_matrix[j*Ncon]=abs(cothe)
        radvert_matrix[j*Ncon]=abs(siphi)
        polarhoz_matrix[j*Ncon] =theta_matrix[j*Ncon],radhoz_matrix[j*Ncon]
        polarvert_matrix[j*Ncon]=phi_matrix[j*Ncon]  ,radvert_matrix[j*Ncon]
        d=data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3])                  # Normalise the direction of the ray
        if abs(d[2])>0 and Ncon>0:
         Norm[0]=np.array([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
         Norm[0]/=np.linalg.norm(Norm[0]) # Normalise the vector
         yax=np.cross(Norm[0],d)            # Compute another vector in the plane for the axis.
         yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        elif Ncon>0:
         Norm[0]=np.array([0,0,1])        # If d_z is 0 then this vector is always in the plane.
         yax=np.cross(Norm[0],d)            # Compute another vector in the plane to form co-ordinate axis.
         yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        if Ncon>0:
          Norm=np.outer(np.cos(anglevec),Norm[0])+np.outer(np.sin(anglevec),yax) # Use the outer product to multiple the axis
        delth=ra.angle_space(delangle[i],nre)
        beta=ra.beta_leng(dist*mulfac,delth,0)
        #print(beta,j)
        for k in range(1,Ncon+1):
          Cones[j*Ncon+k-1][0]=np.array([cx,cy,cz])
          coneout=(mulfac*data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3]))+beta*Norm[k-1]
          Cones[j*Ncon+k-1][1]=coneout
          r=np.linalg.norm(coneout)
          if coneout[0]!=0:
            theta=np.arctan(coneout[1]/coneout[0])
          else:
            theta=np.pi/2
          if coneout[0]<0 and coneout[1]>0:
            theta=np.pi-abs(theta)
          elif coneout[0]<0 and coneout[1]<0:
            theta=np.pi+abs(theta)
          elif coneout[0]>0 and coneout[1]<0:
            theta=2*np.pi-abs(theta)
          phi=np.arccos(coneout[2]/r)
          theta_matrix[j*Ncon+k]=theta
          phi_matrix[j*Ncon+k]=phi
          print(theta,abs(cothe))
          polarhoz_matrix[j*Ncon+k] =theta,abs(cothe)
          polarvert_matrix[j*Ncon+k]=phi  ,abs(siphi)
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
        os.makedirs('./ConeFigures/'+plottype)
      if not os.path.exists('./ConeFigures/'+plottype):
        os.makedirs('./ConeFigures/'+plottype)
      mp.figure(2*i)
      #polarhoz_matrix=np.sort(polarhoz_matrix,0)
      print(polarhoz_matrix)
      mp.polar(polarhoz_matrix[:,0],polarhoz_matrix[:,1],'+')
      filename='ConeFigures/'+plottype+'/GainsPatternHoz%03d.jpg'%Nra[i]
      mp.title('Horiztonal angle and antenna gain')
      mp.savefig(filename)
      mp.figure(2*i+1)
      #polarvert_matrix=np.sort(polarvert_matrix,0)
      #print(polarhoz_matrix)
      mp.polar(polarvert_matrix[:,0],polarvert_matrix[:,1],'+')
      filename='ConeFigures/'+plottype+'/GainsPatternVert%03d.jpg'%Nra[i]
      mp.title('Vertical angle and antenna gain')
      mp.savefig(filename)
    mp.show()
    mp.clf()
    return


def PlotConesOnSquare(plottype):
    '''Plot the cone calculations.'''
    ConeOn=0 # Plot Cones and Rays if 1
    Cut=2    # Plot x,y plane cuts
    Vid=0    # Rotate for video
    cmaptopt=str('plasma')
    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
    # Take Tx to be 0,0,0
    Tx            =np.load('Parameters/Origin.npy')
    delangle      =np.load('Parameters/delangle.npy')
    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
    Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

    # Room contains all the obstacles and walls.
    Room=rom.room(Oblist)

    Nz=int(Room.maxyleng()/(h))
    Nx=int(Room.maxxleng()/(h))
    Ny=int(Room.maxyleng()/(h))

    #Oblist=Oblist*L


    Room=rom.room(Oblist)

    xmin=Room.bounds[0][0]
    xmax=Room.bounds[1][0]
    ymin=Room.bounds[0][1]
    ymax=Room.bounds[1][1]
    zmin=Room.bounds[0][2]
    zmax=Room.bounds[1][2]
    xgrid=np.linspace(xmin,xmax,Nx)
    # Generate co-ordinates fot the grid
    for yp in range(0,Ny):
      yt=np.tile(ymin+(yp/Ny)*(ymax-ymin),(1,Nx))
      if yp==0:
        xygrid=np.vstack((yt,xgrid)).T
      else:
        xygrid=np.vstack((xygrid,np.vstack((xgrid,yt)).T))
    for zp in range(0,Nz):
      zt=np.tile(zmin+(zp/Nz)*(zmax-zmin),(Nx*Ny,1))
      xyz=np.hstack((xygrid,zt))
      if zp==0:
          cubep=xyz
      else:
          cubep=np.vstack((cubep,xyz))
    # Create Figures for each ray number
    for i in range(0, nra):
      # The intersection Points
      data_matrix   =np.load('./Mesh/'+plottype+'/RayMeshPoints%dRefs%dm.npy'%(Nra[i],Nre))
      # The arrays of Power Values
      Power=np.load('./Mesh/'+plottype+'/Power_grid%dRefs%dm%d.npy'%(Nra[i],Nre,0))
      Powx,Powy,Powz=np.mgrid[xmin:xmax:Nx*1.0j,ymin:ymax:Ny*1.0j,zmin:zmax:Nz*1.0j]
      Pmin=np.amin(Power)
      Pmax=np.amax(Power)
      # Plot the transmitter
      iternum=int(Nra[i])
      dist=np.linalg.norm(data_matrix[i][1][0:3]-Tx)
      # Parameters for calculating Cone length
      zsteps=int(2+np.pi/delangle[i])
      refangle=0
      nref=0
      Ncon=ra.no_cones(h,dist,delangle[i],refangle,nref)
      delth=ra.angle_space(delangle[i],nref)
      beta=ra.beta_leng(dist,delth,refangle)
      anglevec=np.linspace(0.0,2*ma.pi,num=int(Ncon), endpoint=False) # Create an array of all the angles
      Norm=np.zeros((Ncon,3),dtype=np.float) # Initialise the matrix of normals
      # Array to store Cone points
      Cones=np.zeros((iternum*Ncon,2,3))
      for j in range(0,iternum):
        cx,cy,cz,_=data_matrix[j][1]
        d=data_matrix[j][1][0:3]-Tx              # The direction of the ray
        d/=np.linalg.norm(d)
        if abs(d[2])>0 and Ncon>0:
         Norm[0]=np.array([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
         Norm[0]/=np.linalg.norm(Norm[0]) # Normalise the vector
         yax=np.cross(Norm[0],d)          # Compute another vector in the plane for the axis.
         yax/=np.linalg.norm(yax)         # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        elif Ncon>0:
         Norm[0]=np.array([0,0,1])        # If d_z is 0 then this vector is always in the plane.
         yax=np.cross(Norm[0],d)            # Compute another vector in the plane to form co-ordinate axis.
         yax/=np.linalg.norm(yax)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
        if Ncon>0:
          Norm=np.outer(np.cos(anglevec),Norm[0])+np.outer(np.sin(anglevec),yax) # Use the outer product to multiple the axis
        for k in range(0,Ncon):
            Cones[j*Ncon+k][0]=np.array([cx,cy,cz])
            Cones[j*Ncon+k][1]=np.array([cx,cy,cz])+beta*Norm[k]
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
        os.makedirs('./ConeFigures/'+plottype)
      if not os.path.exists('./ConeFigures/'+plottype):
        os.makedirs('./ConeFigures/'+plottype)
      mlab.savefig('ConeFigures/Rays.jpg',size=(1000,1000))
      #Plot all rays and cones
      mlab.figure(0)
      mlab.clf()
      mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
      for j in range(0,iternum*Ncon):
        x=np.array([Cones[j,0,0]])
        y=np.array([Cones[j,0,1]])
        z=np.array([Cones[j,0,2]])
        xp=np.append(x,[Cones[j,1,0]])
        yp=np.append(y,[Cones[j,1,1]])
        zp=np.append(z,[Cones[j,1,2]])
        mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.5)
      filename=str('ConeFigures/'+plottype+'/Cone'+str(iternum)+'SquareWithCones.jpg')
      mlab.savefig(filename,size=(1000,1000))
      mlab.clf()
      mlab.close(all=True)
      if Cut==1:
        count=0
        xyk=np.empty((zsteps,1))
        for l in range(0,zsteps):
          k=ma.ceil(l-zsteps/2)
          if k>ma.ceil(zsteps/2):
            break
          mid=(np.cos(delangle[i])-np.sin(k*delangle[i])**2)/(np.cos(delangle[i]*k)**2)
          if abs(mid)>1:
            xyk[l]=1
          else:
            bot=ma.acos(mid)
            xyk[l]=int(2*np.pi/bot)
        for l in range(0,zsteps):
          mlab.figure(l)
          k=ma.ceil(l-zsteps/2)
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='z_axes',
                            slice_index=int(zsteps-l),
                            colormap=cmapopt,
                            vmax=Pmax,
                            vmin=Pmin
                        )
          mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
          mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
          for k2 in range(0,int(xyk[l])):
            j=count
            if count>iternum*Ncon-1:
              break
            x=np.array([Cones[j,0,0]])
            y=np.array([Cones[j,0,1]])
            z=np.array([Cones[j,0,2]])
            xc=np.append(x,[Tx[0]])
            yc=np.append(y,[Tx[1]])
            zc=np.append(z,[Tx[2]])
            mlab.plot3d(xc,yc,zc,color= (0,1,1),opacity=0.5)
            if ConeOn==1:
              for j in range(0,Ncon):
                xp=np.append(x,[Cones[j,1,0]])
                yp=np.append(y,[Cones[j,1,1]])
                zp=np.append(z,[Cones[j,1,2]])
                mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.5)
                count+=1
            else: count+=Ncon
          filename=str('ConeFigures/'+plottype+'/Cone'+str(iternum)+'Square'+str(int(l))+'.jpg')
          mlab.savefig(filename,size=(1000,1000))
          mlab.clf()
          mlab.close(all=True)
      if Cut==0:
        for l in range(0,Nz):
           mlab.figure(l)
           mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='z_axes',
                            slice_index=l,
                            colormap=cmapopt,
                            vmax=Pmax,
                            vmin=Pmin
                        )
           mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
           mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
           count=0
           for l2 in range(0,Nra[i]):
             j=count
             x=np.array([Cones[j][0][0]])
             y=np.array([Cones[j][0][1]])
             z=np.array([Cones[j][0][2]])
             xc=np.append(x,[Tx[0]])
             yc=np.append(y,[Tx[1]])
             zc=np.append(z,[Tx[2]])
             mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.5)
             if ConeOn==1:
               for k in range(0, int(Ncon)):
                 j=count
                 if count>iternum*Ncon-1:
                  break
                 xp=np.append(x,[Cones[j][1][0]])
                 yp=np.append(y,[Cones[j][1][1]])
                 zp=np.append(z,[Cones[j][1][2]])
                 mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.5)
                 count+=1
             else: count+=Ncon
           filename=str('ConeFigures/'+plottype+'/Cone'+str(iternum)+'FullSquareZ'+str(int(l))+'.jpg')
           mlab.savefig(filename,size=(1000,1000))
           mlab.clf()
           mlab.close(all=True)
        for l in range(0,Nx):
          mlab.figure(l)
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='x_axes',
                            slice_index=l,
                            colormap=cmapopt,
                            vmax=Pmax,
                            vmin=Pmin
                            )
          mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
          mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
          count=0
          for l2 in range(0,Nra[i]):
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xc=np.append(x,[Tx[0]])
            yc=np.append(y,[Tx[1]])
            zc=np.append(z,[Tx[2]])
            mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.5)
            if ConeOn==1:
              for k in range(0, int(Ncon)):
                j=count
                if count>iternum*Ncon-1:
                 break
                xp=np.append(x,[Cones[j][1][0]])
                yp=np.append(y,[Cones[j][1][1]])
                zp=np.append(z,[Cones[j][1][2]])
                mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.5)
                count+=1
            else: count+=Ncon
          filename=str('ConeFigures/'+plottype+'/Cone'+str(iternum)+'FullSquareX'+str(int(l))+'.jpg')
          mlab.savefig(filename,size=(1000,1000))
          mlab.clf()
          mlab.close(all=True)
        for l in range(0,Ny):
          count=0
          mlab.figure(l)
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='y_axes',
                            slice_index=l,
                            colormap=cmapopt,
                            vmax=Pmax,
                            vmin=Pmin
                        )
          mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
          mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
          for l2 in range(0,Nra[i]):
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xc=np.append(x,[Tx[0]])
            yc=np.append(y,[Tx[1]])
            zc=np.append(z,[Tx[2]])
            mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.5)
            if ConeOn:
              for k in range(0, int(Ncon)):
                j=count
                if count>iternum*Ncon-1:
                  break
                x=np.array([Cones[j][0][0]])
                y=np.array([Cones[j][0][1]])
                z=np.array([Cones[j][0][2]])
                xp=np.append(x,[Cones[j][1][0]])
                yp=np.append(y,[Cones[j][1][1]])
                zp=np.append(z,[Cones[j][1][2]])
                mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.5)
                count+=1
            else: count+=Ncon
          filename=str('ConeFigures/'+plottype+'/Cone'+str(iternum)+'FullSquareY'+str(int(l))+'.jpg')
          mlab.savefig(filename,size=(1000,1000))
          mlab.clf()
          mlab.close(all=True)
      # if Vid==1:
          # mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            # plane_orientation='y_axes', slice_index=0,
                            # colormap=cmapopt,
                            # vmax=Pmax, vmin=Pmin
                        # )
          # mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            # plane_orientation='x_axes', slice_index=0,
                            # colormap=cmapopt,
                            # vmax=Pmax,vmin=Pmin
                        # )
          # mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            # plane_orientation='z_axes', slice_index=0,
                            # colormap=cmapopt,
                            # vmax=Pmax,vmin=Pmin
                        # )
          # try:
              # engine = mayavi.engine
          # except NameError:
              # from mayavi.api import Engine
              # engine = Engine()
              # engine.start()
          # if len(engine.scenes) == 0:
              # engine.new_scene()
          # # -------------------------------------------
          # scene = engine.scenes[0]
          # scene.scene.camera.position = [8.709726813612612, 8.684071038273604, 8.679837720224306]
          # scene.scene.camera.focal_point = [1.524111658260839, 1.4984558829218613, 1.4942225648724614]
          # scene.scene.camera.view_angle = 30.0
          # scene.scene.camera.view_up = [0.0, 0.0, 1.0]
          # scene.scene.camera.clipping_range = [5.9135330467319855, 20.69871253517983]
          # scene.scene.camera.compute_view_plane_normal()
          # scene.scene.render()
          # image_plane_widget2 = engine.scenes[0].children[1488].children[0].children[0]
          # image_plane_widget2.ipw.point2 = array([-0.16666667,  3.16666667,  3.        ])
          # image_plane_widget2.ipw.slice_index = 9
          # image_plane_widget2.ipw.origin = array([-0.16666667, -0.16666667,  3.        ])
          # image_plane_widget2.ipw.slice_position = 3.0
          # image_plane_widget2.ipw.point1 = array([ 3.16666667, -0.16666667,  3.        ])
          # image_plane_widget2.ipw.point2 = array([-0.16666667,  3.16666667,  3.        ])
          # image_plane_widget2.ipw.origin = array([-0.16666667, -0.16666667,  3.        ])
          # image_plane_widget2.ipw.point1 = array([ 3.16666667, -0.16666667,  3.        ])
          # image_plane_widget1 = engine.scenes[0].children[1487].children[0].children[0]
          # image_plane_widget1.ipw.point2 = array([ 3.        , -0.16666667,  3.16666667])
          # image_plane_widget1.ipw.slice_index = 9
          # image_plane_widget1.ipw.origin = array([ 3.        , -0.16666667, -0.16666667])
          # image_plane_widget1.ipw.slice_position = 3.0
          # image_plane_widget1.ipw.point1 = array([ 3.        ,  3.16666667, -0.16666667])
          # image_plane_widget1.ipw.point2 = array([ 3.        , -0.16666667,  3.16666667])
          # image_plane_widget1.ipw.origin = array([ 3.        , -0.16666667, -0.16666667])
          # image_plane_widget1.ipw.point1 = array([ 3.        ,  3.16666667, -0.16666667])
          # image_plane_widget = engine.scenes[0].children[1486].children[0].children[0]
          # image_plane_widget.ipw.point2 = array([ 3.16323145, -0.10695289, -0.16666667])
          # image_plane_widget.ipw.plane_orientation = 3
          # image_plane_widget.ipw.origin = array([-0.16323145,  0.10695289, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.16323145,  0.10695289,  3.16666667])
          # image_plane_widget.ipw.point2 = array([ 3.16323145, -0.10695289, -0.16666667])
          # image_plane_widget.ipw.origin = array([-0.16323145,  0.10695289, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.16323145,  0.10695289,  3.16666667])
          # image_plane_widget1.ipw.point2 = array([ 2.72003802, -0.16666667,  3.1429848 ])
          # image_plane_widget1.ipw.slice_index = 0
          # image_plane_widget1.ipw.plane_orientation = 3
          # image_plane_widget1.ipw.origin = array([ 3.27996198, -0.16666667, -0.1429848 ])
          # image_plane_widget1.ipw.slice_position = 0.0
          # image_plane_widget1.ipw.point1 = array([ 3.27996198,  3.16666667, -0.1429848 ])
          # image_plane_widget1.ipw.point2 = array([ 2.72003802, -0.16666667,  3.1429848 ])
          # image_plane_widget1.ipw.origin = array([ 3.27996198, -0.16666667, -0.1429848 ])
          # image_plane_widget1.ipw.point1 = array([ 3.27996198,  3.16666667, -0.1429848 ])
          # image_plane_widget.ipw.point2 = array([ 3.16646401, -0.02598989, -0.16666667])
          # image_plane_widget.ipw.origin = array([-0.16646401,  0.02598989, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.16646401,  0.02598989,  3.16666667])
          # image_plane_widget.ipw.point2 = array([ 3.16646401, -0.02598989, -0.16666667])
          # image_plane_widget.ipw.origin = array([-0.16646401,  0.02598989, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.16646401,  0.02598989,  3.16666667])
          # scene.scene.camera.position = [10.218705996236485, 10.19305022089747, 10.188816902848195]
          # scene.scene.camera.focal_point = [1.524111658260839, 1.4984558829218613, 1.4942225648724614]
          # scene.scene.camera.view_angle = 30.0
          # scene.scene.camera.view_up = [0.0, 0.0, 1.0]
          # scene.scene.camera.clipping_range = [8.501025372481612, 23.351545576226158]
          # scene.scene.camera.compute_view_plane_normal()
          # scene.scene.render()
          # scene.scene.camera.position = [10.014008607801468, 10.326333798987228, 10.256535809164832]
          # scene.scene.camera.focal_point = [1.524111658260839, 1.4984558829218613, 1.4942225648724614]
          # scene.scene.camera.view_angle = 30.0
          # scene.scene.camera.view_up = [-0.4084477922716211, -0.4144272692768555, 0.8132775906590367]
          # scene.scene.camera.clipping_range = [8.501152646787327, 23.35138536409761]
          # scene.scene.camera.compute_view_plane_normal()
          # scene.scene.render()
          # image_plane_widget1.ipw.point2 = array([-0.27996198, -0.16666667,  2.59453612])
          # image_plane_widget1.ipw.origin = array([ 0.27996198, -0.16666667, -0.69143349])
          # image_plane_widget1.ipw.point1 = array([ 0.27996198,  3.16666667, -0.69143349])
          # image_plane_widget1.ipw.point2 = array([-0.27996198, -0.16666667,  2.59453612])
          # image_plane_widget1.ipw.origin = array([ 0.27996198, -0.16666667, -0.69143349])
          # image_plane_widget1.ipw.point1 = array([ 0.27996198,  3.16666667, -0.69143349])
          # image_plane_widget.ipw.point2 = array([ 3.16565024,  0.05819841, -0.16666667])
          # image_plane_widget.ipw.origin = array([-0.16565024, -0.05819841, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.16565024, -0.05819841,  3.16666667])
          # image_plane_widget.ipw.point2 = array([ 3.16565024,  0.05819841, -0.16666667])
          # image_plane_widget.ipw.origin = array([-0.16565024, -0.05819841, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.16565024, -0.05819841,  3.16666667])
          # image_plane_widget.ipw.point2 = array([ 3.03872615,  3.05819841, -0.16666667])
          # image_plane_widget.ipw.origin = array([-0.29257433,  2.94180159, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.29257433,  2.94180159,  3.16666667])
          # image_plane_widget.ipw.point2 = array([ 3.03872615,  3.05819841, -0.16666667])
          # image_plane_widget.ipw.origin = array([-0.29257433,  2.94180159, -0.16666667])
          # image_plane_widget.ipw.point1 = array([-0.29257433,  2.94180159,  3.16666667])
    # mlab.clf()
    # mlab.close(all=True)
    return

def PlotDirections(plottype=str()):

    ##Plot the obstacles and the room first

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L    =np.load('Parameters/Raytracing.npy')[0:3]
    # Take Tx to be 0,0,0
    delang     =np.load('Parameters/delangle.npy')

    mlab.points3d(0,0,0,scale_factor=0.1)
    mulfac=4
    for i in range(0,nra):
      directionname='Parameters/Directions%03d.npy'%Nra[i]
      data_matrix   =np.load(directionname)         # Matrix of ray directions
      delangle=delang[i]
      #mlab.savefig('ConeFigures/Rays.jpg',size=(1000,1000))
      #mlab.clf()
      #mlab.close(all=True)
      delth=2*np.arcsin(np.sqrt(2)*ma.sin(delangle/2))
      ta=np.sqrt(1-ma.tan(delth/2)**2)   # Nra>2 and an integer. Therefore tan(theta) exists.
      s=ma.sin(delth/2)
      for l in range(0,int(Nra[i])):
        j=l
        if data_matrix[j][0]!=0:
          a=1/abs(data_matrix[j][0])
        else:
          a=1
        if data_matrix[j][1]!=0:
          b=1/abs(data_matrix[j][1])
        else:
          b=1
        if data_matrix[j][2]!=0:
          c=1/abs(data_matrix[j][2])
        else:
          c=1
        sqfac=1 #min(a,b,c)
        x=0
        y=0
        z=0
        x2,y2,z2=sqfac*mulfac*data_matrix[j][0:3]/np.linalg.norm(data_matrix[j][0:3])
        x=np.append(x,[x2])
        y=np.append(y,[y2])
        z=np.append(z,[z2])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
      filename='ConeFigures/'+plottype+'/Rays%03d.jpg'%Nra[i]
      mlab.savefig(filename,size=(1000,1000))
      filename='ConeFigures/Rays%03d.jpg'%Nra[i]
      mlab.savefig(filename,size=(1000,1000))
      #gui = GUI()
      #gui.start_event_loop()
      #mlab.clf()
      #mlab.close(all=True)
    return

def PlotPowerSlice(plottype):
    '''Plot slices through the environment showing heatmaps of the power.
    Step through the xx, y, and z axis and plot the power. The error residual,
    and the distances of the rays.
    '''
    myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
    cmapopt= myfile.read()         # read the entire file into a string
    myfile.close()
    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    Nob         =np.load('Parameters/Nob.npy')
    Nsur        =np.load('Parameters/Nsur.npy')
    Tx          =np.load('Parameters/Origin.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
        ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
    #Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

    # Room contains all the obstacles and walls.
    Room=rom.room(Oblist)

    Nz=int(Room.maxzleng()/(h))
    Nx=int(Room.maxxleng()/(h))
    Ny=int(Room.maxyleng()/(h))
    # Get the x, y, z co-ordinates for P
    xmin=Room.bounds[0][0]
    xmax=Room.bounds[1][0]
    ymin=Room.bounds[0][1]
    ymax=Room.bounds[1][1]
    zmin=Room.bounds[0][2]
    zmax=Room.bounds[1][2]
    if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
        os.makedirs('./ConeFigures/'+plottype)
    if not os.path.exists('./ConeFigures/'+plottype):
        os.makedirs('./ConeFigures/'+plottype)
    xgrid=np.linspace(xmin,xmax,Nx)
    # Generate co-ordinates fot the grid
    for yp in range(0,Ny):
      yt=np.tile(ymin+(yp/Ny)*(ymax-ymin),(1,Nx))
      if yp==0:
        xygrid=np.vstack((yt,xgrid)).T
      else:
        xygrid=np.vstack((xygrid,np.vstack((xgrid,yt)).T))
    for zp in range(0,Nz):
      zt=np.tile(zmin+(zp/Nz)*(zmax-zmin),(Nx*Ny,1))
      xyz=np.hstack((xygrid,zt))
      if zp==0:
          cubep=xyz
      else:
          cubep=np.vstack((cubep,xyz))
    #TrueGrid=np.load('Parameters/'+plottype+'/True.npy')
    RoomP=OuterBoundary[0]#Oblist[0]
    for j in range(1,Nob):
      RoomP=np.concatenate((RoomP,Oblist[j]),axis=0)
    for i in range(0,nra):
      Power=np.load('./Mesh/'+plottype+'/Power_grid%dRefs%dm%d.npy'%(Nra[i],Nre,0))
      RadA=np.load('./Mesh/'+plottype+'/RadA_grid%dRefs%dm%d.npy'%(Nra[i],Nre,0))
      RadS=np.zeros((Nob,Nx,Ny,Nz))
      for j in range(0,Nsur):
        RadS[j]=np.load('./Mesh/'+plottype+'/RadS%d_grid%dRefs%dm%d.npy'%(j,Nra[i],Nre,0))
      Powx,Powy,Powz=np.mgrid[xmin:xmax:Nx*1.0j,ymin:ymax:Ny*1.0j,zmin:zmax:Nz*1.0j]
      Pmin=np.amin(Power)
      Pmax=np.amax(Power)
      Radmax=max(np.amax(RadA),np.amax(RadS))
      Radmin=min(np.amin(RadA),np.amin(RadS))
      x,y,z=np.mgrid[xmin:xmax:Nx*1.0j,ymin:ymax:Ny*1.0j,zmin:zmax:Nz*1.0j]
      mlab.volume_slice(x,y,z,Power, plane_orientation='x_axes', slice_index=10)
      #mlab.outline()
      #mlab.show()
      mlab.clf()
      mlab.figure()
      for l in range(0,Nx):
        mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
        for j in range(0,RoomP.shape[0],3):
          x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
          y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
          z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
          for zi in range(4):
            if y[zi]==0 or y[zi]==2:
              pass
            else:
              print(zi,y[zi])
          mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.2)
        mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='x_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Pmax,
                            vmin=Pmin
                        )
        filename='ConeFigures/'+plottype+'/Cone%dPowersliceX%d.jpg'%(Nra[i],l)
        mlab.savefig(filename,size=(1000,1000))
        mlab.clf()
        mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
        for j in range(0,RoomP.shape[0],3):
          x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
          y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
          z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
          mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
          for zi in range(4):
            if y[zi]==0 or y[zi]==2:
              pass
            else:
              print(zi,y[zi])
        mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,RadA),
                            plane_orientation='x_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Radmax,
                            vmin=Radmin
                        )
        filename='ConeFigures/'+plottype+'/Cone%dRadAX%d.jpg'%(Nra[i],l)
        mlab.savefig(filename,size=(1000,1000))
        mlab.clf()
        for j in range(0,Nsur):
          mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
          for k in range(0,RoomP.shape[0],3):
            x=np.array([RoomP[k][0],RoomP[k+1][0],RoomP[k+2][0],RoomP[k][0]])
            y=np.array([RoomP[k][1],RoomP[k+1][1],RoomP[k+2][1],RoomP[k][1]])
            z=np.array([RoomP[k][2],RoomP[k+1][2],RoomP[k+2][2],RoomP[k][2]])
            mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
            for zi in range(4):
              if y[zi]==0 or y[zi]==2:
                pass
              else:
                print(zi,y[zi])
          mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,RadS[j]),
                            plane_orientation='x_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Radmax,
                            vmin=Radmin
                        )
          filename='ConeFigures/'+plottype+'/Cone%dRadS%dX%d.jpg'%(Nra[i],j,l)
          mlab.savefig(filename,size=(1000,1000))
          mlab.clf()
      for l in range(0,Ny):
        mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
        for j in range(0,RoomP.shape[0],3):
          x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
          y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
          z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
          mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
          for zi in range(4):
            if y[zi]==0 or y[zi]==2:
              pass
            else:
              print(zi,y[zi])
        mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='y_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Pmax,
                            vmin=Pmin
                        )
        filename='ConeFigures/'+plottype+'/Cone%dPowersliceY%d.jpg'%(Nra[i],l)
        mlab.savefig(filename,size=(1000,1000))
        mlab.clf()
        mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
        for j in range(0,RoomP.shape[0],3):
          x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
          y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
          z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
          mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
          for zi in range(4):
            if y[zi]==0 or y[zi]==2:
              pass
            else:
              print(zi,y[zi])
        mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,RadA),
                            plane_orientation='y_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Radmax,
                            vmin=Radmin
                        )
        filename='ConeFigures/'+plottype+'/Cone%dRadAY%d.jpg'%(Nra[i],l)
        mlab.savefig(filename,size=(1000,1000))
        mlab.clf()
        for j in range(0,Nsur):
          mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
          for k in range(0,RoomP.shape[0],3):
            x=np.array([RoomP[k][0],RoomP[k+1][0],RoomP[k+2][0],RoomP[k][0]])
            y=np.array([RoomP[k][1],RoomP[k+1][1],RoomP[k+2][1],RoomP[k][1]])
            z=np.array([RoomP[k][2],RoomP[k+1][2],RoomP[k+2][2],RoomP[k][2]])
            mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
            for zi in range(4):
              if y[zi]==0 or y[zi]==2:
                pass
              else:
                print(zi,y[zi])
          mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,RadS[j]),
                            plane_orientation='y_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Radmax,
                            vmin=Radmin
                        )
          filename='ConeFigures/'+plottype+'/Cone%dRadS%dY%d.jpg'%(Nra[i],j,l)
          mlab.savefig(filename,size=(1000,1000))
          mlab.clf()
      for l in range(0,Nz):
        mlab.clf()
        mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
        for j in range(0,RoomP.shape[0],3):
          x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
          y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
          z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
          mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
          for zi in range(4):
            if y[zi]==0 or y[zi]==2:
              pass
            else:
              print(zi,y[zi])
        mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='z_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Pmax,
                            vmin=Pmin
                        )
        filename='ConeFigures/'+plottype+'/Cone%dPowersliceZ%d.jpg'%(Nra[i],l)
        mlab.savefig(filename,size=(1000,1000))
        mlab.clf()
        mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
        for j in range(0,RoomP.shape[0],3):
          x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
          y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
          z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
          mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
          for zi in range(4):
            if y[zi]==0 or y[zi]==2:
              pass
            else:
              print(zi,y[zi])
        mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,RadA),
                            plane_orientation='z_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Radmax,
                            vmin=Radmin
                        )
        filename='ConeFigures/'+plottype+'/Cone%dRadAZ%d.jpg'%(Nra[i],l)
        mlab.savefig(filename,size=(1000,1000))
        mlab.clf()
        for j in range(0,Nsur):
          mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
          for k in range(0,RoomP.shape[0],3):
            x=np.array([RoomP[k][0],RoomP[k+1][0],RoomP[k+2][0],RoomP[k][0]])
            y=np.array([RoomP[k][1],RoomP[k+1][1],RoomP[k+2][1],RoomP[k][1]])
            z=np.array([RoomP[k][2],RoomP[k+1][2],RoomP[k+2][2],RoomP[k][2]])
            mlab.plot3d(x,y,z,color=(0.75,0.75,0.75),opacity=0.05)
            for zi in range(4):
              if y[zi]==0 or y[zi]==2:
                pass
              else:
                print(zi,y[zi])
          mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.05)
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,RadS[j]),
                            plane_orientation='z_axes',
                            slice_index=int(l),
                            colormap=cmapopt,
                            vmax=Radmax,
                            vmin=Radmin
                        )
          filename='ConeFigures/'+plottype+'/Cone%dRadS%dZ%d.jpg'%(Nra[i],j,l)
          mlab.savefig(filename,size=(1000,1000))
          mlab.clf()
    return Power

def PlotSingleCone(plottype):
  '''Plot a single cone from the Ray Tracer. '''
  ##----Retrieve the Raytracing Parameters-----------------------------
  Nra         =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nre,h ,L =np.load('Parameters/Raytracing.npy')
  # Take Tx to be 0,0,0
  Tx=     np.load('Parameters/Origin.npy')/L
  delangle      =np.load('Parameters/delangle.npy')
  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
  Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)

  Nz=int(Room.maxyleng()/(h))
  Nx=int(Room.maxxleng()/(h))
  Ny=int(Room.maxyleng()/(h))

  Oblist=Oblist

  Room=rom.room(Oblist)

  xmin=Room.bounds[0][0]
  xmax=Room.bounds[1][0]
  ymin=Room.bounds[0][1]
  ymax=Room.bounds[1][1]
  zmin=Room.bounds[0][2]
  zmax=Room.bounds[1][2]
  xgrid=np.linspace(xmin,xmax,Nx)
  for i in range(0,  nra):
      data_matrix   =np.load('./Mesh/'+plottype+'RayMeshPoints%dRefs%dm.npy'%(Nra[i],Nre))
      Power=np.load('./Mesh/'+plottype+'Power_grid%dRefs%dm%d.npy'%(Nra[i],Nre,0))
      Powx,Powy,Powz=np.mgrid[xmin:xmax:Nx*1.0j,ymin:ymax:Ny*1.0j,zmin:zmax:Nz*1.0j]
      Cones=np.load('./Mesh/'+plottype+'SingleCone%dRefs%dm.npy'%(Nra[i],Nre))
      mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
      for j in range(0, Nra[i]):
        x=Tx[0]
        y=Tx[1]
        z=Tx[2]
        x2,y2,z2=data_matrix[j][1][0:3]
        x=np.append(x,[x2])
        y=np.append(y,[y2])
        z=np.append(z,[z2])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
        nob=data_matrix[j][1][3]
        dist=np.linalg.norm(np.array([x2,y2,z2])-Tx)
        direc=(np.array([x2,y2,z2])-Tx)/dist
        if j==0:
          obst=Room.obst[int(nob-1)]
          norm=np.cross(obst[1]-obst[0],obst[2]-obst[0])
          norm/=(np.linalg.norm(norm))
          check=(np.linalg.norm(direc)*np.linalg.norm(norm))
          if abs(check-0.0)<=epsilon:
            raise ValueError('direction or normal has no length')
          else:
            nleng=np.linalg.norm(norm)
            dleng=np.linalg.norm(direc)
            frac=np.dot(direc,norm)/(nleng*dleng)
          refangle=ma.acos(frac)
          if refangle>np.pi/2:
            refangle=np.pi-refangle
          Ncon=ra.no_cones(h,dist,delangle[i],refangle,0)
          Ns=int(Cones.shape[0]/((Ncon+1)))
          p0=Tx
          for l in range(0,Ns):
            j=l*(Ncon+1)
            p0=Cones[j]
            for k in range(0,Ncon):
              j=l*(Ncon+1)+k+1
              x,y,z=Room.coordinate(h,Cones[j][0],Cones[j][1],Cones[j][2])
              xp=np.append(p0[0],[x])
              yp=np.append(p0[1],[y])
              zp=np.append(p0[2],[z])
              mlab.plot3d(xp,yp,zp,color= (1,0,1))
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
        os.makedirs('./ConeFigures/'+plottype)
      if not os.path.exists('./ConeFigures/'+plottype):
        os.makedirs('./ConeFigures/'+plottype)
      filename='ConeFigures/'+plottype+'/SingleCone%d.jpg'%Nra[i]
      mlab.savefig(filename,size=(1000,1000))
      #mlab.clf()
      gui = GUI()
      gui.start_event_loop()
      #mlab.close(all=True)
    #mlab.show()
  return

def jobfromTx(Tx,h):
  H=(Tx[2]-0.5*h)//h
  t=(Tx[0]-0.5*h)//h
  u=(Tx[1]-0.5*h)//h
  return int(H*100+t*10+u)


if __name__=='__main__':
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  #PlotSingleCone(plottype)
  #PlotPowerSlice(plottype)
  PlotRays(plottype)
  #PlotDirections(plottype)
  #PlotConesOnSquare(plottype)
  #PlotConesGains(plottype)
  #PlotPolarTheoryGains(plottype)
  #PlotPolarGains(plottype)
  print('Running  on python version')
  print(sys.version)
exit()

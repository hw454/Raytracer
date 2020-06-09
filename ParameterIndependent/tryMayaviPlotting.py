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

epsilon=sys.float_info.epsilon

def PlotRays():
    ##Plot the obstacles and the room first

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L    =np.load('Parameters/Raytracing.npy')
    #Nra=98
    #Nre=5

    #L=1
    #Nra=int(np.sqrt(Nrao/2.0)-1)*int(np.sqrt(2.0*Nrao))+1

    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')
    Tx            =np.load('Parameters/Origin.npy')
    Nob           =len(Oblist)
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')
    Nob2          =len(OuterBoundary)
    Room          =np.concatenate((Oblist,OuterBoundary),axis=0)
    Oblist        =Room
    RoomP=Oblist[0]
    for j in range(1,Nob):
      RoomP=np.concatenate((RoomP,Oblist[j]),axis=0)
    for j in range(0,Nob2):
      RoomP=np.concatenate((RoomP,OuterBoundary[j]),axis=0)
    mlab.clf()
    #mlab.figure(1)
    for i in range(nra):
      mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.25)
      ##---Retrieve the Ray points ---------------------------------------
      data_matrix=L*np.load('./Mesh/RayMeshPoints'+str(int(Nra[i]))+'Refs'+str(int(Nre))+'m.npy')
      #data_matrix=data_matrix #print(data_matrix)
      for j in range(0,int(Nra[i])):
        x=np.array([data_matrix[j][0][0]])
        y=np.array([data_matrix[j][0][1]])
        z=np.array([data_matrix[j][0][2]])
        s=np.array([data_matrix[j][0][3]])
        x=np.append(x,[data_matrix[j][1][0]])
        y=np.append(y,[data_matrix[j][1][1]])
        z=np.append(z,[data_matrix[j][1][2]])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
      for k in range(0,int(len(RoomP)/3)):
        j=k*3
        x=np.array([RoomP[j][0],RoomP[j+1][0],RoomP[j+2][0],RoomP[j][0]])
        y=np.array([RoomP[j][1],RoomP[j+1][1],RoomP[j+2][1],RoomP[j][1]])
        z=np.array([RoomP[j][2],RoomP[j+1][2],RoomP[j+2][2],RoomP[j][2]])
        mlab.plot3d(x,y,z)
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
      mlab.savefig('ConeFigures/Room'+str(int(Nra[i]))+'.jpg',size=(1000,1000))
      mlab.clf()
      mlab.close()
    return

def PlotCones():
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
      directionname=str('Parameters/Directions'+str(int(i))+'.npy')
      data_matrix   =np.load(directionname)         # Matrix of ray directions
      mlab.points3d(0,0,0,scale_factor=0.1)
      zsteps=int(np.pi/delangle[i])
      #mlab.savefig('ConeFigures/Rays.jpg',size=(1000,1000))
      #mlab.clf()
      #mlab.close()
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
      for j in range(0,int(Nra[i])):
        x=0
        y=0
        z=0
        x2,y2,z2=mulfac*data_matrix[j][0:3]
        x=np.append(x,[x2])
        y=np.append(y,[y2])
        z=np.append(z,[z2])
        mlab.plot3d(x,y,z,color= (0, 1, 1))
      mlab.savefig('ConeFigures/Rays'+str(int(Nra[i]))+'.jpg',size=(1000,1000))
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
      filename=str('ConeFigures/Cone'+str(int(Nra[i]))+'.jpg')
      mlab.savefig(filename,size=(1000,1000))
      mlab.clf()
      mlab.close()
    #mlab.show()
    return

def PlotConesOnSquare():
    '''Plot the cone calculations.'''

    Cut=1
    Vid=0
    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')
    # Take Tx to be 0,0,0
    Tx=     np.load('Parameters/Origin.npy')
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

    Oblist=Oblist*L

    Room=rom.room(Oblist)

    xmin=Room.bounds[0][0]
    xmax=Room.bounds[1][0]
    ymin=Room.bounds[0][1]
    ymax=Room.bounds[1][1]
    zmin=Room.bounds[0][2]
    zmax=Room.bounds[1][2]
    xgrid=np.linspace(xmin,xmax,Nx)
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
    for i in range(0, nra):
      data_matrix   =L*np.load('./Mesh/RayMeshPoints'+str(int(Nra[i]))+'Refs'+str(int(Nre))+'m.npy')
      Power=np.load('./Mesh/Power_grid'+str(int(Nra[i]))+'Refs'+str(int(Nre))+'m'+str(0)+'.npy')
      Powx,Powy,Powz=np.mgrid[xmin:xmax:Nx*1.0j,ymin:ymax:Ny*1.0j,zmin:zmax:Nz*1.0j]
      Pmin=np.amin(Power)
      Pmax=np.amax(Power)
      mlab.points3d(Tx[0],Tx[1],Tx[2],scale_factor=0.1)
      zsteps=int(np.pi/delangle[i])
      iternum=int(Nra[i])
      dist=np.linalg.norm(data_matrix[i][0][0:3]-data_matrix[i][1][0:3])
      refangle=0
      nref=0
      Ncon=ra.no_cones(h,dist,delangle[i],refangle,nref)
      delth=ra.angle_space(delangle[i],nref)
      beta=ra.beta_leng(dist,delth,refangle)
      anglevec=np.linspace(0.0,2*ma.pi,num=int(Ncon), endpoint=False) # Create an array of all the angles
      Norm=np.zeros((Ncon,3),dtype=np.float) # Initialise the matrix of normals
      Cones=np.zeros((iternum*Ncon,2,3))
      for j in range(0,iternum):#int(Nrao)):
        cx,cy,cz,cs=data_matrix[j][1]
        d=data_matrix[j][1][0:3]-data_matrix[j][0][0:3]               # The direction of the ray
        d/=np.linalg.norm(d)
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
            dist=np.linalg.norm(data_matrix[j][1][0:3]-data_matrix[j][0][0:3])
            Cones[j*Ncon+k][1]=(data_matrix[j][1][0:3])+beta*Norm[k]
      if not os.path.exists('./ConeFigures'):
        os.makedirs('./ConeFigures')
      #mlab.savefig('ConeFigures/Rays.jpg',size=(1000,1000))
      count=0
      N=int(zsteps)
      xysteps=int(ma.ceil(abs(2.0*np.pi/delangle[i])))
      for l in range(0,N+1):
        #j=int(Nre)*k
        mid=(np.cos(delangle[i])-np.sin(l*delangle[i])**2)/(np.cos(delangle[i]*l)**2)
        if abs(mid)>1:
          break
        bot=ma.acos(mid)
        xyk=int(2*np.pi/bot)
        if xyk<=1:
          break
        #mlab.volume_slice(Powx,Powy,Powz,Power, plane_orientation='z_axes', slice_index=N-l)
        #mlab.volume_slice(Powx,Powy,Powz,Power, plane_orientation='x_axes', slice_index=int(0.5*Nx))
        #mlab.volume_slice(Powx,Powy,Powz,Power, plane_orientation='y_axes', slice_index=int(0.5*Ny))
        if l==0:
          SI=Nz
        else:
          j=count
          z=np.array([Cones[j][0][2]])
          SI=int(Nz-(zmax-z)/(h*L))
        if Cut==0:
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='z_axes',
                            slice_index=SI,
                            colormap='viridis',
                            vmax=Pmax,
                            vmin=Pmin
                        )
          mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.1)
        if l==0 or l==N:
          j=count
          x=np.array([Cones[0,0,0]])
          y=np.array([Cones[j][0][1]])
          z=np.array([Cones[j][0][2]])
          xc=np.append(x,[Tx[0]])
          yc=np.append(y,[Tx[1]])
          zc=np.append(z,[Tx[2]])
          mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.1)
          for k in range(0, int(Ncon)):
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xp=np.append(x,[Cones[j][1][0]])
            yp=np.append(y,[Cones[j][1][1]])
            zp=np.append(z,[Cones[j][1][2]])
            mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.1)
            count+=1
          if Cut==0:
            filename=str('ConeFigures/Cone'+str(iternum)+'Square'+str(int(l))+'.jpg')
          #mlab.show()
            mlab.savefig(filename,size=(1000,1000))
            mlab.clf()
        else:
          #mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.5,0.5,0.5),opacity=0.5)
          for k in range(0,int(xyk*Ncon)):
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            if j % Ncon ==0:
              xc=np.append(x,[Tx[0]])
              yc=np.append(y,[Tx[1]])
              zc=np.append(z,[Tx[2]])
              mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.1)
            xp=np.append(x,[Cones[j][1][0]])
            yp=np.append(y,[Cones[j][1][1]])
            zp=np.append(z,[Cones[j][1][2]])
            mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.1)
            count+=1
          if Cut==0:
            filename=str('ConeFigures/Cone'+str(iternum)+'Square'+str(int(l))+'.jpg')
            #mlab.show()
            mlab.savefig(filename,size=(1000,1000))
            mlab.clf()
      if Cut==1 and Vid==0:
        count=0
        for l in range(0,Nz):
           mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='z_axes',
                            slice_index=l,
                            colormap='viridis',
                            vmax=Pmax,
                            vmin=Pmin
                        )
           mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.1)
           count=0
           for l2 in range(0,Nra[i]):
             j=count
             x=np.array([Cones[j][0][0]])
             y=np.array([Cones[j][0][1]])
             z=np.array([Cones[j][0][2]])
             xc=np.append(x,[Tx[0]])
             yc=np.append(y,[Tx[1]])
             zc=np.append(z,[Tx[2]])
             mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.1)
             for k in range(0, int(Ncon)):
               j=count
               x=np.array([Cones[j][0][0]])
               y=np.array([Cones[j][0][1]])
               z=np.array([Cones[j][0][2]])
               xp=np.append(x,[Cones[j][1][0]])
               yp=np.append(y,[Cones[j][1][1]])
               zp=np.append(z,[Cones[j][1][2]])
               mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.1)
               count+=1
             if l2==Nra[i]-1:
              filename=str('ConeFigures/Cone'+str(iternum)+'FullSquareZ'+str(int(l))+'.jpg')
              mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.1)
              mlab.savefig(filename,size=(1000,1000))
              mlab.clf()
        for l in range(0,Nx):
          mid=(np.cos(delangle[i])-np.sin(l*delangle[i])**2)/(np.cos(delangle[i]*l)**2)
          if abs(mid)>1:
            break
          bot=ma.acos(mid)
          xyk=int(2*np.pi/bot)
          if xyk<=1:
            break
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='x_axes',
                            slice_index=l,
                            colormap='viridis',
                            vmax=Pmax,
                            vmin=Pmin
                            )
          count=0
          for l2 in range(0,Nra[i]):
            if l2==0:
              count=0
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xc=np.append(x,[Tx[0]])
            yc=np.append(y,[Tx[1]])
            zc=np.append(z,[Tx[2]])
            mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.1)
            for k in range(0, int(Ncon)):
              j=count
              x=np.array([Cones[j][0][0]])
              y=np.array([Cones[j][0][1]])
              z=np.array([Cones[j][0][2]])
              xp=np.append(x,[Cones[j][1][0]])
              yp=np.append(y,[Cones[j][1][1]])
              zp=np.append(z,[Cones[j][1][2]])
              mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.1)
              count+=1
            if l2==Nra[i]-1:
              mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.75,0.75,0.75),opacity=0.1)
              filename=str('ConeFigures/Cone'+str(iternum)+'FullSquareX'+str(int(l))+'.jpg')
              #mlab.show()
              mlab.savefig(filename,size=(1000,1000))
              mlab.clf()
        for l in range(0,Ny):
          mid=(np.cos(delangle[i])-np.sin(l*delangle[i])**2)/(np.cos(delangle[i]*l)**2)
          if abs(mid)>1:
            break
          bot=ma.acos(mid)
          xyk=int(2*np.pi/bot)
          if xyk<=1:
            break
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='y_axes',
                            slice_index=l,
                            colormap='viridis',
                            vmax=Pmax,
                            vmin=Pmin
                        )
          count=0
          for l2 in range(0,Nra[i]):
            j=count
            x=np.array([Cones[j][0][0]])
            y=np.array([Cones[j][0][1]])
            z=np.array([Cones[j][0][2]])
            xc=np.append(x,[Tx[0]])
            yc=np.append(y,[Tx[1]])
            zc=np.append(z,[Tx[2]])
            mlab.plot3d(xc,yc,zc,color= (0, 1, 1),opacity=0.1)
            for k in range(0, int(Ncon)):
              j=count
              x=np.array([Cones[j][0][0]])
              y=np.array([Cones[j][0][1]])
              z=np.array([Cones[j][0][2]])
              xp=np.append(x,[Cones[j][1][0]])
              yp=np.append(y,[Cones[j][1][1]])
              zp=np.append(z,[Cones[j][1][2]])
              mlab.plot3d(xp,yp,zp,color= (1,0,1),opacity=0.1)
              count+=1
          if l2==Nra[i]-1:
            mlab.plot3d(cubep[:,0],cubep[:,1],cubep[:,2],color=(0.5,0.5,0.5),opacity=0.1)
            filename=str('ConeFigures/Cone'+str(iternum)+'FullSquareY'+str(int(l))+'.jpg')
            #mlab.show()
            mlab.savefig(filename,size=(1000,1000))
            mlab.clf()
      else:
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='y_axes', slice_index=0,
                            colormap='viridis',
                            vmax=Pmax, vmin=Pmin
                        )
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='x_axes', slice_index=0,
                            colormap='viridis',
                            vmax=Pmax,vmin=Pmin
                        )
          mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(Powx,Powy,Powz,Power),
                            plane_orientation='z_axes', slice_index=0,
                            colormap='viridis',
                            vmax=Pmax,vmin=Pmin
                        )
          try:
              engine = mayavi.engine
          except NameError:
              from mayavi.api import Engine
              engine = Engine()
              engine.start()
          if len(engine.scenes) == 0:
              engine.new_scene()
          # -------------------------------------------
          scene = engine.scenes[0]
          scene.scene.camera.position = [8.709726813612612, 8.684071038273604, 8.679837720224306]
          scene.scene.camera.focal_point = [1.524111658260839, 1.4984558829218613, 1.4942225648724614]
          scene.scene.camera.view_angle = 30.0
          scene.scene.camera.view_up = [0.0, 0.0, 1.0]
          scene.scene.camera.clipping_range = [5.9135330467319855, 20.69871253517983]
          scene.scene.camera.compute_view_plane_normal()
          scene.scene.render()
          image_plane_widget2 = engine.scenes[0].children[1488].children[0].children[0]
          image_plane_widget2.ipw.point2 = array([-0.16666667,  3.16666667,  3.        ])
          image_plane_widget2.ipw.slice_index = 9
          image_plane_widget2.ipw.origin = array([-0.16666667, -0.16666667,  3.        ])
          image_plane_widget2.ipw.slice_position = 3.0
          image_plane_widget2.ipw.point1 = array([ 3.16666667, -0.16666667,  3.        ])
          image_plane_widget2.ipw.point2 = array([-0.16666667,  3.16666667,  3.        ])
          image_plane_widget2.ipw.origin = array([-0.16666667, -0.16666667,  3.        ])
          image_plane_widget2.ipw.point1 = array([ 3.16666667, -0.16666667,  3.        ])
          image_plane_widget1 = engine.scenes[0].children[1487].children[0].children[0]
          image_plane_widget1.ipw.point2 = array([ 3.        , -0.16666667,  3.16666667])
          image_plane_widget1.ipw.slice_index = 9
          image_plane_widget1.ipw.origin = array([ 3.        , -0.16666667, -0.16666667])
          image_plane_widget1.ipw.slice_position = 3.0
          image_plane_widget1.ipw.point1 = array([ 3.        ,  3.16666667, -0.16666667])
          image_plane_widget1.ipw.point2 = array([ 3.        , -0.16666667,  3.16666667])
          image_plane_widget1.ipw.origin = array([ 3.        , -0.16666667, -0.16666667])
          image_plane_widget1.ipw.point1 = array([ 3.        ,  3.16666667, -0.16666667])
          image_plane_widget = engine.scenes[0].children[1486].children[0].children[0]
          image_plane_widget.ipw.point2 = array([ 3.16323145, -0.10695289, -0.16666667])
          image_plane_widget.ipw.plane_orientation = 3
          image_plane_widget.ipw.origin = array([-0.16323145,  0.10695289, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.16323145,  0.10695289,  3.16666667])
          image_plane_widget.ipw.point2 = array([ 3.16323145, -0.10695289, -0.16666667])
          image_plane_widget.ipw.origin = array([-0.16323145,  0.10695289, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.16323145,  0.10695289,  3.16666667])
          image_plane_widget1.ipw.point2 = array([ 2.72003802, -0.16666667,  3.1429848 ])
          image_plane_widget1.ipw.slice_index = 0
          image_plane_widget1.ipw.plane_orientation = 3
          image_plane_widget1.ipw.origin = array([ 3.27996198, -0.16666667, -0.1429848 ])
          image_plane_widget1.ipw.slice_position = 0.0
          image_plane_widget1.ipw.point1 = array([ 3.27996198,  3.16666667, -0.1429848 ])
          image_plane_widget1.ipw.point2 = array([ 2.72003802, -0.16666667,  3.1429848 ])
          image_plane_widget1.ipw.origin = array([ 3.27996198, -0.16666667, -0.1429848 ])
          image_plane_widget1.ipw.point1 = array([ 3.27996198,  3.16666667, -0.1429848 ])
          image_plane_widget.ipw.point2 = array([ 3.16646401, -0.02598989, -0.16666667])
          image_plane_widget.ipw.origin = array([-0.16646401,  0.02598989, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.16646401,  0.02598989,  3.16666667])
          image_plane_widget.ipw.point2 = array([ 3.16646401, -0.02598989, -0.16666667])
          image_plane_widget.ipw.origin = array([-0.16646401,  0.02598989, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.16646401,  0.02598989,  3.16666667])
          scene.scene.camera.position = [10.218705996236485, 10.19305022089747, 10.188816902848195]
          scene.scene.camera.focal_point = [1.524111658260839, 1.4984558829218613, 1.4942225648724614]
          scene.scene.camera.view_angle = 30.0
          scene.scene.camera.view_up = [0.0, 0.0, 1.0]
          scene.scene.camera.clipping_range = [8.501025372481612, 23.351545576226158]
          scene.scene.camera.compute_view_plane_normal()
          scene.scene.render()
          scene.scene.camera.position = [10.014008607801468, 10.326333798987228, 10.256535809164832]
          scene.scene.camera.focal_point = [1.524111658260839, 1.4984558829218613, 1.4942225648724614]
          scene.scene.camera.view_angle = 30.0
          scene.scene.camera.view_up = [-0.4084477922716211, -0.4144272692768555, 0.8132775906590367]
          scene.scene.camera.clipping_range = [8.501152646787327, 23.35138536409761]
          scene.scene.camera.compute_view_plane_normal()
          scene.scene.render()
          image_plane_widget1.ipw.point2 = array([-0.27996198, -0.16666667,  2.59453612])
          image_plane_widget1.ipw.origin = array([ 0.27996198, -0.16666667, -0.69143349])
          image_plane_widget1.ipw.point1 = array([ 0.27996198,  3.16666667, -0.69143349])
          image_plane_widget1.ipw.point2 = array([-0.27996198, -0.16666667,  2.59453612])
          image_plane_widget1.ipw.origin = array([ 0.27996198, -0.16666667, -0.69143349])
          image_plane_widget1.ipw.point1 = array([ 0.27996198,  3.16666667, -0.69143349])
          image_plane_widget.ipw.point2 = array([ 3.16565024,  0.05819841, -0.16666667])
          image_plane_widget.ipw.origin = array([-0.16565024, -0.05819841, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.16565024, -0.05819841,  3.16666667])
          image_plane_widget.ipw.point2 = array([ 3.16565024,  0.05819841, -0.16666667])
          image_plane_widget.ipw.origin = array([-0.16565024, -0.05819841, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.16565024, -0.05819841,  3.16666667])
          image_plane_widget.ipw.point2 = array([ 3.03872615,  3.05819841, -0.16666667])
          image_plane_widget.ipw.origin = array([-0.29257433,  2.94180159, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.29257433,  2.94180159,  3.16666667])
          image_plane_widget.ipw.point2 = array([ 3.03872615,  3.05819841, -0.16666667])
          image_plane_widget.ipw.origin = array([-0.29257433,  2.94180159, -0.16666667])
          image_plane_widget.ipw.point1 = array([-0.29257433,  2.94180159,  3.16666667])

    return

def PlotDirections():

    ##Plot the obstacles and the room first

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra[0])
    Nre,h ,L    =np.load('Parameters/Raytracing.npy')
    # Take Tx to be 0,0,0
    delang     =np.load('Parameters/delangle.npy')

    mlab.points3d(0,0,0,scale_factor=0.1)
    mulfac=4
    for i in range(0,nra):
      directionname=str('Parameters/Directions'+str(int(i))+'.npy')
      data_matrix   =np.load(directionname)         # Matrix of ray directions
      delangle=delang[i]
      #mlab.savefig('ConeFigures/Rays.jpg',size=(1000,1000))
      #mlab.clf()
      #mlab.close()
      delth=2*np.arcsin(np.sqrt(2)*ma.sin(delangle/2))
      ta=np.sqrt(1-ma.tan(delth/2)**2)   # Nra>2 and an integer. Therefore tan(theta) exists.
      s=ma.sin(delth/2)
      for l in range(0,int(Nra[0,i])):
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
      filename=str('ConeFigures/Rays'+str(Nra[0,i])+'.jpg')
      mlab.savefig(filename,size=(1000,1000))
      mlab.clf()
      mlab.close()
    return

def PlotPowerSlice():
    ##----Retrieve the Raytracing Parameters-----------------------------
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')
        ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
    Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

    # Room contains all the obstacles and walls.
    Room=rom.room(Oblist)

    Nz=int(Room.maxyleng()/(h))
    Nx=int(Room.maxxleng()/(h))
    Ny=int(Room.maxyleng()/(h))
    xmin=Room.bounds[0][0]
    xmax=Room.bounds[1][0]
    ymin=Room.bounds[0][1]
    ymax=Room.bounds[1][1]
    zmin=Room.bounds[0][2]
    zmax=Room.bounds[1][2]
    # Get the x, y, z co-ordinates for P

    for i in range(0,nra):
      Power=np.load('./Mesh/Power_grid'+str(int(Nra[i]))+'Refs'+str(int(Nre))+'m'+str(0)+'.npy')
      x,y,z=np.mgrid[xmin:xmax:Nx*1.0j,ymin:ymax:Ny*1.0j,zmin:zmax:Nz*1.0j]
      mlab.volume_slice(x,y,z,Power, plane_orientation='x_axes', slice_index=10)
      #mlab.outline()
    mlab.show()
    return Power

def PlotSingleCone():
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
      data_matrix   =np.load('./Mesh/RayMeshPoints'+str(int(Nra[i]))+'Refs'+str(int(Nre))+'m.npy')
      Power=np.load('./Mesh/Power_grid'+str(int(Nra[i]))+'Refs'+str(int(Nre))+'m'+str(0)+'.npy')
      Powx,Powy,Powz=np.mgrid[xmin:xmax:Nx*1.0j,ymin:ymax:Ny*1.0j,zmin:zmax:Nz*1.0j]
      Cones=np.load('./Mesh/SingleCone'+str(int(Nra[i]))+'Refs'+str(int(Nre))+'m.npy')
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
      filename=str('ConeFigures/SingleCone'+str(int(Nra[i]))+'.jpg')
      mlab.savefig(filename,size=(1000,1000))
      #mlab.clf()
      gui = GUI()
      gui.start_event_loop()
      #mlab.close()
    #mlab.show()
  return

if __name__=='__main__':
  PlotSingleCone()
  PlotConesOnSquare()
  PlotCones()
  PlotPowerSlice()
  PlotRays()
  PlotDirections()
  print('Running  on python version')
  print(sys.version)
exit()

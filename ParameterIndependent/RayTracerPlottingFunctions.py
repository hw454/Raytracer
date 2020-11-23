#!/usr/bin/env python3
# Updated Hayley Wragg 2020-06-30
'''Code to plot heatmaps and error plots for RayTracer numpy files.
'''
import DictionarySparseMatrix as DSM
import matplotlib.pyplot as mp
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import ParameterLoad as PI
import openpyxl as wb
import numpy as np
import os
import sys
import Room as rom
epsilon=sys.float_info.epsilon

def plot_grid(plottype=str(),testnum=1,roomnumstat=0):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nsur        =np.load('Parameters/Nsur.npy')
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  numjobs    =np.load('Parameters/Numjobs.npy')
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  LOS    =np.load('Parameters/LOS.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  InnerOb   =np.load('Parameters/InnerOb.npy')
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  Roomnum=roomnumstat
  for i in range(testnum):
   for index in range(0,Roomnum):
    for j in range(0,nra):
      for job in range(0,numjobs):
        Nr=int(Nra[j])
        Nre=int(Nre)
        pstr       ='./Mesh/'+plottype+'/Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
        P   =np.load(pstr)
        RadAstr    ='./Mesh/'+plottype+'/RadA_grid%02dRefs%dm%d_tx%03d.npy'%(Nr,Nre,index,job)
        RadB=np.zeros((Nsur,P.shape[0],P.shape[1],P.shape[2]))
        ThetaMesh=np.zeros((P.shape[0],P.shape[1],P.shape[2]))
        RadA=np.load(RadAstr)
        if LOS==0:
          for k in range(0,Nsur):
            RadSistr='./Mesh/'+plottype+'/RadS%d_grid%dRefs%dm%d_tx%03d.npy'%(k,Nra[j],Nre,0,job)
            RadB[k]=np.load(RadSistr)
          #Thetastr='Mesh/'+plottype+'/AngNpy%03dRefs%03dNs%0d.npy_tx%03d.npy'%(Nra,Nre,Ns,job)
          #ThetaMesh=np.load(Thetastr).astype(float)
        n=P.shape[2]
        lb=np.amin(P)
        ub=np.amax(P)
        if LOS:
          rlb=np.amin(RadA)
          rub=np.amax(RadA)
        else:
          rlb=min(np.amin(RadA),np.amin(RadB))
          rub=max(np.amax(RadA),np.amax(RadB))
          #tlb=np.amin(ThetaMesh)
          #tub=np.amax(ThetaMesh)
        if not os.path.exists('./GeneralMethodPowerFigures'):
          os.makedirs('./GeneralMethodPowerFigures')
          os.makedirs('./GeneralMethodPowerFigures/'+plottype)
          os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job)
        if not os.path.exists('./GeneralMethodPowerFigures/'+plottype):
          os.makedirs('./GeneralMethodPowerFigures/'+plottype)
          os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job)
        if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job):
          os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job)
        for i in range(0,n):
          mp.clf()
          mp.figure(i)
          mp.imshow(P[:,:,i], cmap=cmapopt, vmax=ub,vmin=lb)
          mp.colorbar()
          rayfolder='./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/PowerSlice/Nra%03d'%(job,Nr)
          if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/PowerSlice'%job):
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/PowerSlice'%job)
            os.makedirs(rayfolder)
          elif not os.path.exists(rayfolder):
            os.makedirs(rayfolder)
          filename=rayfolder+'/'+boxstr+'PowerSliceNra%03dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,n)#.eps')
          mp.savefig(filename)
          mp.clf()
          mp.figure(i)
          mp.imshow(RadA[:,:,i], cmap=cmapopt, vmax=rub,vmin=rlb)
          mp.colorbar()
          rayfolder='./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/RadSlice/Nra%03d'%(job,Nr)
          if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/RadSlice'%job):
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/RadSlice'%job)
            os.makedirs(rayfolder)
          elif not os.path.exists(rayfolder):
            os.makedirs(rayfolder)
          filename=rayfolder+'/'+boxstr+'RadASliceNra%02dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,n)#.eps')
          mp.savefig(filename)
          mp.clf()
          if not LOS:
            for k in range(0,Nsur):
              mp.figure(i)
              mp.imshow(RadB[k,:,:,i], cmap=cmapopt, vmax=rub,vmin=rlb)
              mp.colorbar()
              filename=rayfolder+'/'+boxstr+'RadS%02dSliceNra%03dNref%03dslice%03dof%03d.jpg'%(k,Nr,Nre,i+1,n)#.eps')
              mp.savefig(filename)
              mp.clf()
            #mp.figure(i)
            #mp.imshow(ThetaMesh[:,:,i], cmap=cmapopt, vmax=tub,vmin=tlb)
            #mp.colorbar()
            # rayfolder='./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/ThetaSlice/Nra%03d'%(job,Nr)
            # if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/ThetaSlice'%job):
              # os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/ThetaSlice'%job)
              # os.makedirs(rayfolder)
            # elif not os.path.exists(rayfolder):
              # os.makedirs(rayfolder)
            # filename=rayfolder+'/'+boxstr+'ThetaSliceNra%03dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,n)#.eps')
            # mp.savefig(filename)
            mp.clf()
   Roomnum*=2
  return

def plot_residual(plottype,testnum,roomnumstat):
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  InnerOb   =np.load('Parameters/InnerOb.npy')
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  for i in range(nra):
    for k in range(testnum):
      pratstr='./Mesh/'+plottype+'/PowerRat_grid%03dRefs%03dm%03d.npy'%(Nra[i],Nre,k)
      Prat   =np.load(pratstr)
      n=Prat.shape[2]
      lb=np.amin(Prat)
      ub=np.amax(Prat)
      for j in range(0,n):
        mp.clf()
        mp.figure(j)
        mp.imshow(Prat[:,:,j], cmap=cmapopt, vmax=ub,vmin=lb)
        mp.colorbar()
        rayfolder='./GeneralMethodPowerFigures/'+plottype+'/DiffSlice/Nra%03d'%Nra[i]
        if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/DiffSlice'):
          os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/DiffSlice')
          os.makedirs(rayfolder)
        elif not os.path.exists(rayfolder):
          os.makedirs(rayfolder)
        filename=rayfolder+'/'+boxstr+'DiffSliceNra%03dNref%03dslice%03dof%03d.jpg'%(Nra[i],Nre,j+1,n)#.eps')
        mp.savefig(filename)
        mp.clf()
      errorname='./Errors/'+plottype+'/ErrorsNrays%03dRefs%03dRoomnum%03dto%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+(k)*2)
      Res=np.load(errorname)
      mp.figure(k+1)
      mp.plot(Nra,Res)
      filename='./Errors/'+plottype+'/Residual%03dto%03dNref%03d.jpg'%(Nra[0],Nra[-1],Nre)
      mp.savefig(filename)
  return

def plot_times(plottype,testnum,roomnumstat):
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  MaxInter   =np.load('Parameters/MaxInter.npy')
  Nob        =np.load('Parameters/Nob.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  for j in range(testnum):
    timename='./Times/'+plottype+'/TimesNra%03dRefs%03dRoomnum%03dto%03dMaxInter%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+(j)*2,MaxInter)
    T=np.load(timename)
    fig=mp.figure(j+1)
    ax=mp.subplot(111)
    line=ax.plot(Nra,T,label='MaxInter%03d'%MaxInter)
    timeslowname='./Times/'+plottype+'/TimesNra%03dRefs%03dRoomnum%03dto%03dMaxInter%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+(j)*2,Nob)
    if timeslowname.isfile():
      T2=np.load(timeslowname)
      line2=ax.plot(Nra,T2,label='MaxInter%03d'%Nob)
    ax.legend()
    mp.title('Time for full GRL against ray number')
    ax.set_ylabel('Number of seconds')
    ax.set_xlabel('Number of rays')
    filename='./Times/'+plottype+'/Times%03dto%03dNref%03d.jpg'%(Nra[0],Nra[-1],Nre)
    mp.savefig(filename)
  return

def plot_quality(plottype,testnum,roomnumstat,job=0):
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  truestr='Mesh/True/'+plottype+'/True.npy'
  P3=np.load(truestr)
  Q2=DSM.QualityFromPower(P3)
  for j in range(testnum):
    qualityname='./Quality/'+plottype+'/QualityNrays%03dRefs%03dRoomnum%03dto%03d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+(j)*2,job)
    Qu=np.load(qualityname)
    mp.figure(j+1)
    mp.plot(Nra,Qu)
    mp.plot(Nra,Q2)
    filename='./Quality/'+plottype+'/Quality%03dto%03dNref%03d.jpg'%(Nra[0],Nra[-1],Nre)
    mp.savefig(filename)
  return

def plot_quality_contour(plottype,testnum,roomnumstat):
  #plottype2  ='MultiRefOffCentre'
  numjobs    =np.load('Parameters/Numjobs.npy')
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  InnerOb       =np.load('Parameters/InnerOb.npy')              # Whether the innerobjects should be included or not.
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.
  Orig          =np.load('Parameters/Origin.npy')
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  if InnerOb:
    # Oblist=np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
    # Ntri=np.append(NtriOb,NtriOut)
    Box='Box'
  else:
    Box='NoBox'
    # Oblist=OuterBoundary
    # Ntri=NtriOut
  #Room contains all the obstacles and walls.
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Room.__set_MaxInter__(MaxInter)
  Nsur=int(Room.Nsur)
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  roomnummax=roomnumstat
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  for j in range(testnum):
    for i in range(nra):
      Nr=int(Nra[i])
      Mesh=DSM.DS(Nx,Ny,Nz,Nsur*int(Nre)+1,Nr*(int(Nre)+1),np.complex128,int(split))
      rom.FindInnerPoints(Room,Mesh,Orig)
      Nr=Nra[i]
      pstr       ='./Mesh/'+plottype+'/'+Box+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,0)
      P   =np.load(pstr)
      Nx=P.shape[0]
      Ny=P.shape[1]
      Nz=P.shape[2]
      Qmesh=-50*np.ones((Nx,Ny,Nz))
      Q2mesh=-50*np.ones((Nx,Ny,Nz))
      QP2mesh=-50*np.ones((Nx,Ny,Nz))
      QPmesh=-50*np.ones((Nx,Ny,Nz))
      QMmesh=-50*np.ones((Nx,Ny,Nz))
      QM2mesh=-50*np.ones((Nx,Ny,Nz))
      Tmesh=100*np.ones((Nx,Ny,Nz))
      T2mesh=100*np.ones((Nx,Ny,Nz))
      feature_x = L*np.linspace(0.0,Nx*h,Nx)
      feature_y = L*np.linspace(0.0,Ny*h,Ny)
      [X, Y] = np.meshgrid(feature_y,feature_x )
      # QPmesh =np.load('./Mesh/'+plottype+'/QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      # Qmesh  =np.load('./Mesh/'+plottype+'/QualityAverage%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      # QP2mesh=np.load('./Mesh/'+plottype2+'/QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      # Q2mesh =np.load('./Mesh/'+plottype2+'/QualityAverage%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      # Tmesh  =np.load('./Mesh/'+plottype+'/TimeFullCalc%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      for job in range(numjobs):
       Tx=np.load('Parameters/Origin_job%03d.npy'%job)
       Txind=Room.position(Tx,h)
       if not Room.CheckTxInner(Tx):
          Qmesh[Txind] =np.nan
          QPmesh[Txind]=np.nan
          QMmesh[Txind]=np.nan
          Tmesh[Txind] =np.nan
       if job<numjobs+1:# and job!=35:# and job!=36 and job!=37 and job!=38:
          print('job',job)
          print('Tx',Tx,Txind)
          qualityname='./Quality/'+plottype+'/'+Box+'Quality%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          Qu=np.load(qualityname)
          Qmesh[Txind]=Qu
          #qualityname='./Quality/'+plottype2+'/'+Box+'Quality%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          #Qu2=np.load(qualityname)
          #Q2mesh[Txind]=Qu2
          qualityPname='./Quality/'+plottype+'/'+Box+'QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          QuP=np.load(qualityPname)
          QPmesh[Txind]=QuP
          #qualityPname='./Quality/'+plottype2+'/'+Box+'QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          #QuP2=np.load(qualityPname)
          #QP2mesh[Txind]=QuP2
          qualityMname='./Quality/'+plottype+'/'+Box+'QualityMin%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          QuM=np.load(qualityMname)
          QMmesh[Txind]=QuM
          #qualityMname='./Quality/'+plottype2+'/'+Box+'QualityMin%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          #QuM2=np.load(qualityMname)
          #QM2mesh[Txind]=QuM2
          Ti=np.load('./Times/'+plottype+'/'+Box+'TimesNra%03dRefs%03dRoomnum%dto%03dMaxInter%d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnummax,MaxInter,job))
          Tmesh[Txind]=Ti
          #Ti2=np.load('./Times/'+plottype2+'/TimesNra%03dRefs%03dRoomnum%dto%03dMaxInter%d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnummax,MaxInter,job))
          #T2mesh[Txind]=Ti2
      Qmin=min(np.amin(Qmesh),np.amin(QPmesh),np.amin(QMmesh))
      Qmax=max(np.amax(Qmesh),np.amax(QPmesh),np.amax(QMmesh))
      Tmin=np.amin(Tmesh)
      Tmax=np.amax(Tmesh)
      norm=matplotlib.colors.Normalize(vmin=Qmin, vmax=Qmax)
      normt=matplotlib.colors.Normalize(vmin=Tmin, vmax=Tmax)
      print('Q ave range',np.amin(Qmesh),np.amax(Qmesh))
      print('QP range',np.amin(QPmesh),np.amax(QPmesh))
      print('Q min range', np.amin(QMmesh),np.amax(QMmesh))
      print('Time range',Tmin,Tmax)
      np.save('./Mesh/'+plottype+'/'+Box+'QualityAverage%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0),Qmesh)
      np.save('./Mesh/'+plottype+'/'+Box+'QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0),QPmesh)
      np.save('./Mesh/'+plottype+'/'+Box+'QualityMin%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0),QMmesh)
      for k in range(0,Nz):
        fig = mp.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylim(0, L*Nx*h)
        ax.set_xlim(0, L*Ny*h)
        ax.set_zlim(Qmin, Qmax)
        ax.plot_surface(X=X,Y=Y,Z=Qmesh[:,:,k],vmin=Qmin,vmax=Qmax,cmap=cmapopt)
        fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        ax.set_title('Quality of coverage by transmitter location, z=%02d'%k)
        ax.set_xlabel('Y position of Tx')
        ax.set_ylabel('X position of Tx')
        filename='./Quality/'+plottype+'/'+Box+'Qualitysurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf()
        mp.close()
        # fig = mp.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X=X,Y=Y,Z=abs(Qmesh[:,:,k]-Q2mesh[:,:,k]),vmin=0,vmax=1,cmap=cmapopt)
        # #fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        # ax.set_title('Quality Ref change of coverage by transmitter location, z=%02d'%k)
        # filename='./Quality/'+plottype+'/QualityBothsurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        # mp.savefig(filename)
        # mp.clf()
        fig = mp.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylim(0, L*Nx*h)
        ax.set_xlim(0, L*Ny*h)
        ax.set_zlim(Qmin, Qmax)
        ax.plot_surface(X,Y,QPmesh[:,:,k],vmin=Qmin,vmax=Qmax,cmap=cmapopt)
        fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        ax.set_title('Quality Percentile of coverage by transmitter location z=%02d'%k)
        ax.set_xlabel('Y position of Tx')
        ax.set_ylabel('X position of Tx')
        filename='./Quality/'+plottype+'/'+Box+'QualityPercentileSurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf()
        mp.close()
        fig = mp.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylim(0, L*Nx*h)
        ax.set_xlim(0, L*Ny*h)
        ax.set_zlim(Qmin, Qmax)
        ax.plot_surface(X,Y,QMmesh[:,:,k],vmin=Qmin,vmax=Qmax,cmap=cmapopt)
        fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        ax.set_title('Quality (min) of coverage by transmitter location z=%02d'%k)
        ax.set_xlabel('Y position of Tx')
        ax.set_ylabel('X position of Tx')
        filename='./Quality/'+plottype+'/'+Box+'QualityMinSurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf()
        mp.close()
        # fig = mp.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X,Y,Z=abs(QP2mesh[:,:,k]-QPmesh[:,:,k]),vmin=0,vmax=1,cmap=cmapopt)
        # #fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        # ax.set_title('Quality Percentile Ref change of coverage by transmitter location z=%02d'%k)
        # filename='./Quality/'+plottype+'/QualityPercentileBothSurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        # mp.savefig(filename)
        # mp.clf()
        fig, ax = mp.subplots(1, 1)
        ax.contourf(X,Y,Qmesh[:,:,k])
        fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        ax.set_title('Quality of coverage by transmitter location z=%02d'%k)
        ax.set_xlabel('X position of Tx')
        ax.set_ylabel('Y position of Tx')
        filename='./Quality/'+plottype+'/'+Box+'QualityContour%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf()
        mp.close()
        fig, ax = mp.subplots(1, 1)
        ax.contourf(X,Y,QPmesh[:,:,k])
        fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        ax.set_title('Quality Percentile of coverage by transmitter location z=%02d'%k)
        ax.set_xlabel('X position of Tx')
        ax.set_ylabel('Y position of Tx')
        filename='./Quality/'+plottype+'/'+Box+'QualityPercentileContour%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf()
        mp.close()
        fig, ax = mp.subplots(1, 1)
        ax.contourf(X,Y,QMmesh[:,:,k])
        fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
        ax.set_title('Quality (min) of coverage by transmitter location z=%02d'%k)
        ax.set_xlabel('X position of Tx')
        ax.set_ylabel('Y position of Tx')
        filename='./Quality/'+plottype+'/'+Box+'QualityMinContour%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf
        mp.close()
        fig, ax = mp.subplots(1, 1)
        ax.contourf(X,Y,Tmesh[:,:,k])
        fig.colorbar(mp.cm.ScalarMappable(norm=normt, cmap=cmapopt), ax=ax)
        ax.set_title('Time for calculation by transmitter location z=%02d'%k)
        ax.set_xlabel('X position of Tx')
        ax.set_ylabel('Y position of Tx')
        filename='./Times/'+plottype+'/'+Box+'TimeContour%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf()
        mp.close()
        fig = mp.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylim(0, L*Nx*h)
        ax.set_xlim(0, L*Ny*h)
        ax.set_zlim(Tmin, Tmax)
        ax.plot_surface(X,Y,Tmesh[:,:,k],vmin=Tmin,vmax=Tmax,cmap=cmapopt)
        fig.colorbar(mp.cm.ScalarMappable(norm=normt, cmap=cmapopt), ax=ax)
        ax.set_title('Time for calculation by transmitter location z=%02d'%k)
        ax.set_xlabel('Y position of Tx')
        ax.set_ylabel('X position of Tx')
        filename='./Times/'+plottype+'/'+Box+'TimeSurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        mp.savefig(filename)
        mp.clf()
        mp.close()
        # fig, ax = mp.subplots(1, 1)
        # ax.contourf(X,Y,T2mesh[:,:,k])
        # fig.colorbar(mp.cm.ScalarMappable(norm=normt, cmap=cmapopt), ax=ax)
        # ax.set_title('Time for calculation by transmitter location z=%02d'%k)
        # ax.set_xlabel('X position of Tx')
        # ax.set_ylabel('Y position of Tx')
        # filename='./Times/'+plottype2+'/TimeContour%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        # mp.savefig(filename)
        # mp.clf()
        # fig = mp.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_ylim(0, L*Nx*h)
        # ax.set_xlim(0, L*Ny*h)
        # ax.set_zlim(Tmin, Tmax)
        # ax.plot_surface(X,Y,T2mesh[:,:,k],vmin=Tmin,vmax=Tmax,cmap=cmapopt)
        # fig.colorbar(mp.cm.ScalarMappable(norm=normt, cmap=cmapopt), ax=ax)
        # ax.set_title('Time for calculation by transmitter location z=%02d'%k)
        # ax.set_xlabel('Y position of Tx')
        # ax.set_ylabel('X position of Tx')
        # filename='./Times/'+plottype2+'/TimeSurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        # mp.savefig(filename)
        # mp.clf()
        # fig = mp.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_ylim(0, L*Nx*h)
        # ax.set_xlim(0, L*Ny*h)
        # ax.set_zlim(Tmin, Tmax)
        # ax.plot_surface(X,Y,T2mesh[:,:,k],vmin=Tmin,vmax=Tmax,cmap=cmapopt)
        # ax.plot_surface(X,Y,Tmesh[:,:,k],vmin=Tmin,vmax=Tmax,cmap=cmapopt)
        # fig.colorbar(mp.cm.ScalarMappable(norm=normt, cmap=cmapopt), ax=ax)
        # ax.set_title('Time for calculation, Both, by transmitter location z=%02d'%k)
        # ax.set_xlabel('Y position of Tx')
        # ax.set_ylabel('X position of Tx')
        # filename='./Times/'+plottype+'/TimeBothSurface%03dto%03dNref%03d_z%02d.jpg'%(Nra[0],Nra[-1],Nre,k)
        # mp.savefig(filename)
        # mp.clf()
        mp.close()
  return

if __name__=='__main__':
  Sheetname='InputSheet.xlsx'
  out=PI.DeclareParameters(Sheetname)
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  InBook     =wb.load_workbook(filename=Sheetname,data_only=True)
  SimParstr  ='SimulationParameters'
  SimPar     =InBook[SimParstr]
  testnum    =SimPar.cell(row=17,column=3).value
  roomnumstat=SimPar.cell(row=18,column=3).value
  #plot_times(plottype,testnum,roomnumstat)
  #plot_grid(plottype,testnum,roomnumstat)        # Plot the power in slices.
  plot_quality_contour(plottype,testnum,roomnumstat)
  ResOn      =np.load('Parameters/ResOn.npy')
  if ResOn:
    plot_residual(plottype,testnum,roomnumstat)
  exit()

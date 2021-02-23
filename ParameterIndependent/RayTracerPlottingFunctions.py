#!/usr/bin/env python3
# Updated Hayley Wragg 2020-06-30
'''Code to plot heatmaps and error plots for RayTracer numpy files.
'''
import DictionarySparseMatrix as DSM
import RayTracerMainProgram as RTM
import matplotlib.pyplot as mp
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import ParameterLoad as PI
import openpyxl as wb
import numpy as np
import os
import sys
import Room as rom
import num2words as nw
epsilon=sys.float_info.epsilon
xcheck=2
ycheck=4
zcheck=9

def plot_grid(plottype=str(),testnum=1,roomnumstat=0):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nsur        =np.load('Parameters/Nsur.npy')
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nre=4
  Nra        =np.load('Parameters/Nra.npy')
  numjobs    =np.load('Parameters/Numjobs.npy')
  Nrs        =np.load('Parameters/Nrs.npy')
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
  plotfit= myfile.read()         # read the entire file into a string
  myfile.close()
  LOS    =np.load('Parameters/LOS.npy')
  PerfRef=np.load('Parameters/PerfRef.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  InnerOb   =np.load('Parameters/InnerOb.npy')
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      #if Nrs<nsur:
        #LOSstr=nw.num2words(Nrs)+'Ref'
      #else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      #if Nrs<nsur:
        #LOSstr=nw.num2words(Nrs)+'Ref'
      #else:
      LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  Roomnum=roomnumstat
  numjobs=500
  for i in range(testnum):
   for index in range(0,Roomnum):
    for j in range(0,nra):
      for job in range(0,numjobs+1):
        Nr=int(Nra[j])
        Nre=int(Nre)
        Txstr='Parameters/Origin_job%03d.npy'%job
        if os.path.isfile(Txstr):
          Tx=np.load(Txstr)
          if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
            loca='Centre'
          else:
            loca='OffCentre'
          plottype=LOSstr+boxstr+loca
          plottype='MultiPerfRefNoBoxCentre'
          LOS=0
          pstr       ='./Mesh/'+plottype+'/'+boxstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
          meshname='./Mesh/'+plottype+'/DSM%03dRefs%03d_tx%03d.npy'%(Nr,Nre,job)
          if os.path.isfile(pstr):
            Mesh= DSM.load_dict(meshname)
            P   =np.load(pstr)
            print(pstr)
            RadAstr    ='./Mesh/'+plottype+'/'+boxstr+'RadA_grid%02dRefs%dm%d_tx%03d.npy'%(Nr,Nre,index,job)
            RadB=np.zeros((Nsur,P.shape[0],P.shape[1],P.shape[2]))
            ThetaMesh=np.zeros((P.shape[0],P.shape[1],P.shape[2]))
            RadA=np.load(RadAstr)
            if LOS==0:
              for k in range(0,Nsur):
                RadSistr='./Mesh/'+plottype+'/'+boxstr+'RadS%d_grid%dRefs%dm%d_tx%03d.npy'%(k,Nra[j],Nre,0,job)
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
              mp.savefig(filename,bbox_inches=plotfit)
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
              mp.savefig(filename,bbox_inches=plotfit)
              mp.clf()
              if not LOS:
                for k in range(0,Nsur):
                  mp.figure(i)
                  mp.imshow(RadB[k,:,:,i], cmap=cmapopt, vmax=rub,vmin=rlb)
                  mp.colorbar()
                  filename=rayfolder+'/'+boxstr+'RadS%02dSliceNra%03dNref%03dslice%03dof%03d.jpg'%(k,Nr,Nre,i+1,n)#.eps')
                  mp.savefig(filename,bbox_inches=plotfit)
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
                # mp.savefig(filename,bbox_inches=plotfit)
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
  myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
  plotfit= myfile.read()         # read the entire file into a string
  myfile.close()
  NTx=np.load('Parameters/Numjobs.npy')
  for job in range(NTx+1):
   for i in range(nra):
    Nr=int(Nra[i])
    for k in range(testnum):
      pratstr='./Mesh/'+plottype+'/'+boxstr+'PowerRes_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,k,job)
      if os.path.isfile(pratstr):
        Prat   =np.load(pratstr)
      else:
        RTM.Residual(plottype,boxstr,roomnumstat+(k)*2,job)
        if os.path.isfile(pratstr):
          Prat=np.load(pratstr)
          n=Prat.shape[2]
          lb=np.amin(Prat)
          ub=np.amax(Prat)
        else:
          continue
      for j in range(0,n):
          mp.clf()
          mp.figure(j)
          mp.imshow(Prat[:,:,j], cmap=cmapopt, vmax=ub,vmin=lb)
          mp.colorbar()
          rayfolder='./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/DiffSlice/Nra%03d'%(job,Nr)
          if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job):
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job)
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/DiffSlice'%job)
            os.makedirs(rayfolder)
          elif not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job):
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/DiffSlice'%job)
            os.makedirs(rayfolder)
          elif not os.path.exists(rayfolder):
            os.makedirs(rayfolder)
          filename=rayfolder+'/'+boxstr+'DiffSliceNra%03dNref%03dslice%03dof%03d_tx%03d.jpg'%(Nr,Nre,j+1,n,job)#.eps')
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
      errorname='./Errors/'+plottype+'/ErrorsNrays%03dRefs%03dRoomnum%03dto%03d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+(k)*2,job)
      if os.path.isfile(errorname):
        Res=np.load(errorname)
        mp.figure(k+1)
        mp.plot(Nra,Res)
        filename='./Errors/'+plottype+'/Residual%03dto%03dNref%03d_tx%03d.jpg'%(Nra[0],Nra[-1],Nre,job)
        mp.savefig(filename,bbox_inches=plotfit)
  return

def plot_times(plottype,testnum,roomnumstat):
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  MaxInter   =np.load('Parameters/MaxInter.npy')
  Nob        =np.load('Parameters/Nob.npy')
  myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
  plotfit= myfile.read()         # read the entire file into a string
  myfile.close()
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
    mp.savefig(filename,bbox_inches=plotfit)
  return

def plot_quality(plottype,testnum,roomnumstat,job=0):
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
  plotfit= myfile.read()         # read the entire file into a string
  myfile.close()
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
    mp.savefig(filename,bbox_inches=plotfit)
  return

def plot_quality_contour(plottype,testnum,roomnumstat):
  numjobs    =np.load('Parameters/Numjobs.npy')
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  Oblist        =np.load('Parameters/Obstacles.npy').astype(float)      # The obstacles which are within the outerboundary
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  InnerOb       =np.load('Parameters/InnerOb.npy')              # Whether the innerobjects should be included or not.
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.
  MaxInter=3
  Orig          =np.load('Parameters/Origin.npy')
  LOS           =np.load('Parameters/LOS.npy')
  PerfRef       =np.load('Parameters/PerfRef.npy')
  Nrs           =np.load('Parameters/Nrs.npy')
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
  plotfit= myfile.read()         # read the entire file into a string
  myfile.close()

  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      #if Nrs<nsur:
        #LOSstr=nw.num2words(Nrs)+'Ref'
      #else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      #if Nrs<nsur:
        #LOSstr=nw.num2words(Nrs)+'Ref'
      #else:
      LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  if InnerOb:
    Box='Box'
  else:
    Box='NoBox'
  foldtype=LOSstr+Box
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
      rom.FindInnerPoints(Room,Mesh)
      Nr=Nra[i]
      #pstr       ='./Mesh/'+plottype+'/'+Box+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,432)
      #P   =np.load(pstr)
      #Nx=P.shape[0]
      #Ny=P.shape[1]
      #Nz=P.shape[2]
      Nx=5
      Ny=5
      Nz=5
      Qmesh=-50*np.ones((Nx,Ny,Nz))
      Q2mesh=-50*np.ones((Nx,Ny,Nz))
      QP2mesh=-50*np.ones((Nx,Ny,Nz))
      QPmesh=-50*np.ones((Nx,Ny,Nz))
      QMmesh=-50*np.ones((Nx,Ny,Nz))
      QM2mesh=-50*np.ones((Nx,Ny,Nz))
      Tmesh=100*np.ones((Nx,Ny,Nz))
      T2mesh=100*np.ones((Nx,Ny,Nz))
      print(Nx,Ny,Nz)
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
       if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
         loca='Centre'
       else:
         loca='OffCentre'
       plottype=LOSstr+Box+loca
       Txind=Room.position(Tx,h)
       if not Room.CheckTxInner(Tx):
          if any(t>1.0 or t<0 for t in Tx):
            print('Tx outside room',Tx)
            continue
          else:
            print('Tx inside an obstacle',Tx)
            Qmesh[Txind] =np.nan
            QPmesh[Txind]=np.nan
            QMmesh[Txind]=np.nan
            Tmesh[Txind] =np.nan
       if job<numjobs+1:# and job!=35:# and job!=36 and job!=37 and job!=38:
          print('job',job)
          print('Tx',Tx,Txind)
          Boxstr=Box
          qualityname='./Quality/'+plottype+'/'+Boxstr+'Quality%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          if os.path.isfile(qualityname):
            Qu=np.load(qualityname)
            Qmesh[Txind]=Qu
          else:
            print('Quality Average not found')
            print(qualityname)
          qualityPname='./Quality/'+plottype+'/'+Boxstr+'QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          if os.path.isfile(qualityPname):
            QuP=np.load(qualityPname)
            QPmesh[Txind]=QuP
          else:
            print('Quality Percentile not found')
            print(qualityname)
          qualityMname='./Quality/'+plottype+'/'+Boxstr+'QualityMin%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,job)
          if os.path.isfile(qualityMname):
            QuM=np.load(qualityMname)
            QMmesh[Txind]=QuM
          else:
            print('Quality Min not found')
            print(qualityname)
          Tname='./Times/'+plottype+'/'+Boxstr+'TimesNra%03dRefs%03dRoomnum%dto%03dMaxInter%d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnummax,MaxInter,job)
          if os.path.isfile(Tname):
            Ti=np.load(Tname)
            Tmesh[Txind]=Ti
          else:
            print('times not found')
            print(qualityname)
      if numjobs>0:
        Qmin=min(np.amin(Qmesh),np.amin(QPmesh),np.amin(QMmesh))
        Qmax=max(np.amax(Qmesh),np.amax(QPmesh),np.amax(QMmesh))
        if Qmin==Qmax:
          print('Q is the same everywhere')
          print(Qmesh)
        else:
          Tmin=np.amin(Tmesh)
          Tmax=np.amax(Tmesh)
          norm=matplotlib.colors.Normalize(vmin=Qmin, vmax=Qmax)
          normt=matplotlib.colors.Normalize(vmin=Tmin, vmax=Tmax)
          print('Q ave range',np.amin(Qmesh),np.amax(Qmesh))
          print('QP range',np.amin(QPmesh),np.amax(QPmesh))
          print('Q min range', np.amin(QMmesh),np.amax(QMmesh))
          print('Time range',Tmin,Tmax)
          if not os.path.exists('./Mesh'):
            os.mkdir('./Mesh')
            os.mkdir('./Mesh/'+foldtype)
          if not os.path.exists('./Mesh/'+foldtype):
            os.mkdir('./Mesh/'+foldtype)
          if not os.path.exists('./Quality'):
            os.mkdir('./Quality')
            os.mkdir('./Quality/'+foldtype)
          if not os.path.exists('./Quality/'+foldtype):
            os.mkdir('./Quality/'+foldtype)
          np.save('./Mesh/'+foldtype+'/'+Box+'QualityAverage%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0),Qmesh)
          np.save('./Mesh/'+foldtype+'/'+Box+'QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0),QPmesh)
          np.save('./Mesh/'+foldtype+'/'+Box+'QualityMin%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0),QMmesh)
          text_file = open('Quality/'+foldtype+'/OptimalLocationAverage%03dRefs%03dm%03d.txt'%(Nr,Nre,0), 'w')
          n = text_file.write('Optimal transmitter location at ')
          n = text_file.write(str(np.argmax(Qmesh)))
          text_file.close()
          text_file = open('Quality/'+foldtype+'/OptimalLocationPercentile%03dRefs%03dm%03d.txt'%(Nr,Nre,0), 'w')
          n = text_file.write('Optimal transmitter location at ')
          n = text_file.write(str(np.argmax(QPmesh)))
          text_file.close()
          text_file = open('Quality/'+foldtype+'/OptimalLocationMinimum%03dRefs%03dm%03d.txt'%(Nr,Nre,0), 'w')
          n = text_file.write('Optimal transmitter location at ')
          n = text_file.write(str(np.argmax(QMmesh)))
          text_file.close()
          for k in range(0,Nz):
            fig = mp.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_ylim(0, L*Nx*h)
            ax.set_xlim(0, L*Ny*h)
            ax.set_zlim(Qmin, Qmax)
            ax.plot_surface(X=X,Y=Y,Z=Qmesh[:,:,k],vmin=Qmin,vmax=Qmax,cmap=cmapopt)
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality of coverage by transmitter location, z=%02f'%(k*h+h/2))
            ax.set_xlabel('Y position of Tx')
            ax.set_ylabel('X position of Tx')
            filename='./Quality/'+foldtype+'/'+Box+'Qualitysurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig = mp.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_ylim(0, L*Nx*h)
            ax.set_xlim(0, L*Ny*h)
            ax.set_zlim(Qmin, Qmax)
            ax.plot_surface(X,Y,QPmesh[:,:,k],vmin=Qmin,vmax=Qmax,cmap=cmapopt)
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality Percentile of coverage by transmitter location z=%02f'%(k*h+h/2))
            ax.set_xlabel('Y position of Tx')
            ax.set_ylabel('X position of Tx')
            filename='./Quality/'+foldtype+'/'+Box+'QualityPercentileSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig = mp.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_ylim(0, L*Nx*h)
            ax.set_xlim(0, L*Ny*h)
            ax.set_zlim(Qmin, Qmax)
            ax.plot_surface(X,Y,QMmesh[:,:,k],vmin=Qmin,vmax=Qmax,cmap=cmapopt)
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality (min) of coverage by transmitter location z=%02f'%(k*h+h/2))
            ax.set_xlabel('Y position of Tx')
            ax.set_ylabel('X position of Tx')
            filename='./Quality/'+foldtype+'/'+Box+'QualityMinSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig, ax = mp.subplots(1, 1)
            ax.contourf(X,Y,Qmesh[:,:,k])
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality of coverage by transmitter location z=%02f'%(k*h+h/2))
            ax.set_xlabel('X position of Tx')
            ax.set_ylabel('Y position of Tx')
            filename='./Quality/'+foldtype+'/'+Box+'QualityContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig, ax = mp.subplots(1, 1)
            ax.contourf(X,Y,QPmesh[:,:,k])
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality Percentile of coverage by transmitter location z=%02f'%(k*h+h/2))
            ax.set_xlabel('X position of Tx')
            ax.set_ylabel('Y position of Tx')
            filename='./Quality/'+foldtype+'/'+Box+'QualityPercentileContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig, ax = mp.subplots(1, 1)
            ax.contourf(X,Y,QMmesh[:,:,k])
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality (min) of coverage by transmitter location z=%02d'%(k*h+h/2))
            ax.set_xlabel('X position of Tx')
            ax.set_ylabel('Y position of Tx')
            filename='./Quality/'+foldtype+'/'+Box+'QualityMinContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf
            mp.close()
            fig, ax = mp.subplots(1, 1)
            ax.contourf(X,Y,Tmesh[:,:,k])
            fig.colorbar(mp.cm.ScalarMappable(norm=normt, cmap=cmapopt), ax=ax)
            ax.set_title('Time for calculation by transmitter location z=%02d'%(k*h+h/2))
            ax.set_xlabel('X position of Tx')
            ax.set_ylabel('Y position of Tx')
            if not os.path.exists('./Times'):
              os.mkdir('./Times')
              os.mkdir('./Times/'+foldtype)
            if not os.path.exists('./Times/'+foldtype):
              os.mkdir('./Times/'+foldtype)
            filename='./Times/'+foldtype+'/'+Box+'TimeContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig = mp.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_ylim(0, L*Nx*h)
            ax.set_xlim(0, L*Ny*h)
            ax.set_zlim(Tmin, Tmax)
            ax.plot_surface(X,Y,Tmesh[:,:,k],vmin=Tmin,vmax=Tmax,cmap=cmapopt)
            fig.colorbar(mp.cm.ScalarMappable(norm=normt, cmap=cmapopt), ax=ax)
            ax.set_title('Time for calculation by transmitter location z=%02f'%(k*h+h/2))
            ax.set_xlabel('Y position of Tx')
            ax.set_ylabel('X position of Tx')
            filename='./Times/'+foldtype+'/'+Box+'TimeSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
  return

def main():
  Sheetname='InputSheet.xlsx'
  out=PI.DeclareParameters(Sheetname)
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  InnerOb=np.load('Parameters/InnerOb.npy')
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  InBook     =wb.load_workbook(filename=Sheetname,data_only=True)
  SimParstr  ='SimulationParameters'
  SimPar     =InBook[SimParstr]
  testnum    =SimPar.cell(row=17,column=3).value
  roomnumstat=SimPar.cell(row=18,column=3).value
  #plot_times(plottype,testnum,roomnumstat)
  plot_grid(plottype,testnum,roomnumstat)        # Plot the power in slices.
  #plot_quality_contour(plottype,testnum,roomnumstat)
  ResOn      =np.load('Parameters/ResOn.npy')
  if ResOn:
    plot_residual(plottype,testnum,roomnumstat)
  return

if __name__=='__main__':
  main()
  exit()

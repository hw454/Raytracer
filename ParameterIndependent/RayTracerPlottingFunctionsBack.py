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
import math as ma
import os
import sys
import Room as rom
import num2words as nw
from itertools import product
epsilon=sys.float_info.epsilon
xcheck=2
ycheck=4
zcheck=9

def plot_grid(InnerOb,Nr,Nrs,LOS,Nre,PerfRef,Ns,Q,Par,index):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  numjobs    =np.load('Parameters/Numjobs.npy')
  if Ns==5:
    numjobs=126
  else:
    numjobs=500
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  _,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nra           =np.load('Parameters/Nra.npy')
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.
  Orig          =np.load('Parameters/Origin.npy')
  Nrs           =np.load('Parameters/Nrs.npy')
  Nsur          =np.load('Parameters/Nsur%d.npy'%index)
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles

  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
  plotfit= myfile.read()         # read the entire file into a string
  myfile.close()

  if Nre>1:
    Refstr=nw.num2words(Nre)+'Ref'
  else:
    Refstr='NoRef'
  roomnummax=roomnumstat
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index).astype(float)      # The obstacles which are within the outerboundary
  Nsur       =np.load('Parameters/Nsur%d.npy'%index)
  refindex=np.load('Parameters/refindex%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  obnumbers=np.zeros((Nrs,1))
  k=0
  Obstr=''
  if 0<Nrs<Nsur:
    obnumbers=np.zeros(Nrs)
    for ob, refind in enumerate(refindex):
      if abs(refind)>epsilon:
        obnumbers[k]=ob
        k+=1
        Obstr=Obstr+'Ob%02d'%ob
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'PerfRef'
      else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'Ref'
      else:
        LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  foldtype=Refstr+boxstr
  #Room contains all the obstacles and walls.
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Nsur=int(Room.Nsur)
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  Ns=max(Nx,Ny,Nz)
  Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1. Repeat Znobrat to match Mesh dimensions
  Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
  refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nsur*Nre+1. Repeat refindex to match Mesh dimensions
  refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
  # Calculate the necessry parameters for the power calculation.
  for j in range(0,nra):
   Nr=int(Nra[j])
   gainname      ='Parameters/Tx%03dGains%03d.npy'%(Nr,index)
   Gt            = np.load(gainname)
   for job in range(0,numjobs+1):
      if Ns==10:
        if not job==444 and not job==432:
          continue
      else:
        Tx=RTM.MoveTx(job,Nx,Ny,Nz,h)
        if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
          loca='Centre'
        else:
           loca='OffCentre'
        foldtype=Refstr+boxstr
        plottype=LOSstr+boxstr+loca
        meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
        powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
        pstr       =powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
        if os.path.isfile(pstr):
          print('Plotting power at Tx=',Tx)
          P   =np.load(pstr)
        else:
          meshname=meshfolder+'/DSM_tx%03d'%(job)
          mesheg=meshname+'%02dx%02dy%02dz.npz'%(0,0,0)
          if os.path.isfile(mesheg):
            Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
            P,ind=DSM.power_compute(foldtype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,Ns,LOS,PerfRef)
          else:
            continue
        RadAstr    =meshfolder+'/'+boxstr+'RadA_grid%02dRefs%dm%d_tx%03d.npy'%(Nr,Nre,index,job)
        RadA=np.zeros((P.shape[0],P.shape[1],P.shape[2]))
        RadB=np.zeros((Nsur,P.shape[0],P.shape[1],P.shape[2]))
        ThetaMesh=np.zeros((P.shape[0],P.shape[1],P.shape[2]))
        if os.path.isfile(RadAstr):
          RadA=np.load(RadAstr)
        if not LOS:
          for k in range(0,Nsur):
            RadSistr=meshfolder+'/'+boxstr+'RadS%d_grid%dRefs%dm%d_tx%03d.npy'%(k,Nra[j],Nre,0,job)
            if os.path.isfile(RadSistr):
              RadB[k]=np.load(RadSistr)
              Thetastr=powerfolder+'/'+boxstr+'AngNpy%03dRefs%03dNs%03d_tx%03d.npy'%(Nra[j],Nre,Ns,job)
            if os.path.isfile(Thetastr):
              ThetaMesh=np.load(Thetastr).astype(float)
          rlb=min(np.amin(RadA),np.amin(RadB))
          rub=max(np.amax(RadA),np.amax(RadB))
          tlb=np.amin(ThetaMesh)
          tub=np.amax(ThetaMesh)
        else:
          rlb=np.amin(RadA)
          rub=np.amax(RadA)
        n=P.shape[2]
        lb=np.amin(P)
        ub=np.amax(P)
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
          filename=rayfolder+'/'+boxstr+Obstr+'PowerSliceNra%03dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,n)#.eps')
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
          if not LOS:
            mp.figure(i)
            mp.imshow(ThetaMesh[:,:,i], cmap=cmapopt, vmax=tub,vmin=tlb)
            mp.colorbar()
            rayfolder='./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/ThetaSlice/Nra%03d'%(job,Nr)
            if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/ThetaSlice'%job):
              os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/ThetaSlice'%job)
              os.makedirs(rayfolder)
            elif not os.path.exists(rayfolder):
              os.makedirs(rayfolder)
            filename=rayfolder+'/'+boxstr+Obstr+'ThetaSliceNra%02dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,n)#.eps')
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
          filename=rayfolder+'/'+boxstr+Obstr+'RadASliceNra%02dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,n)#.eps')
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
          if not LOS:
            for k in range(0,Nsur):
              mp.figure(i)
              mp.imshow(RadB[k,:,:,i], cmap=cmapopt, vmax=rub,vmin=rlb)
              mp.colorbar()
              filename=rayfolder+'/'+boxstr+Obstr+'RadS%02dSliceNra%03dNref%03dslice%03dof%03d.jpg'%(k,Nr,Nre,i+1,n)#.eps')
              mp.savefig(filename,bbox_inches=plotfit)
              mp.clf()
              mp.clf()
  return

def plot_mesh(Mesh,Room,Tx,foldtype,plottype,boxstr,Obstr,Nr,Nre,Ns,plotfit,LOS=0,index=0):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nsur=Room.Nsur
  h=Room.get_meshwidth(Mesh)
  job=RTM.jobfromTx(Tx,h)
  meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
  pstr       =meshfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
  if os.path.isfile(pstr):
    P   =np.load(pstr)
  else: return
  RadAstr    =meshfolder+'/'+boxstr+'RadA_grid%02dRefs%dm%d_tx%03d.npy'%(Nr,Nre,index,job)
  RadA=np.zeros((P.shape[0],P.shape[1],P.shape[2]))
  RadB=np.zeros((Nsur,P.shape[0],P.shape[1],P.shape[2]))
  ThetaMesh=np.zeros((P.shape[0],P.shape[1],P.shape[2]))
  if os.path.isfile(RadAstr):
    RadA=np.load(RadAstr)
  if LOS==0:
    for k in range(0,Nsur):
      RadSistr=meshfolder+'/'+boxstr+'RadS%d_grid%dRefs%dm%d_tx%03d.npy'%(k,Nra[j],Nre,0,job)
      if os.path.isfile(RadSistr):
        RadB[k]=np.load(RadSistr)
      Thetastr=meshfolder+'/'+boxstr+'AngNpy%03dRefs%03dNs%03d_tx%03d.npy'%(Nr,Nre,Ns,job)
      if os.path.isfile(Thetastr):
        ThetaMesh=np.load(Thetastr).astype(float)
  Nz=P.shape[2]
  lb=np.amin(P)
  ub=np.amax(P)
  if LOS:
    rlb=np.amin(RadA)
    rub=np.amax(RadA)
  else:
    rlb=min(np.amin(RadA),np.amin(RadB))
    rub=max(np.amax(RadA),np.amax(RadB))
    tlb=np.amin(ThetaMesh)
    tub=np.amax(ThetaMesh)
  if not os.path.exists('./GeneralMethodPowerFigures'):
    os.makedirs('./GeneralMethodPowerFigures')
    os.makedirs('./GeneralMethodPowerFigures/'+plottype)
    os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job)
  if not os.path.exists('./GeneralMethodPowerFigures/'+plottype):
    os.makedirs('./GeneralMethodPowerFigures/'+plottype)
    os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job)
  if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job):
    os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d'%job)
  for i in range(0,Nz):
    mp.clf()
    mp.figure(i)
    mp.imshow(P[:,:,i], cmap=cmapopt, vmax=ub,vmin=lb)
    mp.title('Power for Transmitter (%02f,%02f,%02f) at z=%d'%(Tx[0],Tx[1],Tx[2],i))
    mp.colorbar()
    rayfolder='./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/PowerSlice/Nra%03d'%(job,Nr)
    if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/PowerSlice'%job):
      os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/Tx%03d/PowerSlice'%job)
      os.makedirs(rayfolder)
    if not os.path.exists(rayfolder):
      os.makedirs(rayfolder)
    filename=rayfolder+'/'+boxstr+'PowerSliceNra%03dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,Nz)#.eps')
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
    filename=rayfolder+'/'+boxstr+'RadASliceNra%02dNref%03dslice%03dof%03d.jpg'%(Nr,Nre,i+1,Nz)#.eps')
    mp.savefig(filename,bbox_inches=plotfit)
    mp.clf()
    if not LOS:
      for k in range(0,Nsur):
        mp.figure(i)
        mp.imshow(RadB[k,:,:,i], cmap=cmapopt, vmax=rub,vmin=rlb)
        mp.colorbar()
        filename=rayfolder+'/'+boxstr+'RadS%02dSliceNra%03dNref%03dslice%03dof%03d.jpg'%(k,Nr,Nre,i+1,Nz)#.eps')
        mp.savefig(filename,bbox_inches=plotfit)
        mp.clf()
  return

def plot_residual(plottype,testnum,roomnumstat):
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  Nrs        =np.load('Parameters/Nrs.npy')
  Nsur       =np.load('Parameters/Nsur.npy')
  Ns         =np.load('Parameters/Ns.npy')
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
      refstr='Parameters/refindex%03d.npy'%k
      refindex=np.load(refstr)
      obnumbers=np.zeros((Nrs,1))
      l=0
      Obstr=''
      if Nrs<Nsur:
        for ob, refin in enumerate(refindex):
          if abs(refin)>epsilon:
            obnumbers[k]=ob
            l+=1
            Obstr=Obstr+'Ob%02d'%ob
      meshfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
      pratstr=meshfolder+'/'+boxstr+Obstr+'PowerRes_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,k,job)
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
          filename=rayfolder+'/'+boxstr+Obstr+'DiffSliceNra%03dNref%03dslice%03dof%03d_tx%03d.jpg'%(Nr,Nre,j+1,n,job)#.eps')
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
      errorname='./Errors/'+plottype+'/'+boxstr+Obstr+'ErrorsNrays%03dRefs%03dRoomnum%03dto%03d_tx%03d.npy'%(nra,Nre,roomnumstat,roomnumstat+(k)*2,job)
      if os.path.isfile(errorname):
        Res=np.load(errorname)
        mp.figure(k+1)
        mp.plot(Nra,Res)
        filename='./Errors/'+plottype+'/'+boxstr+Obstr+'Residual%03dto%03dNref%03d_tx%03d.jpg'%(Nra[0],Nra[-1],Nre,job)
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

def plot_quality_contour(InnerOb,Nr,Nrs,LOS,Nre,PerfRef,Ns,Q,Par,index):
  numjobs    =np.load('Parameters/Numjobs.npy')
  numjobs=126
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  _,_,L,split    =np.load('Parameters/Raytracing.npy')
  Nra        =np.load('Parameters/Nra.npy')
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.
  Orig          =np.load('Parameters/Origin.npy')
  #Nrs           =np.load('Parameters/Nrs.npy')
  Nsur          =np.load('Parameters/Nsur%d.npy'%index)
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles

  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  myfile = open('Parameters/PlotFit.txt', 'rt') # open lorem.txt for reading text
  plotfit= myfile.read()         # read the entire file into a string
  myfile.close()
  h=1.0/Ns
  if Nre>1:
    Refstr=nw.num2words(Nre)+'Ref'
  else:
    Refstr='NoRef'
  roomnummax=roomnumstat
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Oblist        =np.load('Parameters/Obstacles%d.npy'%index).astype(float)      # The obstacles which are within the outerboundary
  Nsur          =np.load('Parameters/Nsur%d.npy'%index)
  refindex      =np.load('Parameters/refindex%03d.npy'%index)
  Pol           = np.load('Parameters/Pol%03d.npy'%index)
  NtriOb        =np.load('Parameters/NtriOb.npy')               # Number of triangles forming the surfaces of the obstacles
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
  Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
  obnumbers=np.zeros((Nrs,1))
  k=0
  Obstr=''
  if 0<Nrs<Nsur:
    obnumbers=np.zeros(Nrs)
    for ob, refind in enumerate(refindex):
      if abs(refind)>epsilon:
        obnumbers[k]=ob
        k+=1
        Obstr=Obstr+'Ob%02d'%ob
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  if LOS:
    LOSstr='LOS'
  elif PerfRef:
    if Nre>2:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'PerfRef'
      else:
        LOSstr='MultiPerfRef'
    else:
      LOSstr='SinglePerfRef'
  else:
    if Nre>2 and Nrs>1:
      if Nrs<Nsur:
        LOSstr=nw.num2words(Nrs)+'Ref'
      else:
        LOSstr='MultiRef'
    else:
      LOSstr='SingleRef'
  foldtype=Refstr+boxstr
  #Room contains all the obstacles and walls.
  Room=rom.room(Oblist,Ntri)
  Nob=Room.Nob
  Nsur=int(Room.Nsur)
  Nx=int(Room.maxxleng()/h)
  Ny=int(Room.maxyleng()/h)
  Nz=int(Room.maxzleng()/h)
  Ns=max(Nx,Ny,Nz)
  Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1. Repeat Znobrat to match Mesh dimensions
  Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a 1 for placement in the LOS row
  refindex=np.tile(refindex,(Nre,1))        # The number of rows is Nsur*Nre+1. Repeat refindex to match Mesh dimensions
  refindex=np.insert(refindex,0,1.0+0.0j)   # Use a 1 for placement in the LOS row
  # Calculate the necessry parameters for the power calculation.
  for i in range(nra):
    Nr=int(Nra[i])
    gainname      ='Parameters/Tx%03dGains%03d.npy'%(Nr,index)
    Gt            = np.load(gainname)
    Mesh=DSM.DS(Nx,Ny,Nz,Nsur*int(Nre)+1,Nr*(int(Nre)+1),np.complex128,int(split))
    rom.FindInnerPoints(Room,Mesh)
    #pstr       =meshfolder+'/'+Box+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,432)
    #P   =np.load(pstr)
    qualtype=Refstr+LOSstr+boxstr
    meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    qfolder   ='./Mesh/'+qualtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    Qmeshstr=meshfolder+'/'+boxstr+Obstr+'QualityAverage%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,index)
    QPmeshstr=meshfolder+'/'+boxstr+Obstr+'QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,index)
    QMmeshstr=meshfolder+'/'+boxstr+Obstr+'QualityMin%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,index)
    check=0
    if not os.path.exists('./Mesh'):
      os.makedirs('./Mesh')
      os.makedirs('./Mesh/'+foldtype)
      os.makedirs(meshfolder)
    if not os.path.exists('./Mesh/'+foldtype):
      os.makedirs('./Mesh/'+foldtype)
      os.makedirs(meshfolder)
    if not os.path.exists(meshfolder):
      os.makedirs(meshfolder)
    if os.path.isfile(Qmeshstr):
      Qmesh=np.load(Qmeshstr)
      print('Q loaded from ',Qmeshstr)
      check=0
    if os.path.isfile(QPmeshstr):
      print('QP loaded from ',QPmeshstr)
      QPmesh=np.load(QPmeshstr)
      check+=0
    if os.path.isfile(QMmeshstr):
      print('QM loaded from ',QMmeshstr)
      QMmesh=np.load(QMmeshstr)
      check+=0
    feature_x = L*np.linspace(0.0,Nx*h,Nx)
    feature_y = L*np.linspace(0.0,Ny*h,Ny)
    [X, Y] = np.meshgrid(feature_y,feature_x )
    if check<3:
      Qmesh=ma.nan*np.ones((Nx,Ny,Nz))
      QPmesh=ma.nan*np.ones((Nx,Ny,Nz))
      QMmesh=ma.nan*np.ones((Nx,Ny,Nz))
      for job in range(numjobs):
        #Tx=np.load('Parameters/Origin_job%03d.npy'%job)
        Tx=RTM.MoveTx(job,Nx,Ny,Nz,h)
        if abs(Tx[0]-0.5)<epsilon and abs(Tx[1]-0.5)<epsilon and abs(Tx[2]-0.5)<epsilon:
          loca='Centre'
        else:
          loca='OffCentre'
        plottype=LOSstr+boxstr+loca
        meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
        powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
        Txind=Room.position(Tx,h)
        if not Room.CheckTxInner(Tx):
          if any(t>1.0 or t<0 for t in Tx):
            print('Tx outside room',Tx)
            continue
          else:
            print('Tx inside an obstacle',Tx)
            Qmesh[Txind] =ma.nan
            QPmesh[Txind]=ma.nan
            QMmesh[Txind]=ma.nan
        if job<numjobs+1:# and job!=35:# and job!=36 and job!=37 and job!=38:
          print('job',job)
          print('Tx',Tx,Txind)
          qualityname='./Quality/'+plottype+'/'+boxstr+Obstr+'Quality%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
          if os.path.isfile(qualityname):
            Qu=np.load(qualityname)
            Qmesh[Txind]=Qu
          else:
            pstr       =powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
            if os.path.isfile(pstr):
              P=np.load(pstr)
              Qu=DSM.QualityFromPower(P)
              if not os.path.exists('./Quality'):
                os.makedirs('./Quality')
                os.makedirs('./Quality/'+plottype)
              if not os.path.exists('./Quality/'+plottype):
                os.makedirs('./Quality/'+plottype)
              np.save(qualityname,Qu)
              Qmesh[Txind]=Qu
            else:
              print('Power file not found')
              print(pstr)
              meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
              meshname=meshfolder+'/DSM_tx%03d'%(job)
              mesheg=meshname+'%02dx%02dy%02dz.npz'%(0,0,0)
              if os.path.isfile(mesheg):
                Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
                P,ind=DSM.power_compute(foldtype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,Ns,LOS,PerfRef)
                if not os.path.exists('./Mesh'):
                  os.makedirs('./Mesh')
                  os.makedirs('./Mesh/'+plottype)
                  os.makedirs(powerfolder)
                if not os.path.exists('./Mesh/'+plottype):
                  os.makedirs('./Mesh/'+plottype)
                  os.makedirs(powerfolder)
                if not os.path.exists(powerfolder):
                  os.makedirs(powerfolder)
                if not os.path.exists('./Quality'):
                  os.makedirs('./Quality')
                  os.makedirs('./Quality/'+plottype)
                if not os.path.exists('./Quality/'+plottype):
                  os.makedirs('./Quality/'+plottype)
                np.save(pstr,P)
                Qu=DSM.QualityFromPower(P)
                np.save(qualityname,Qu)
                Qmesh[Txind]=Qu
              else:
                Qmesh[Txind]=ma.nan
                print('Mesh file not found')
                print(meshname)
                print('Quality Average not found')
                print(qualityname)
          qualityPname='./Quality/'+plottype+'/'+boxstr+Obstr+'QualityPercentile%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
          if os.path.isfile(qualityPname):
            QuP=np.load(qualityPname)
            QPmesh[Txind]=QuP
          else:
            pstr       =powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
            if os.path.isfile(pstr):
              P=np.load(pstr)
              Qu=DSM.QualityPercentileFromPower(P)
              if not os.path.exists('./Quality'):
                os.makedirs('./Quality')
                os.makedirs('./Quality/'+plottype)
              if not os.path.exists('./Quality/'+plottype):
                os.makedirs('./Quality/'+plottype)
              np.save(qualityPname,Qu)
              Qmesh[Txind]=Qu
            else:
              print('Power not found')
              print(pstr)
              meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
              meshname=meshfolder+'/DSM_tx%03d'%(job)
              mesheg=meshname+'%02dx%02dy%02dz.npz'%(0,0,0)
              if os.path.isfile(mesheg):
                Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
                P,ind=DSM.power_compute(foldtype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,Ns,LOS,PerfRef)
                if not os.path.exists('./Mesh'):
                  os.makedirs('./Mesh')
                  os.makedirs('./Mesh/'+plottype)
                  os.makedirs(powerfolder)
                if not os.path.exists('./Mesh/'+plottype):
                  os.makedirs('./Mesh/'+plottype)
                  os.makedirs(powerfolder)
                if not os.path.exists(powerfolder):
                  os.makedirs(powerfolder)
                if not os.path.exists('./Quality'):
                  os.makedirs('./Quality')
                  os.makedirs('./Quality/'+plottype)
                if not os.path.exists('./Quality/'+plottype):
                  os.makedirs('./Quality/'+plottype)
                np.save(pstr,P)
                Qu=DSM.QualityPercentileFromPower(P)
                np.save(qualityPname,Qu)
                Qmesh[Txind]=Qu
              else:
                print('Mesh not found')
                print(meshname)
                Qmesh[Txind]=ma.nan
                print('Quality Percentile not found')
                print(qualityname)
          qualityMname='./Quality/'+plottype+'/'+boxstr+Obstr+'QualityMin%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
          if os.path.isfile(qualityMname):
            QuM=np.load(qualityMname)
            QMmesh[Txind]=QuM
          else:
            pstr       =powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
            if os.path.isfile(pstr):
              P=np.load(pstr)
              Qu=DSM.QualityMinFromPower(P)
              if not os.path.exists('./Quality'):
                os.makedirs('./Quality')
                os.makedirs('./Quality/'+plottype)
              if not os.path.exists('./Quality/'+plottype):
                os.makedirs('./Quality/'+plottype)
              np.save(qualityMname,Qu)
              QMmesh[Txind]=Qu
            else:
              print('Power not found')
              print(pstr)
              meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
              meshname=meshfolder+'/DSM_tx%03d'%(job)
              mesheg=meshname+'%02dx%02dy%02dz.npz'%(0,0,0)
              if os.path.isfile(mesheg):
                Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
                P,ind=DSM.power_compute(foldtype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,Ns,LOS,PerfRef)
                if not os.path.exists('./Mesh'):
                  os.makedirs('./Mesh')
                  os.makedirs('./Mesh/'+plottype)
                  os.makedirs(powerfolder)
                if not os.path.exists('./Mesh/'+plottype):
                  os.makedirs('./Mesh/'+plottype)
                  os.makedirs(powerfolder)
                if not os.path.exists(powerfolder):
                  os.makedirs(powerfolder)
                if not os.path.exists('./Quality'):
                  os.makedirs('./Quality')
                  os.makedirs('./Quality/'+plottype)
                if not os.path.exists('./Quality/'+plottype):
                  os.makedirs('./Quality/'+plottype)
                np.save(pstr,P)
                Qu=DSM.QualityMinFromPower(P)
                np.save(qualityMname,Qu)
                Qmesh[Txind]=Qu
              else:
                print('Mesh not found')
                print(meshname)
                Qmesh[Txind]=ma.nan
                print('Quality Min not found')
                print(qualityname)
    #if any(not ma.isnan(q) for q in Qmesh.flatten()):
    #  print(any(not ma.isnan(q) for q in Qmesh.flatten()))
    ResultsFolder='./OptimisationResults'
    SpecResultsFolder=ResultsFolder+'/'+LOSstr+boxstr+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
    if not os.path.exists(ResultsFolder):
      os.makedirs(ResultsFolder)
      os.makedirs(ResultsFolder+'/'+LOSstr+boxstr)
      os.makedirs(SpecResultsFolder)
    if not os.path.exists(ResultsFolder+'/'+LOSstr+boxstr):
      os.makedirs(ResultsFolder+'/'+LOSstr+boxstr)
      os.makedirs(SpecResultsFolder)
    if not os.path.exists(SpecResultsFolder):
      os.makedirs(SpecResultsFolder)
    TxOpt=np.unravel_index(np.argmax(Qmesh),Qmesh.shape) #np.where(Qmesh==np.amax(Qmesh))
    TxOpt=Room.coordinate(h,TxOpt[0],TxOpt[1],TxOpt[2])
    print('Optimal T for Q ave from Exhaust:',TxOpt)
    np.save(SpecResultsFolder+'/'+Obstr+'OptimumExhaustOriginAverage.npy',TxOpt)
    TxOptM=np.unravel_index(np.argmax(QMmesh),QMmesh.shape) #np.where(QMmesh==np.amax(QMmesh))
    TxOptM=Room.coordinate(h,TxOptM[0],TxOptM[1],TxOptM[2])
    print('Optimal T for Q min from Exhaust:',TxOptM)
    np.save(SpecResultsFolder+'/'+Obstr+'OptimumExhaustOriginMin.npy',TxOptM)
    TxOptP=np.unravel_index(np.argmax(QPmesh),QPmesh.shape)
    TxOptP=Room.coordinate(h,TxOptP[0],TxOptP[1],TxOptP[2])
    print('Optimal T for Q percentile from Exhaust:',TxOptP)
    np.save(SpecResultsFolder+'/'+Obstr+'OptimumExhaustOriginPercentile.npy',TxOpt)
    text_file = open(SpecResultsFolder+'/'+Obstr+'OptimalGainExhaust.txt', 'w')
    n = text_file.write('\n Optimal location for exhaustive search of average power is at ')
    n = text_file.write(str(TxOpt))
    n = text_file.write('\n Optimal location for exhaustive search of min power is at ')
    n = text_file.write(str(TxOptM))
    n = text_file.write('\n Optimal location for exhaustive search of 10th percentile power is at ')
    n = text_file.write(str(TxOptP))
    text_file.close()
    print('Optimal saved to '+SpecResultsFolder+'/'+Obstr+'OptimalGainExhaust.txt')
    Qterms=np.array([q for q in Qmesh.flatten() if not ma.isnan(q)])
    QPterms=np.array([q for q in QPmesh.flatten() if not ma.isnan(q) ])
    QMterms=np.array([q for q in QMmesh.flatten() if not ma.isnan(q) ])
    Qmin=min(np.amin(Qterms),np.amin(QPterms),np.amin(QMterms))
    Qmax=max(np.amax(Qterms),np.amax(QPterms),np.amax(QMterms))
    if Qmin==Qmax:
      print('Q is the same everywhere')
      print(Qmesh)
    else:
      norm=matplotlib.colors.Normalize(vmin=Qmin, vmax=Qmax)
      print('Q ave range',np.amin(Qterms),np.amax(Qterms))
      print('QP range',np.amin(QPterms),np.amax(QPterms))
      print('Q min range', np.amin(QMterms),np.amax(QMterms))
      if not os.path.exists('./Mesh/'+qualtype):
        os.mkdir('./Mesh/'+qualtype)
      if not os.path.exists('./Quality'):
        os.mkdir('./Quality')
        os.mkdir('./Quality/'+qualtype)
      if not os.path.exists('./Quality/'+qualtype):
        os.mkdir('./Quality/'+qualtype)
      np.save(Qmeshstr,Qmesh)
      np.save(QPmeshstr,QPmesh)
      np.save(QMmeshstr,QMmesh)
      Qstr='Quality/'+qualtype+'/'+boxstr+Obstr+'OptimalLocationAverage%03dRefs%03dm%03d'%(Nr,Nre,index)
      TxOpt=np.argmax(Qmesh)
      np.save(Qstr+'.npy',TxOpt)
      text_file = open(Qstr+'.txt', 'w')
      n = text_file.write('Optimal transmitter location at ')
      n = text_file.write(str(TxOpt))
      text_file.close()
      Qstr='Quality/'+qualtype+'/'+boxstr+Obstr+'OptimalLocationPercentile%03dRefs%03dm%03d'%(Nr,Nre,index)
      TxOpt=np.argmax(QPmesh)
      np.save(Qstr+'.npy',TxOpt)
      text_file = open(Qstr+'.txt', 'w')
      n = text_file.write('Optimal transmitter location at ')
      n = text_file.write(str(TxOpt))
      text_file.close()
      Qstr='Quality/'+qualtype+'/'+boxstr+Obstr+'OptimalLocationMinimum%03dRefs%03dm%03d'%(Nr,Nre,index)
      TxOpt=np.argmax(QMmesh)
      np.save(Qstr+'.npy',TxOpt)
      text_file = open(Qstr+'.txt', 'w')
      n = text_file.write('Optimal transmitter location at ')
      n = text_file.write(str(TxOpt))
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
          filename='./Quality/'+qualtype+'/'+boxstr+Obstr+'QualityAverageSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
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
          filename='./Quality/'+qualtype+'/'+boxstr+Obstr+'QualityPercentileSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
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
          filename='./Quality/'+qualtype+'/'+boxstr+Obstr+'QualityMinSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
          mp.close()
          fig, ax = mp.subplots(1, 1)
          ax.contourf(X,Y,Qmesh[:,:,k])
          fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
          ax.set_title('Quality of coverage by transmitter location z=%02f'%(k*h+h/2))
          ax.set_xlabel('X position of Tx')
          ax.set_ylabel('Y position of Tx')
          filename='./Quality/'+qualtype+'/'+boxstr+Obstr+'QualityAverageContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
          mp.close()
          fig, ax = mp.subplots(1, 1)
          ax.contourf(X,Y,QPmesh[:,:,k])
          fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
          ax.set_title('Quality Percentile of coverage by transmitter location z=%02f'%(k*h+h/2))
          ax.set_xlabel('X position of Tx')
          ax.set_ylabel('Y position of Tx')
          filename='./Quality/'+qualtype+'/'+boxstr+Obstr+'QualityPercentileContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
          mp.close()
          fig, ax = mp.subplots(1, 1)
          ax.contourf(X,Y,QMmesh[:,:,k])
          fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
          ax.set_title('Quality (min) of coverage by transmitter location z=%02d'%(k*h+h/2))
          ax.set_xlabel('X position of Tx')
          ax.set_ylabel('Y position of Tx')
          filename='./Quality/'+qualtype+'/'+boxstr+Obstr+'QualityMinContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
          mp.savefig(filename,bbox_inches=plotfit)
          mp.clf()
          mp.close()
  return

def main():
  parameters=  np.flip(np.load('Parameters/Parameterarray.npy'),axis=0)
  _,_,L,split    =np.load('Parameters/Raytracing.npy')
  for arr in parameters:
    InnerOb,Nr,Nrs,LOS,Nre,PerfRef,Ns,Q,Par,index=arr.astype(int)
    if not Nr==22 or not Q==0:
      continue
    #plot_times(plottype,testnum,roomnumstat)
    if Par==0 and Ns==5:
      plot_quality_contour(InnerOb,Nr,Nrs,LOS,Nre,PerfRef,Ns,Q,Par,index)
      plot_grid(InnerOb,Nr,Nrs,LOS,Nre,PerfRef,Ns,Q,Par,index)        # Plot the power in slices.
    else:
      plot_grid(InnerOb,Nr,Nrs,LOS,Nre,PerfRef,Ns,Q,Par,index)        # Plot the power in slices.
    #ResOn      =np.load('Parameters/ResOn.npy')
    #if ResOn:
    #  plot_residual(plottype,testnum,roomnumstat)
  return

if __name__=='__main__':
  main()
  exit()

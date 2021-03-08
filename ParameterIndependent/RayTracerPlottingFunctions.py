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
  Nra        =np.load('Parameters/Nra.npy')
  numjobs    =np.load('Parameters/Numjobs.npy')
  Nrs        =np.load('Parameters/Nrs.npy')
  Ns         =np.load('Parameters/Ns.npy')
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
  if Nre>1:
    Refstr=nw.num2words(Nre)+''
  else:
    Refstr='NoRef'
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
  if InnerOb:
    boxstr='Box'
  else:
    boxstr='NoBox'
  Roomnum=roomnumstat
  for i in range(testnum):
    for index in range(0,Roomnum):
      refindex=np.load('Parameters/refindex%03d.npy'%index)
      obnumbers=np.zeros((Nrs,1))
      k=0
      Obstr=''
      if Nrs<Nsur:
        for ob, refin in enumerate(refindex):
          if abs(refin)>epsilon:
            obnumbers[k]=ob
            k+=1
            Obstr=Obstr+'Ob%02d'%ob
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
            foldtype=Refstr+boxstr+loca
            plottype=LOSstr+boxstr+loca
            meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
            powerfolder='./Mesh/'+plottype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
            pstr       =powerfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
            if os.path.isfile(pstr):
              print('Plotting power at Tx=',Tx)
              P   =np.load(pstr)
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
                  Thetastr=powerfolder+'/'+boxstr+'AngNpy%03dRefs%03dNs%03d_tx%03d.npy'%(Nra[j],Nre,Ns,job)
                  if os.path.isfile(Thetastr):
                    ThetaMesh=np.load(Thetastr).astype(float)
              n=P.shape[2]
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
    Roomnum*=2
  return

def plot_mesh(Mesh,Room,Tx,plottype,boxstr,Obstr,Nr,Nre,Ns,plotfit,LOS=0,index=0):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nsur=Room.Nsur
  h=Room.get_meshwidth(Mesh)
  job=RTM.jobfromTx(Tx,h)
  pstr       =meshfolder+'/'+boxstr+Obstr+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,index,job)
  if os.path.isfile(pstr):
    P   =np.load(pstr)
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

def plot_quality_contour(plottype,testnum,roomnumstat):
  numjobs    =np.load('Parameters/Numjobs.npy')
  numjobs=126
  roomnumstat=np.load('Parameters/roomnumstat.npy')
  Nre,h,L,split    =np.load('Parameters/Raytracing.npy')
  Nre=6
  Nra        =np.load('Parameters/Nra.npy')
  Ntri          =np.load('Parameters/NtriOut.npy')              # Number of triangles forming the surfaces of the outerboundary
  MaxInter      =np.load('Parameters/MaxInter.npy')             # The number of intersections a single ray can have in the room in one direction.
  Orig          =np.load('Parameters/Origin.npy')
  Nrs           =np.load('Parameters/Nrs.npy')
  Nsur          =np.load('Parameters/Nsur.npy')
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
  for index in range(testnum):
    Oblist        =np.load('Parameters/Obstacles%d.npy'%index).astype(float)      # The obstacles which are within the outerboundary
    InnerOb    =np.load('Parameters/InnerOb%d.npy'%index)
    LOS        =np.load('Parameters/LOS%d.npy'%index)
    PerfRef    =np.load('Parameters/PerfRef%d.npy'%index)
    Nsur       =np.load('Parameters/Nsur%d.npy'%index)
    refindex=np.load('Parameters/refindex%03d.npy'%index)
    freq          = np.load('Parameters/frequency%03d.npy'%index)
    Freespace     = np.load('Parameters/Freespace%03d.npy'%index)
    Pol           = np.load('Parameters/Pol%03d.npy'%index)
    MaxInter     =np.load('Parameters/MaxInter.npy')
    ##----Retrieve the Obstacle Parameters--------------------------------------
    Znobrat      =np.load('Parameters/Znobrat%03d.npy'%index)
    refindex     =np.load('Parameters/refindex%03d.npy'%index)
    Antpar        =np.load('Parameters/Antpar%03d.npy'%index)
    obnumbers=np.zeros((Nrs,1))
    k=0
    Obstr=''
    if Nrs<Nsur:
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
        if Nrs<nsur:
          LOSstr=nw.num2words(Nrs)+'Ref'
        else:
          LOSstr='MultiRef'
      else:
        LOSstr='SingleRef'
    foldtype=Refstr+boxstr
    #Room contains all the obstacles and walls.
    Room=rom.room(Oblist,Ntri)
    Nob=Room.Nob
    Room.__set_MaxInter__(MaxInter)
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
      Nr=22 #int(Nra[i])
      gainname      ='Parameters/Tx%dGains%d.npy'%(Nr,index)
      Gt            = np.load(gainname)
      Mesh=DSM.DS(Nx,Ny,Nz,Nsur*int(Nre)+1,Nr*(int(Nre)+1),np.complex128,int(split))
      rom.FindInnerPoints(Room,Mesh)
      #pstr       =meshfolder+'/'+Box+'Power_grid%03dRefs%03dm%03d_tx%03d.npy'%(Nr,Nre,0,432)
      #P   =np.load(pstr)
      #Nx=P.shape[0]
      #Ny=P.shape[1]
      #Nz=P.shape[2]
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
      # QPmesh =np.load(meshfolder+'/QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      # Qmesh  =np.load(meshfolder+'/QualityAverage%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      # QMmesh  =np.load(meshfolder+'/QualityMin%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      # Tmesh  =np.load(meshfolder+'/TimeFullCalc%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
      for job in range(numjobs):
       Tx=np.load('Parameters/Origin_job%03d.npy'%job)
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
            Tmesh[Txind] =ma.nan
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
              Qmesh[Txind]=Qu
            else:
              print('Power file not found')
              print(pstr)
              meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
              meshname=meshfolder+'/DSM_tx%03d'%(job)
              if os.path.isfile(meshname):
                Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
                P,ind=DSM.power_compute(foldtype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,Ns,LOS,PerfRef)
                Qu=DSM.QualityFromPower(P)
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
              Qmesh[Txind]=Qu
              meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
              meshname=meshfolder+'/DSM_tx%03d'%(job)
            else:
              if os.path.isfile(meshname):
                Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
                P,ind=DSM.power_compute(foldtype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,Ns,LOS,PerfRef)
                Qu=DSM.QualityPercentileFromPower(P)
                Qmesh[Txind]=Qu
              else:
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
              QMmesh[Txind]=Qu
            else:
              meshfolder='./Mesh/'+foldtype+'/Nra%03dRefs%03dNs%0d'%(Nr,Nre,Ns)
              meshname=meshfolder+'/DSM_tx%03d'%(job)
              if os.path.isfile(meshname):
                Mesh= DSM.load_dict(meshname,Nx,Ny,Nz)
                P,ind=DSM.power_compute(foldtype,Mesh,Room,Znobrat,refindex,Antpar,Gt,Pol,Nr,Nre,Ns,LOS,PerfRef)
                Qu=DSM.QualityMinFromPower(P)
                Qmesh[Txind]=Qu
              else:
                Qmesh[Txind]=ma.nan
              print('Quality Min not found')
              print(qualityname)
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
      if numjobs>0:
        Qmin=min(np.amin(Qmesh),np.amin(QPmesh),np.amin(QMmesh))
        Qmax=max(np.amax(Qmesh),np.amax(QPmesh),np.amax(QMmesh))
        if Qmin==Qmax:
          print('Q is the same everywhere')
          print(Qmesh)
        else:
          norm=matplotlib.colors.Normalize(vmin=Qmin, vmax=Qmax)
          print('Q ave range',np.amin(Qmesh),np.amax(Qmesh))
          print('QP range',np.amin(QPmesh),np.amax(QPmesh))
          print('Q min range', np.amin(QMmesh),np.amax(QMmesh))
          foldtype=Refstr+LOSstr+boxstr
          if not os.path.exists('./Mesh'):
            os.mkdir('./Mesh')
            os.mkdir('./Mesh/'+foldtype)
            os.mkdir('./Mesh/'+plottype)
            os.mkdir(meshfolder)
          if not os.path.exists('./Mesh/'+foldtype):
            os.mkdir('./Mesh/'+foldtype)
          if not os.path.exists('./Mesh/'+plottype):
            os.mkdir('./Mesh/'+plottype)
            os.mkdir(meshfolder)
          if not os.path.exists(meshfolder):
            os.mkdir(meshfolder)
          if not os.path.exists('./Quality'):
            os.mkdir('./Quality')
            os.mkdir('./Quality/'+foldtype)
          if not os.path.exists('./Quality/'+foldtype):
            os.mkdir('./Quality/'+foldtype)
          np.save(meshfolder+'/'+boxstr+Obstr+'QualityAverage%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,index),Qmesh)
          np.save(meshfolder+'/'+boxstr+Obstr+'QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,index),QPmesh)
          np.save(meshfolder+'/'+boxstr+Obstr+'QualityMin%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,index),QMmesh)
          Qstr='Quality/'+foldtype+'/'+boxstr+Obstr+'OptimalLocationAverage%03dRefs%03dm%03d'%(Nr,Nre,index)
          TxOpt=np.argmax(Qmesh)
          np.save(Qstr+'.npy',TxOpt)
          text_file = open(Qstr+'.txt', 'w')
          n = text_file.write('Optimal transmitter location at ')
          n = text_file.write(str(TxOpt))
          text_file.close()
          Qstr='Quality/'+foldtype+'/'+boxstr+Obstr+'OptimalLocationPercentile%03dRefs%03dm%03d'%(Nr,Nre,index)
          TxOpt=np.argmax(QPmesh)
          np.save(Qstr+'.npy',TxOpt)
          text_file = open(Qstr+'.txt', 'w')
          n = text_file.write('Optimal transmitter location at ')
          n = text_file.write(str(TxOpt))
          text_file.close()
          Qstr='Quality/'+foldtype+'/'+boxstr+Obstr+'OptimalLocationMinimum%03dRefs%03dm%03d'%(Nr,Nre,index)
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
            filename='./Quality/'+foldtype+'/'+boxstr+Obstr+'QualitySurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
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
            filename='./Quality/'+foldtype+'/'+boxstr+Obstr+'QualityPercentileSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
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
            filename='./Quality/'+foldtype+'/'+boxstr+Obstr+'QualityMinSurface%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig, ax = mp.subplots(1, 1)
            ax.contourf(X,Y,Qmesh[:,:,k])
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality of coverage by transmitter location z=%02f'%(k*h+h/2))
            ax.set_xlabel('X position of Tx')
            ax.set_ylabel('Y position of Tx')
            filename='./Quality/'+foldtype+'/'+boxstr+Obstr+'QualityContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig, ax = mp.subplots(1, 1)
            ax.contourf(X,Y,QPmesh[:,:,k])
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality Percentile of coverage by transmitter location z=%02f'%(k*h+h/2))
            ax.set_xlabel('X position of Tx')
            ax.set_ylabel('Y position of Tx')
            filename='./Quality/'+foldtype+'/'+boxstr+Obstr+'QualityPercentileContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
            mp.savefig(filename,bbox_inches=plotfit)
            mp.clf()
            mp.close()
            fig, ax = mp.subplots(1, 1)
            ax.contourf(X,Y,QMmesh[:,:,k])
            fig.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmapopt), ax=ax)
            ax.set_title('Quality (min) of coverage by transmitter location z=%02d'%(k*h+h/2))
            ax.set_xlabel('X position of Tx')
            ax.set_ylabel('Y position of Tx')
            filename='./Quality/'+foldtype+'/'+boxstr+Obstr+'QualityMinContour%03dNref%03d_z%02d.jpg'%(Nr,Nre,k)
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
  #plot_grid(plottype,testnum,roomnumstat)        # Plot the power in slices.
  plot_quality_contour(plottype,testnum,roomnumstat)
  ResOn      =np.load('Parameters/ResOn.npy')
  if ResOn:
    plot_residual(plottype,testnum,roomnumstat)
  return

if __name__=='__main__':
  main()
  exit()

#!/usr/bin/env python3
# Updated Hayley Wragg 2020-06-30
'''Code to plot heatmaps and error plots for RayTracer numpy files.
'''
import DictionarySparseMatrix as DSM
import matplotlib.pyplot as mp
import ParameterLoad as PI
import openpyxl as wb
import numpy as np
import os
import sys
epsilon=sys.float_info.epsilon

def plot_grid(plottype=str(),testnum=1,roomnumstat=0):
  ''' Plots slices of a 3D power grid.

  Loads `Power_grid.npy` and for each z step plots a heatmap of the \
  values at the (x,y) position.
  '''
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  myfile = open('Parameters/Heatmapstyle.txt', 'rt') # open lorem.txt for reading text
  cmapopt= myfile.read()         # read the entire file into a string
  myfile.close()
  LOS    =np.load('Parameters/LOS.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  Roomnum=roomnumstat
  for i in range(testnum):
    for index in range(0,Roomnum):
      for j in range(0,nra):
        Nr=int(Nra[j])
        Nre=int(Nre)
        pstr       ='./Mesh/'+plottype+'/Power_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
        pstrstd    ='./Mesh/'+plottype+'/Power_gridstd%dRefs%dm%d.npy'%(Nr,Nre,index)
        truestr    ='Mesh/True/'+plottype+'/True.npy'
        P3  =np.load(truestr)
        P   =np.load(pstr)
        pratstr='./Mesh/'+plottype+'/PowerRat_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
        Prattil=np.load(pratstr)
        err2=np.load('./Errors/'+plottype+'/Residual%dRefs%dm%d.npy'%(Nr,Nre,index))
        print('Residual GRL to true',err2)
        RadAstr    ='./Mesh/'+plottype+'/RadA_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
        if LOS==0:
          RadBstr    ='./Mesh/'+plottype+'/RadB_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
          TrueRadBstr='Mesh/True/'+plottype+'/TrueRadB.npy'
          RadB=np.load(RadBstr)
          TrueRadB=np.load(TrueRadBstr)
        TrueRadAstr='Mesh/True/'+plottype+'/TrueRadA.npy'
        RadA=np.load(RadAstr)
        TrueRadA=np.load(TrueRadAstr)
        #Pdifftil=abs(np.divide(P-P3,P, where=(abs(P)>epsilon)))  # Normalised Difference Mesh
        #RadAdifftil=abs(np.divide(RadA-TrueRadA,RadA, where=(abs(RadA)>epsilon)))  # Normalised Difference Mesh
        #RadAdiffstr='./Mesh/'+plottype+'/RadADiff_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
        #np.save(RadAdiffstr,RadAdifftil)
        #errrad=np.sum(RadAdifftil)/(P.shape[0]*P.shape[1]*P.shape[2])
        #print('Residual GRL to true RadA',errrad)
        if LOS==0:
          RadBdifftil=abs(np.divide(RadB-TrueRadB,RadB, where=(abs(RadB)>epsilon)))  # Normalised Difference Mesh
          RadBdiffstr='./Mesh/'+plottype+'/RadBDiff_grid%dRefs%dm%d.npy'%(Nr,Nre,index)
          np.save(RadBdiffstr,RadBdifftil)
          errrad=np.sum(RadBdifftil)/(P.shape[0]*P.shape[1]*P.shape[2])
          print('Residual GRL to true RadB',errrad)
        #err3=np.sum(Pdiffhat)/(P.shape[0]*P.shape[1]*P.shape[2])
        #print('Residual of std to true',err3)
        #n2=P2.shape[2]
        #n3=Pdiff.shape[2]
        n=P.shape[2]
        lb=np.amin(P)
        lb3=np.amin(P3)
        lb=min(lb,lb3)
        ub=np.amax(P)
        #lb2=np.amin(P2)
        if LOS:
          rlb=np.amin(RadA)
          rub=np.amax(RadA)
        else:
          rlb=min(np.amin(RadA),np.amin(RadB))
          rub=max(np.amax(RadA),np.amax(RadB))
        #ub2=np.amax(P2)
        ub3=np.amax(P3)
        ub=max(ub,ub3)
        if not os.path.exists('./GeneralMethodPowerFigures'):
          os.makedirs('./GeneralMethodPowerFigures')
        if not os.path.exists('./GeneralMethodPowerFigures/'+plottype):
          os.makedirs('./GeneralMethodPowerFigures/'+plottype)
        for i in range(0,n):
          mp.figure(i)
          #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
          mp.imshow(P[:,:,i], cmap=cmapopt, vmax=ub,vmin=lb)
          mp.colorbar()
          rayfolder='./GeneralMethodPowerFigures/'+plottype+'/PowerSlice/Nra%d'%Nr
          if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/PowerSlice'):
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/PowerSlice')
            os.makedirs(rayfolder)
          elif not os.path.exists(rayfolder):
            os.makedirs(rayfolder)
          filename=rayfolder+'/NoBoxPowerSliceNra%dNref%dslice%dof%d.jpg'%(Nr,Nre,i+1,n)#.eps')
          mp.savefig(filename)
          mp.clf()
          mp.figure(i)
          mp.imshow(RadA[:,:,i], cmap=cmapopt, vmax=rub,vmin=rlb)
          mp.colorbar()
          rayfolder='./GeneralMethodPowerFigures/'+plottype+'/RadSlice/Nra%d'%Nr
          if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/RadSlice'):
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/RadSlice')
            os.makedirs(rayfolder)
          elif not os.path.exists(rayfolder):
            os.makedirs(rayfolder)
          filename=rayfolder+'/NoBoxRadASliceNra%dNref%dslice%dof%d.jpg'%(Nr,Nre,i+1,n)#.eps')
          mp.savefig(filename)
          mp.clf()
          if LOS==0:
            mp.figure(i)
            mp.imshow(RadB[:,:,i], cmap=cmapopt, vmax=rub,vmin=rlb)
            mp.colorbar()
            filename=rayfolder+'/NoBoxRadBSliceNra%dNref%dslice%dof%d.jpg'%(Nr,Nre,i+1,n)#.eps')
            mp.savefig(filename)
            mp.clf()
          for i in range(0,n):
            mp.figure(2*n+i)
            #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
            mp.imshow(DSM.Watts_to_db(Prattil[:,:,i]), cmap=cmapopt, vmax=1,vmin=0)
            mp.colorbar()
            Difffolder='./GeneralMethodPowerFigures/'+plottype+'/DiffSlice/Nra%d'%Nr
            if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/DiffSlice'):
              os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/DiffSlice')
              os.makedirs(Difffolder)
            elif not os.path.exists(Difffolder):
              os.makedirs(Difffolder)
            filename=Difffolder+'/NoBoxPowerDifftilSliceNra%dNref%dslice%dof%d.jpg'%(Nr,Nre,i+1,n)#.eps')
            mp.savefig(filename)
            mp.clf()
        for i in range(n):
          mp.figure(n+i)
          #extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
          mp.imshow(P3[:,:,i], cmap=cmapopt,  vmax=ub,vmin=lb)
          mp.colorbar()
          truefolder='./GeneralMethodPowerFigures/'+plottype+'/TrueSlice'
          if not os.path.exists('./GeneralMethodPowerFigures/'+plottype+'/TrueSlice'):
            os.makedirs('./GeneralMethodPowerFigures/'+plottype+'/TrueSlice')
            os.makedirs(truefolder)
          elif not os.path.exists(truefolder):
            os.makedirs(truefolder)
          filename=truefolder+'/NoBoxTrueSliceNref%dslice%dof%d.jpg'%(Nre,i+1,n)#.eps')
          mp.savefig(filename)
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
  for j in range(testnum):
    errorname='./Errors/'+plottype+'/ErrorsNrays%dRefs%dRoomnum%dto%d.npy'%(nra,Nre,roomnumstat,roomnumstat+(j)*2)
    Res=np.load(errorname)
    mp.figure(j+1)
    mp.plot(Nra,Res)
    filename='./Errors/'+plottype+'/Residual%dto%dNref%d.jpg'%(Nra[0],Nra[-1],Nre)
    mp.savefig(filename)
  return

def plot_times(plottype,testnum,roomnumstat):
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  for j in range(testnum):
    timename='./Times/'+plottype+'/TimesNra%dRefs%dRoomnum%dto%d.npy'%(nra,Nre,roomnumstat,roomnumstat+(j)*2)
    T=np.load(timename)
    mp.figure(j+1)
    mp.plot(Nra,T)
    filename='./Times/'+plottype+'/Times%dto%dNref%d.jpg'%(Nra[0],Nra[-1],Nre)
    mp.savefig(filename)
  return

def plot_quality(plottype,testnum,roomnumstat):
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
    qualityname='./Quality/'+plottype+'/QualityNrays%dRefs%dRoomnum%dto%d.npy'%(nra,Nre,roomnumstat,roomnumstat+(j)*2)
    Qu=np.load(qualityname)
    mp.figure(j+1)
    mp.plot(Nra,Qu)
    mp.plot(Nra,Q2)
    filename='./Quality/'+plottype+'/Quality%dto%dNref%d.jpg'%(Nra[0],Nra[-1],Nre)
    mp.savefig(filename)
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
  testnum    =SimPar.cell(row=16,column=3).value
  roomnumstat=SimPar.cell(row=17,column=3).value
  plot_grid(plottype,testnum,roomnumstat)        # Plot the power in slices.
  plot_residual(plottype,testnum,roomnumstat)
  exit()

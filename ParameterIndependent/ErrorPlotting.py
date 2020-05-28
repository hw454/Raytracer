#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as mp
import os
import sys

epsilon=sys.float_info.epsilon

if __name__=='__main__':
  Nra         =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
    Nra=np.array([Nra])
    nra=1
  else:
    nra=len(Nra)+2
  E0=np.zeros((nra,13))
  E1=np.zeros((nra,13))
  E2=np.zeros((nra,13))
  E0[0]=np.array([422,346,317,306,298,294,288,282,291,282,278,278,278])
  E0[1]=np.array([343,251,219,197,178,176,170,160,162,158,157,158,156])
  E0[2]=np.array([353,181,124,118,99,84,73,75,68,67,70,67,70])
  E0[3]=np.array([305,216,190,133,97,101,90,78,78,82,78,78,78])
  E0[4]=np.array([159,96,70,64,13,49,45,41,35,35,39,13,33])
  E0[5]=np.array([76,48,42,31,27,23,19,21,6,18,17,15,15])
  E0[6]=np.array([41,30,18,19,17,17,15,13,17,13,14,13,13])
  E0[7]=np.array([20,17,16,13,13,14,12,11,11,10,10,10,10])
  E0[8]=np.array([10,6,2,6,2,2,2,2,2,2,2,2,2])
  E0[9]=np.array([16,12,6,2,2,0,0,0,0,0,0,0,0])

  E1[0]=np.array([422,242,196,188,172,159,159,159,159,159,159,159,159])
  E1[1]=np.array([343,186,145,91,50,49,49,49,49,49,49,49,49])
  E1[2]=np.array([353,207,139,80,64,60,60,60,60,60,60,60,60])
  E1[3]=np.array([305,164,107,43,30,24,24,24,24,24,24,24,24])
  E1[4]=np.array([159,104,80,59,8,38,38,38,38,38,38,38,38])
  E1[5]=np.array([76,45,37,28,13,12,12,12,12,12,12,12,12])
  E1[6]=np.array([41,29,27,23,18,4,11,11,11,11,11,11,11])
  E1[7]=np.array([20,18,16,13,10,7,7,7,7,7,7,7,7])
  E1[8]=np.array([10,0,0,0,0,0,0,0,0,0,0,0,0])
  E1[9]=np.array([16,2,6,2,2,2,2,2,2,2,2,2,2])

  E2[0]=np.array([422,153,128,120,120,118,110,110,110,110,110,110,110])
  E2[1]=np.array([343,93,59,29,11,12,10,11,7,7,8,8,8])
  E2[2]=np.array([353,66,26,10,3,0,0,0,0,0,0,0,0])
  E2[3]=np.array([305,2,45,11,0,0,0,0,0,0,0,0,0])
  E2[4]=np.array([159,50,15,4,1,0,0,0,0,0,0,0,0])
  E2[5]=np.array([76,29,17,4,1,0,0,0,0,0,0,0,0])
  E2[6]=np.array([41,21,14,6,1,0,0,0,0,0,0,0,0])
  E2[7]=np.array([20,16,8,6,0,0,0,0,0,0,0,0,0])
  E2[8]=np.array([10,0,0,0,0,0,0,0,0,0,0,0,0])
  E2[9]=np.array([16,4,0,0,0,0,0,0,0,0,0,0,0])

  if not os.path.exists('./ErrorPlotting'):
        os.makedirs('./ErrorPlotting')
  x=np.array([0,2,3,4,5,6,8,10,12,14,16,18,20])
  c0=np.array([0.0,0.0,0.0])
  cd=np.array([0.05,0.075,1/nra])
  for j in range(3):
      mp.figure(j)
      ax=mp.subplot(111)
      if j==0:
          err=E0
      elif j==1:
          err=E1
          c0=np.array([0.0,0.0,0.0])
          cd=np.array([0.075,1/nra,0.05])
      else:
          err=E2
          c0=np.array([0.0,0.0,0.0])
          cd=np.array([1/nra,0.05,0.075])
      ax.plot(x,err[0],label='12 rays',color=tuple(c0))
      c0+=cd
      ax.plot(x,err[1],label='20 rays',color=tuple(c0))
      c0+=cd
      ax.plot(x,err[2],label='98 rays',color=tuple(c0))
      c0+=cd
      ax.plot(x,err[3],label='176 rays',color=tuple(c0))
      c0+=cd
      ax.plot(x,err[4],color=tuple(c0),label='244 rays')
      c0[:]+=cd
      ax.plot(x,err[5],color=tuple(c0),label='318 rays')
      c0[:]+=cd
      ax.plot(x,err[6],color=tuple(c0),label='404 rays')
      c0[:]+=cd
      ax.plot(x,err[7],color=tuple(c0),label='446 rays')
      c0[:]+=cd
      ax.plot(x,err[8],color=tuple(c0),label='502 rays')
      c0[:]+=cd
      ax.plot(x,err[9],color=tuple(c0),label='608 rays')
      c0[:]+=cd
      ax.plot(x,err[10],color=tuple(c0),label='784 rays')
      c0[:]+=cd
      ax.plot(x,err[11],color=tuple(c0),label='1630 rays')
      ax.legend(loc='upper center')
      ax.set_ylabel('Number of missed squares')
      ax.set_xlabel('Factor increase')

      if j==0:
          mp.title('Number of missed squares against the factor increase in normals')
          mp.savefig('ErrorPlotting/ConeIncrease.png')
      elif j==1:
          mp.title('Number of missed squares against the factor increase in cone steps')
          mp.savefig('ErrorPlotting/ConeStepsIncrease.png')
      else:
          mp.title('Number of missed squares against the factor increase in both')
          mp.savefig('ErrorPlotting/BothIncrease.png')
  Nra=np.array([12,20,98,176,244,318,404,446,502,608,784,1630])
  Asy1=np.log10(E0[:,-1],where=(abs(E0[:,-1]-epsilon)>0))
  mp.figure(4)
  ax=mp.subplot(111)
  ax.plot(Nra,Asy1)
  ax.set_ylabel('log10(Asymptote value)')
  ax.set_xlabel('Number of rays')
  mp.title('When multiplying number of normals')
  mp.savefig('ErrorPlotting/ConesAsymptote.png')
  Asy2=np.log10(E1[:,-1],where=(abs(E1[:,-1]-epsilon)>0))
  mp.figure(5)
  ax=mp.subplot(111)
  ax.plot(Nra,Asy2)
  ax.set_ylabel('log10(Asymptote value)')
  ax.set_xlabel('Number of rays')
  mp.title('When multiplying cone steps')
  mp.savefig('ErrorPlotting/ConeStepsAsymptote.png')

  mp.show()

  exit()



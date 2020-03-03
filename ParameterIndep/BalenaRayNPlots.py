#!/usr/bin/env python3
# Hayley Wragg 2018-05-29
''' Code to trace rays around a room using cones to account for
spreading. This version does not remove points inside an object.'''
import numpy as np
import matplotlib.pyplot as mp
import HayleysPlotting as hp
import reflection as ref
import intersection as ins
import linefunctions as lf
#import ray_tracer_test as rtest
import geometricobjects as ob
import roommesh as rmes
import math
import time


if __name__=='__main__':
  #thetaNbNoRND=np.load('../../../../OutputFiles/OutputbNoRND.npy')
  #thetaNbRND=np.load('../../../../OutputFiles/OutputbRND.npy')
  #thetaNaNoRND=np.load('../../../../OutputFiles/OutputaNoRND.npy')
  #thetaNaRND=np.load('../../../../OutputFiles/OutputaRND.npy')
  M=np.array([3,4,5])
  N=np.array([2560,10240,81920, 5120,163840])
  i=1
  for k in range(1,4):
    m=M[k-1]
    for j in range(1,4):
      n=N[j-1]
      thetaNbNoRND=np.load('../../../../OutputFiles/OutputbNoRND'+str(m)+'Refs'+str(n)+'n.npy')
      thetaNbRND=np.load('../../../../OutputFiles/OutputbRND'+str(m)+'Refs'+str(n)+'n.npy')
      thetaNaNoRND=np.load('../../../../OutputFiles/OutputaNoRND'+str(m)+'Refs'+str(n)+'n.npy')
      thetaNaRND=np.load('../../../../OutputFiles/OutputaRND'+str(m)+'Refs'+str(n)+'n.npy')
      #thetaNbNoRND=thetaNbNoRND[thetaNbNoRND!=0]
      #thetaNbRND=thetaNbRND[thetaNbRND!=0]
      #thetaNaNoRND=thetaNaNoRND[thetaNaNoRND!=0]
      #thetaNaRND=thetaNaRND[thetaNaRND!=0]
      i+=1
      mp.figure(i)
      p1=mp.plot(np.log(thetaNbNoRND[2][1:-1]),np.log(thetaNaNoRND[1][1:-1]),marker='x',c='r',label='no phase change on ref')#,np.log(thetaNa[2]))
      p2=mp.plot(np.log(thetaNbRND[2][1:-1]),np.log(thetaNaRND[1][1:-1]),marker='x',c='b',label= 'rnd phase change on ref')#,np.log(thetaNa[2]))
      mp.legend()#mp.legend(handles=(p1,p2),labels=('no phase change on ref','rnd phase change on ref'), loc='upper left')
      mp.xlabel('Log Number of rays')
      mp.ylabel('Difference in power between n and 2n at reference point in Dbm')
      mp.title('Difference in Log(phi)_(0,0) for n, 2n rays, against n')
      mp.savefig('ConeFigures/thetaNagainstN1loc'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(np.log(thetaNbNoRND[2][1:-1]),np.log(thetaNbNoRND[1][1:-1]),marker='x',c='r',label='no phase change on ref')#,np.log(thetaNb[2]))
      p2=mp.plot(np.log(thetaNbRND[2][1:-1]),np.log(thetaNbRND[1][1:-1]),marker='x',c='b',label= 'rnd phase change on ref')#,np.log(thetaNa[2]))
      mp.legend()#handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Log number of rays')
      mp.ylabel('Difference in power between n and 2n at reference point in Dbm')
      mp.title('Log(diff(phi))_(0,0) for n, 2n rays, against log(n)')
      mp.savefig('ConeFigures/thetaNagainstN2loc'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(thetaNbNoRND[2],thetaNaNoRND[0],marker='x',c='r',label='no phase change on ref')
      p2=mp.plot(thetaNbRND[2]  ,thetaNaRND[0]  ,marker='x',c='b',label='rnd phase change on ref')
      mp.legend()#handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Number of rays')
      mp.ylabel('Power at reference point in Dbm')
      mp.title('Power against n')
      mp.savefig('ConeFigures/Power1VN'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(thetaNbNoRND[2],thetaNbNoRND[0],marker='x',c='r',label='no phase change on ref')
      p2=mp.plot(thetaNbRND[2]  ,thetaNbRND[0]  ,marker='x',c='b',label='rnd phase change on ref')
      mp.legend()#handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Number of rays')
      mp.ylabel('Power at reference point in Dbm')
      mp.title('Power against n')
      mp.savefig('ConeFigures/Power2VN'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(thetaNbNoRND[2],thetaNbNoRND[5],marker='x',c='r',label='no phase change on ref')
      p2=mp.plot(thetaNbRND[2]  ,thetaNbRND[5]  ,marker='x',c='b',label='rnd phase change on ref')
      mp.legend()#handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Number of rays')
      mp.ylabel('Power at reference point in Dbm divided by n')
      mp.title('Power over n against n')
      mp.savefig('ConeFigures/PowerVoverN2N'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(thetaNaNoRND[2],thetaNaNoRND[5],marker='x',c='r',label='no phase change on ref')
      p2=mp.plot(thetaNaRND[2]  ,thetaNaRND[5]  ,marker='x',c='b',label='rnd phase change on ref')
      mp.legend()#handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Number of rays')
      mp.ylabel('Power at reference point in Dbm divided by n')
      mp.title('Power over n against n')
      mp.savefig('ConeFigures/PowerVoverN1N'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      mp.plot(thetaNbNoRND[2][2:-1],thetaNaNoRND[4][2:-1],marker='x',c='r',label='no phase change on ref')#,np.log(thetaNa[2]))
      mp.plot(thetaNbRND[2][2:-1],thetaNaRND[4][2:-1],marker='x',c='b',label='rnd phase change on ref')#,np.log(thetaNa[2]))
      mp.legend()#handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Number of rays')
      mp.ylabel('log of the ratio of the difference in power between n and 2n, and between n/2 and n, divided by log(2)')
      mp.title('log(thetaN/theta2N)/log(2) against n')
      mp.savefig('ConeFigures/alpha1loc'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(thetaNbNoRND[2][2:-1],thetaNbNoRND[4][2:-1],marker='x',c='r',label='no phase change on ref')#,np.log(thetaNb[2]))
      p2=mp.plot(thetaNbRND[2][2:-1],thetaNbRND[4][2:-1],marker='x',c='b',label='rnd phase change on ref')#,np.log(thetaNb[2]))
      mp.legend() #handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Number of rays')
      mp.ylabel('log of the ratio of the difference in power between n and 2n, and between n/2 and n, divided by log(2)')
      mp.title('log(thetaN/theta2N)/log(2) against n')
      mp.savefig('ConeFigures/alpha2loc'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      mp.plot(thetaNbNoRND[2][1:-2],thetaNaNoRND[1][2:-1]/thetaNaNoRND[1][1:-2],marker='x',c='r',label='no phase change on ref')#,np.log(thetaNa[2]))
      mp.plot(thetaNbRND[2][1:-2],thetaNaRND[1][2:-1]/thetaNaRND[1][1:-2],marker='x',c='b',label='rnd phase change on ref')#,np.log(thetaNa[2]))
      mp.legend() #handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.ylabel('The ratio of the difference in power between n and 2n, and between n/2 and n')
      mp.title('thetaN/theta2N against n')
      mp.savefig('ConeFigures/Ratio1loc'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(thetaNbNoRND[2][1:-2],thetaNbNoRND[1][2:-1]/thetaNbNoRND[1][1:-2],marker='x',c='r',label='no phase change on ref')#,np.log(thetaNb[2]))
      p2=mp.plot(thetaNbRND[2][1:-2],thetaNbRND[1][2:-1]/thetaNbRND[1][1:-2],marker='x',c='b',label='rnd phase change on ref')#,np.log(thetaNb[2]))
      mp.legend() #handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('Number of rays')
      mp.ylabel('Ratio of the difference in power between n and 2n, and between n/2 and n')
      mp.title('thetaN/theta2N against n')
      mp.savefig('ConeFigures/Ratio2loc'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      mp.plot(np.log(thetaNbNoRND[2][1:-2]),np.log(thetaNaNoRND[1][2:-1]),marker='x',c='r',label='no phase change on ref')#,np.log(thetaNa[2]))
      mp.plot(np.log(thetaNbRND[2][1:-2]),np.log(thetaNaRND[1][2:-1]),marker='x',c='b',label='rnd phase change on ref')#,np.log(thetaNa[2]))
      mp.legend() #handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('log n')
      mp.ylabel('log theta')
      mp.title('log theta log n')
      mp.savefig('ConeFigures/logthetalogn1loc'+str(m)+'Refs'+str(n)+'n.png')
      i+=1
      mp.figure(i)
      p1=mp.plot(np.log(thetaNbNoRND[2][1:-2]),np.log(thetaNbNoRND[1][2:-1]),marker='x',c='r',label='no phase change on ref')#,np.log(thetaNb[2]))
      p2=mp.plot(np.log(thetaNbRND[2][1:-2]),np.log(thetaNbRND[1][2:-1]),marker='x',c='b',label='rnd phase change on ref')#,np.log(thetaNb[2]))
      mp.legend() #handles=[p1,p2],labels=['no phase change on ref','rnd phase change on ref'], loc='upper left')
      mp.xlabel('log n')
      mp.ylabel('log theta')
      mp.title('log theta log n')
      mp.savefig('ConeFigures/logthetalogn2loc'+str(m)+'Refs'+str(n)+'n.png')

      # Heatmap plots
      grid1aRND=np.load('../../../../Heatmapgrids/HeatmapaRND'+str(m)+'Refs'+str(n)+'n.npy')
      grid1aNoRND=np.load('../../../../Heatmapgrids/HeatmapaNoRND'+str(m)+'Refs'+str(n)+'n.npy')
      grid1bRND=np.load('../../../../Heatmapgrids/HeatmapbRND'+str(m)+'Refs'+str(n)+'n.npy')
      grid1bNoRND=np.load('../../../../Heatmapgrids/HeatmapbNoRND'+str(m)+'Refs'+str(n)+'n.npy')
      i+=1
      mp.figure(i)
      mp.imshow(grid1aNoRND, cmap='viridis', interpolation='nearest')
      cbar=mp.colorbar()
      cbar.set_label('Power in dBm', rotation=270)
      mp.savefig('ConeFigures/ConeNoPhaseHeatmap1Loc'+str(m)+'Refs'+str(n)+'n.png',bbox_inches='tight')
      i+=1
      mp.figure(i)
      mp.imshow(grid1aRND, cmap='viridis', interpolation='nearest', vmin=-90, vmax=20)
      cbar=mp.colorbar()
      cbar.set_label('Power in dBm', rotation=270)
      mp.savefig('ConeFigures/ConeAveragedHeatmapwithPhase1Loc'+str(m)+'Refs'+str(n)+'n.png',bbox_inches='tight')
      i+=1
      mp.figure(i)
      mp.imshow(grid1bNoRND, cmap='viridis', interpolation='nearest', vmin=-90, vmax=20)
      cbar=mp.colorbar()
      cbar.set_label('Power in dBm', rotation=270)
      mp.savefig('ConeFigures/ConeNoPhaseHeatmap2Loc'+str(m)+'Refs'+str(n)+'n.png',bbox_inches='tight')
      i+=1
      mp.figure(i)
      mp.imshow(grid1bRND, cmap='viridis', interpolation='nearest', vmin=-90, vmax=20)
      cbar=mp.colorbar()
      cbar.set_label('Power in dBm', rotation=270)
      mp.savefig('ConeFigures/ConeAveragedHeatmapwithPhase2Loc'+str(m)+'Refs'+str(n)+'n.png',bbox_inches='tight')


      # Residual Plots
      grid1aRND=np.load('../../../../Resigrids/ResiaRND'+str(m)+'Refs'+str(n)+'n.npy')
      grid1aNoRND=np.load('../../../../Resigrids/ResiaNoRND'+str(m)+'Refs'+str(n)+'n.npy')
      grid1bRND=np.load('../../../../Resigrids/ResibRND'+str(m)+'Refs'+str(n)+'n.npy')
      grid1bNoRND=np.load('../../../../Resigrids/ResibNoRND'+str(m)+'Refs'+str(n)+'n.npy')
      i+=1
      mp.figure(i)
      mp.imshow(grid1bNoRND, cmap='viridis', interpolation='nearest', vmin=0, vmax=15)
      cbar=mp.colorbar()
      mp.title('Residual- for %s rays and %s rays' %(n/2,n))
      mp.savefig('ConeFigures/ConeResidualbn'+str(n/2)+'and2n'+str(n)+'.png',bbox_inches='tight')
      i+=1
      mp.figure(i)
      mp.imshow(grid1bRND, cmap='viridis', interpolation='nearest', vmin=0, vmax=15)
      cbar=mp.colorbar()
      mp.title('Residual RND- for %s rays and %s rays' %(n/2,n))
      mp.savefig('ConeFigures/ConeResidualRNDbn'+str(n/2)+'and2n'+str(n)+'.png',bbox_inches='tight')
      i+=1
      mp.figure(i)
      mp.imshow(grid1aNoRND, cmap='viridis', interpolation='nearest', vmin=0, vmax=15)
      cbar=mp.colorbar()
      mp.title('Residual- for %s rays and %s rays' %(n/2,n))
      mp.savefig('ConeFigures/ConeResidualan'+str(n/2)+'and2n'+str(n)+'.png',bbox_inches='tight')
      i+=1
      mp.figure(i)
      mp.imshow(grid1aRND, cmap='viridis', interpolation='nearest', vmin=0, vmax=15)
      cbar=mp.colorbar()
      mp.title('Residual RND- for %s rays and %s rays' %(n/2,n))
      mp.savefig('ConeFigures/ConeResidualRNDan'+str(n/2)+'and2n'+str(n)+'.png',bbox_inches='tight')
     # print(thetaNaNoRND)
      #print(thetaNbNoRND)
      #print(thetaNaRND)
      #print(thetaNbRND)
      #mp.show()
      # TEST err=rtest.ray_tracer_test(Room, origin)
      # TEST PRINT print('error after rtest on room', err)
      mp.close('all')
  exit()

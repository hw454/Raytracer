#!/usr/bin/env python3
# Hayley Wragg 2018-05-29
import numpy as np
import sys
import os
import matplotlib.pyplot as mp

def PlotTimes():
    ##Plot the obstacles and the room first

    ##----Retrieve the Raytracing Parameters-----------------------------
    Nrao,Nre,h ,L    =np.load('Parameters/Raytracing.npy')

    #Nra=int(np.sqrt(Nrao/2.0)-1)*int(np.sqrt(2.0*Nrao))+1
    Nra=int(Nrao)

    roomnumstat=np.load('roomnumstat.npy')
    Roomnum    =np.load('Roomnum.npy')


    ##---Retrieve the Timematrix ---------------------------------------
    timename=('./Times/TimesNra'+str(Nra)+'Refs'+str(int(Nre))+'Roomnum'+str(int(roomnumstat))+'to'+str(int(Roomnum))+'.npy')
    Timemat=np.load(timename)
    x=Timemat[:,0]
    hay=Timemat[:,3]
    std=Timemat[:,4]
    mp.figure(1)
    mp.plot(x,hay,label='GRL')
    mp.plot(x,std,label='Std')
    mp.title('Time taken (s) to computer power against no. of parameter sets')
    mp.legend()
    if not os.path.exists('./Times'):
      os.makedirs('./Times')
    filename=str('Times/TimePlot'+str(int(Nra))+'Nref'+str(int(Nre))+'.eps')
    mp.savefig(filename)
    mp.clf()
    return

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  PlotTimes()
exit()

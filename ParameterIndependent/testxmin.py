#!/usr/bin/env python3

import numpy as np
import ParameterInput as PI

out=PI.DeclareParameters()
Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
Oblist        =np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain
RoomP=Oblist[0]
for j in range(1,len(Oblist)):
  RoomP=np.concatenate((RoomP,Oblist[j]),axis=0)
maxpoint=np.array([np.min(RoomP,axis=0),np.max(RoomP,axis=0)])
print(maxpoint)

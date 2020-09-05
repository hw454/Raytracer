#!/usr/bin/env python3
# Keith Briggs 2019-02-07

import numpy as np

class X:
  def __init__(s,nr,nc):
    s.x=np.empty((nr,nc))
    # for testing only...
    k=0
    for r in range(nr):
      for c in range(nc):
        s.x[r,c]=k; k+=1
  def __getitem__(s,i):
    if type(i) is type(1): # row
      return s.x[i] 
    if len(i)==2: # single element
      return s.x[i] 
    if isinstance(i[0],slice): # column
      #if i[0]==slice(None,None,None):
      return s.x[i[0],i[1]]
      #else:
        #pass # TODO - handle slice with start:stop:step
    if isinstance(i[1],slice): # row
      if i[1]==slice(None,None,None):
        return s.x[:,i[0]]
      else:
        pass # TODO - handle slice with start:stop:step
    # TODO handle case [:,:]
    else:
      raise 'indexing error!'

def test_00():
  x=X(6,4)
  print(x[1])
  print(x[1,2])
  print(x[1:3,1])
  print(x[:,1])
  print(x[1:,1])
  print(x[1:6:2,1])
  print(x[2,:])

if __name__=='__main__':
  test_00()

#!/usr/bin/env python3
# Hayley Wragg 2017-05-19
''' MPython Programm to test the modules. '''

import intersection as inter
import HayleysPlotting as hp
import reflection as ref
import numpy as np


if __name__=='__main__':
  np.set_printoptions(precision=2,threshold=1e-12,suppress=True)
  test=inter.test()
  print('Intersection Test 2D', test)
  inter.test3D()
  err=ref.test()
  err=ref.test3D(err)
  print('Reflection Test', err)
  exit()


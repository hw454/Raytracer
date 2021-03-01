#!/usr/bin/env python3
# H Wragg 26th January 2021
'''This code will profile the script 'scriptname' for calls and timing.
Initially developed for profiling the raytracer code in :py:mod:'RayTracerMainProgram.py'.
'scriptname'=:py:mod:'RayTracerMainProgram':py:func:'Main', which calls and varies
the inputs from :py:mod:'ParameterLoad' then runs :py:func:'MeshProgram'
'''
import cProfile as cP
import pstats
from pstats import SortKey
import io
import sys
import importlib
import RayTracerMainProgram as RT
import Rays as Ra

def main():
  pr=cP.Profile()
  pr.enable()
  #Ra.centre_dist_test()
  RT.main(sys.argv)
  pr.disable()
  s=io.StringIO()
  sortby=SortKey.CUMULATIVE
  ps=pstats.Stats(pr,stream=s).sort_stats(sortby)
  ps.print_stats(20)
  print(s.getvalue())
  ps.dump_stats('RayTracerProfile.dmp')

  return 0

if __name__=='__main__':
  main()
  exit()


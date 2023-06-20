#!/usr/bin/env python3
# Keith Briggs 2020-11-02

from sys import argv,stderr

def ray_trace(job,fn):
  print('ray_trace called with job=%3d, using output filename=%s'%(job,fn,))
  print(job//5, job%5)

def main(argv,verbose=False):
  job=0 # default job
  if len(argv)>1: job=int(argv[1])
  fn='ray_trace_output_%03d.txt'%job
  if verbose:
    print('main called with job=%3d, using output filename=%s'%(job,fn,))
  ray_trace(job,fn)

if __name__=='__main__':
  main(argv)

#!/usr/bin/env python3
# Hayley 2019-02-19

import numpy as np
import matplotlib.pyplot as mp

times=np.load('times.npy')
mp.figure(0)
p1=mp.plot(np.log(times[0][:]),times[1][:],marker='x',c='r',label='t1-dense then index')
p2=mp.plot(np.log(times[0][:]),times[2][:],marker='x',c='b',label= 't2-index then dense')
mp.xlabel('log-number of terms in matrix')
mp.ylabel('time')
mp.title('times for dense and index performance')
mp.legend(handles=(p1,p2),labels=('t1-dense then index','t2-index then dense'), loc='upper left')
mp.savefig('Times.png')
mp.show()


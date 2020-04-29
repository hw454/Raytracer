#!/usr/bin/env python3

import cv2
import numpy as np
import glob

def CombineImages(img_array,filename,Nslice):
  size=(0,0)
  for j in range(0,Nslice):
    filesave=str(filename+str(int(j))+'.jpg')
    img = cv2.imread(filesave)
    if isinstance(img,type(None)):
       break
    else:
       print(filesave)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
  return img_array,size

if __name__=='__main__':
    tp=1                               # Time pause in video
    Nra=np.load('./Parameters/Nra.npy')# Number of rays
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
    else:
      nra=len(Nra)
    for i in range(0,nra):
        Nslice=6
        filestart=str('./ConeFigures/Nra'+str(Nra[i])+'/Cone'+str(Nra[i]))
        img_array = []
        filename=str(filestart+'Square')
        img_array,size=CombineImages(img_array,filename,Nslice)
        Nslice=10
        filename=str(filestart+'FullSquareX')
        img_array,size=CombineImages(img_array,filename,Nslice)
        filename=str(filestart+'FullSquareY')
        img_array,size=CombineImages(img_array,filename,Nslice)
        filename=str(filestart+'FullSquareZ')
        img_array,size=CombineImages(img_array,filename,Nslice)

        videoname=str('./ConeFigures/Nra'+str(Nra[i])+'.avi')
        out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    exit()



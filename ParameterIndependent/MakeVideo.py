#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import subprocess

def CombineImages(img_array,filename,Nslice,  size=(0,0)):
  for j in range(0,Nslice):
    filesave=str(filename+str(int(j))+'.jpg')
    img = cv2.imread(filesave)
    if isinstance(img,type(None)):
      pass
    else:
      height, width, layers = img.shape
      size = (width,height)
      img_array.append(img)
  print(size)
  return img_array,size

if __name__=='__main__':
    tp=2                              # Time pause in video
    Nra=np.load('./Parameters/Nra.npy')# Number of rays
    delangle      =np.load('Parameters/delangle.npy')
    myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
    plottype= myfile.read()         # read the entire file into a string
    myfile.close()

    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
    else:
      nra=len(Nra)
    for i in range(0,nra):
        img_array = []
        filestart=str('./ConeFigures/'+plottype+'/Room'+str(Nra[i])+'.jpg')
        img = cv2.imread(filestart)
        if isinstance(img,type(None)):
          pass
        else:
          height, width, layers = img.shape
          size = (width,height)
          img_array.append(img)
        Nslice=int(2+np.pi/delangle[i])
        filestart=str('./ConeFigures/'+plottype+'/Cone'+str(Nra[i]))
        filename=str(filestart+'Square')
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        Nslice=10
        filename=str(filestart+'FullSquareX')
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        filename=str(filestart+'FullSquareY')
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        filename=str(filestart+'FullSquareZ')
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        videoname=str('./ConeFigures/'+plottype+'/Nra'+str(Nra[i])+'.avi')
        out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array)):
            out.write(img_array[j])
        out.release()
        img_array2=[]
        filename=str(filestart+'PowersliceX')
        img_array2,size=CombineImages(img_array2,filename,Nslice,size)
        filename=str(filestart+'PowersliceY')
        img_array2,size=CombineImages(img_array2,filename,Nslice,size)
        filename=str(filestart+'PowersliceZ')
        img_array2,size=CombineImages(img_array2,filename,Nslice,size)
        videoname2=str('./ConeFigures/'+plottype+'/NraPower'+str(Nra[i])+'.avi')
        out = cv2.VideoWriter(videoname2,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array2)):
            out.write(img_array2[j])
        out.release()
        videoname3=str('./ConeFigures/'+plottype+'/Nra'+str(Nra[i])+'Vid.avi')
        bashCommand = str("avimerge -o "+videoname3+" -i " +videoname+" "+videoname2)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        bashCommand=str("rm "+videoname+" "+videoname2)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        img_array4=[]
        filename=str(filestart+'PowerDiffsliceX')
        img_array4,size=CombineImages(img_array4,filename,Nslice,size)
        filename=str(filestart+'PowerDiffsliceY')
        img_array4,size=CombineImages(img_array4,filename,Nslice,size)
        filename=str(filestart+'PowerDiffsliceZ')
        img_array4,size=CombineImages(img_array4,filename,Nslice,size)
        videoname4=str('./ConeFigures/'+plottype+'/NraPowerDiff'+str(Nra[i])+'.avi')
        out = cv2.VideoWriter(videoname4,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array4)):
            out.write(img_array4[j])
        out.release()
        img_array5=[]
        filename=str(filestart+'RadAX')
        img_array5,size=CombineImages(img_array5,filename,Nslice,size)
        filename=str(filestart+'RadAY')
        img_array5,size=CombineImages(img_array5,filename,Nslice,size)
        filename=str(filestart+'RadAZ')
        img_array5,size=CombineImages(img_array5,filename,Nslice,size)
        videoname5=str('./ConeFigures/'+plottype+'/RadA'+str(Nra[i])+'.avi')
        out = cv2.VideoWriter(videoname5,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array5)):
            out.write(img_array5[j])
        out.release()
        img_array6=[]
        filename=str(filestart+'RadBX')
        img_array6,size=CombineImages(img_array6,filename,Nslice,size)
        filename=str(filestart+'RadBY')
        img_array6,size=CombineImages(img_array6,filename,Nslice,size)
        filename=str(filestart+'RadBZ')
        img_array6,size=CombineImages(img_array6,filename,Nslice,size)
        videoname6=str('./ConeFigures/'+plottype+'/RadB'+str(Nra[i])+'.avi')
        out = cv2.VideoWriter(videoname6,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array6)):
            out.write(img_array6[j])
        out.release()
        videoname3=str('./ConeFigures/'+plottype+'/RadBoth'+str(Nra[i])+'Vid.avi')
        bashCommand = str("avimerge -o "+videoname3+" -i " +videoname5+" "+videoname6)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    filestart=str('./ConeFigures/'+plottype+'/')
    img_array3=[]
    filename=str(filestart+'TruesliceX')
    img_array3,size=CombineImages(img_array3,filename,Nslice,size)
    filename=str(filestart+'TruesliceY')
    img_array3,size=CombineImages(img_array3,filename,Nslice,size)
    filename=str(filestart+'TruesliceZ')
    img_array3,size=CombineImages(img_array3,filename,Nslice,size)
    videoname3=str('./ConeFigures/'+plottype+'/True.avi')
    out = cv2.VideoWriter(videoname3,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array3)):
      out.write(img_array3[j])
    out.release()
    exit()



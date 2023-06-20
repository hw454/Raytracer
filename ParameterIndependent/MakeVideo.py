#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import subprocess
import os

def CombineImages(img_array,filename,Nslice,  size=(0,0)):
  for j in range(0,Nslice):
    filesave=filename+'%02d.jpg'%j
    img = cv2.imread(filesave)
    if isinstance(img,type(None)):
      pass
    else:
      height, width, layers = img.shape
      size = (width,height)
      img_array.append(img)
  return img_array,size

def CombineImages2(img_array,filename,Nslice,  size=(0,0)):
  for j in range(1,Nslice+1):
    filesave=filename+'%dof%d.jpg'%(j,Nslice)
    img = cv2.imread(filesave)
    if isinstance(img,type(None)):
      print('noimg',filesave)
    else:
      height, width, layers = img.shape
      size = (width,height)
      img_array.append(img)
  return img_array,size

def RayHeatMaps():
    tp=2                              # Time pause in video
    Nra       =np.load('Parameters/Nra.npy')# Number of rays
    delangle  =np.load('Parameters/delangle.npy')
    Nsur      =np.load('Parameters/Nsur.npy')
    Nre=2#,h,L,split    =np.load('Parameters/Raytracing.npy')
    myfile    =open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
    plottype  =myfile.read()         # read the entire file into a string
    myfile.close()
    ResOn     =np.load('Parameters/ResOn.npy')
    InnerOb   =np.load('Parameters/InnerOb.npy')
    Box='Box' if InnerOb else 'NoBox'
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
    else:
      nra=len(Nra)
    for i in range(0,nra):
        img_array = []
        filestart='./ConeFigures/'+plottype+'/'+Box+'Room%d.jpg'%Nra[i]
        img = cv2.imread(filestart)
        if isinstance(img,type(None)):
          size=(0,0)
        else:
          height, width, layers = img.shape
          size = (width,height)
          img_array.append(img)
        Nslice        =int(2+np.pi/delangle[i])
        filestart     ='./ConeFigures/'+plottype+'/Cone%d'%Nra[i]
        filename      =filestart+'Square'
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        Nslice        =np.load('./Parameters/Ns.npy')
        filename      =filestart+'FullSquareX'
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        filename      =filestart+'FullSquareY'
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        filename      =filestart+'FullSquareZ'
        img_array,size=CombineImages(img_array,filename,Nslice,size)
        videoname     ='./ConeFigures/'+plottype+'/Nra%d.avi'%Nra[i]
        out           = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array)):
            out.write(img_array[j])
        out.release()
        img_array2=[]
        filename       =filestart+'PowersliceX'
        img_array2,size=CombineImages(img_array2,filename,Nslice,size)
        filename       =filestart+'PowersliceY'
        img_array2,size=CombineImages(img_array2,filename,Nslice,size)
        filename       =filestart+'PowersliceZ'
        img_array2,size=CombineImages(img_array2,filename,Nslice,size)
        videoname2     ='./ConeFigures/'+plottype+'/NraPower%d.avi'%Nra[i]
        out = cv2.VideoWriter(videoname2,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array2)):
            out.write(img_array2[j])
        out.release()
        videoname3   ='./ConeFigures/'+plottype+'/Nra%dVid.avi'%Nra[i]
        bashCommand  ="avimerge -o "+videoname3+" -i " +videoname+" "+videoname2
        process      =subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error= process.communicate()
        #bashCommand  ="rm "+videoname+" "+videoname2
        #process      = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        #output, error= process.communicate()
        if ResOn:
          img_array4=[]
          filename       =filestart+'PowerDiffsliceX'
          img_array4,size=CombineImages(img_array4,filename,Nslice,size)
          filename       =filestart+'PowerDiffsliceY'
          img_array4,size=CombineImages(img_array4,filename,Nslice,size)
          filename       =filestart+'PowerDiffsliceZ'
          img_array4,size=CombineImages(img_array4,filename,Nslice,size)
          videoname4     ='./ConeFigures/'+plottype+'/NraPowerDiff%d.avi'%Nra[i]
          out = cv2.VideoWriter(videoname4,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
          for j in range(len(img_array4)):
              out.write(img_array4[j])
          out.release()
        img_array5=[]
        filename       =filestart+'RadAX'
        img_array5,size=CombineImages(img_array5,filename,Nslice,size)
        filename       =filestart+'RadAY'
        img_array5,size=CombineImages(img_array5,filename,Nslice,size)
        filename       =filestart+'RadAZ'
        img_array5,size=CombineImages(img_array5,filename,Nslice,size)
        videoname5     ='./ConeFigures/'+plottype+'/RadA%d.avi'%Nra[i]
        out = cv2.VideoWriter(videoname5,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(img_array5)):
            out.write(img_array5[j])
        out.release()
        videoname3  ='./ConeFigures/'+plottype+'/RadBoth%dVid.avi'%Nra[i]
        bashCommand = "avimerge -o "+videoname3+" -i " +videoname5
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        for l in range(Nsur):
          img_array6=[]
          filename       =filestart+'RadS%dX'%l
          img_array6,size=CombineImages(img_array6,filename,Nslice,size)
          filename       =filestart+'RadS%dY'%l
          img_array6,size=CombineImages(img_array6,filename,Nslice,size)
          filename       =filestart+'RadS%dZ'%l
          img_array6,size=CombineImages(img_array6,filename,Nslice,size)
          videoname6     ='./ConeFigures/'+plottype+'/RadS%dNra%d.avi'%(l,Nra[i])
          out = cv2.VideoWriter(videoname6,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
          for j in range(len(img_array6)):
            out.write(img_array6[j])
          out.release()
          videoname3  ='./ConeFigures/'+plottype+'/RadBoth%dVid.avi'%Nra[i]
          bashCommand = "avimerge -o "+videoname3+" -i " +videoname3+" "+videoname6
          process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
          output, error = process.communicate()
          img_array = []
          #NoBoxPowerSliceNra22Nref2slice1of9.jpg
        Nslice=5
        folder='./GeneralMethodPowerFigures/Cube'+Box+'/'+plottype+'/PowerSlice/Nra%d/Nref%d'%(Nra[i],2)
        if not os.path.exists(folder):
          os.makedirs(folder)
          print('Nofolder')
        flatimg_array=[]
        flatfilestart     =folder+'/'+Box+'PowerSliceNra%dNref%dslice'%(Nra[i],Nre)
        flatimg_array,size=CombineImages2(flatimg_array,flatfilestart,Nslice,size)
        flatvideoname     =folder+'.avi'
        out           = cv2.VideoWriter(flatvideoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(flatimg_array)):
            out.write(flatimg_array[j])
        out.release()
        Nslice=9
        folder='./GeneralMethodPowerFigures/Cuboid'+Box+'/'+plottype+'/PowerSlice/Nra%d'%(Nra[i])
        if not os.path.exists(folder):
          os.makedirs(folder)
          print('Nofolder')
        flatimg_array=[]
        flatfilestart     =folder+'/'+Box+'PowerSliceNra%dNref%dslice'%(Nra[i],Nre)
        flatimg_array,size=CombineImages2(flatimg_array,flatfilestart,Nslice,size)
        flatvideoname     =folder+'.avi'
        out           = cv2.VideoWriter(flatvideoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
        for j in range(len(flatimg_array)):
            out.write(flatimg_array[j])
        out.release()
    if ResOn:
      filestart='./ConeFigures/'+plottype+'/'
      img_array3=[]
      filename       =filestart+'TruesliceX'
      img_array3,size=CombineImages(img_array3,filename,Nslice,size)
      filename       =filestart+'TruesliceY'
      img_array3,size=CombineImages(img_array3,filename,Nslice,size)
      filename       =filestart+'TruesliceZ'
      img_array3,size=CombineImages(img_array3,filename,Nslice,size)
      videoname3     ='./ConeFigures/'+plottype+'/True.avi'
      out = cv2.VideoWriter(videoname3,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
      for j in range(len(img_array3)):
        out.write(img_array3[j])
      out.release()
    return 1

def TransmitterMovie():
  tp=1 # Time pause for video
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  InnerOb    =np.load('Parameters/InnerOb.npy')
  if InnerOb:
      Box='Box'
  else:
      Box='NoBox'
  #numjobs    =np.load('Parameters/Numjobs.npy')
  myfile = open('Parameters/runplottype.txt', 'rt') # open lorem.txt for reading text
  plottype= myfile.read()         # read the entire file into a string
  myfile.close()
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
  else:
      nra=len(Nra)
  for i in range(nra):
    Nr=Nra[i]
    QP=np.load('./Mesh/'+plottype+'/'+Box+'QualityPercentile%03dRefs%03dm%03d_txAll.npy'%(Nr,Nre,0))
    Nz=QP.shape[2]
    filestart='./Quality/'+plottype+'/'+Box+'Qualitysurface%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    img_array=[]
    img_array,size=CombineImages(img_array,filestart,Nz)
    videoname='./Quality/'+plottype+'/'+Box+'Qualitysurface%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
      out.write(img_array[j])
    #print(img_array)
    out.release()
    filestart='./Quality/'+plottype+'/'+Box+'QualityPercentileSurface%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    img_array=[]
    img_array,size=CombineImages(img_array,filestart,Nz,size)
    videoname='./Quality/'+plottype+'/'+Box+'QualityPercentileSurface%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
      out.write(img_array[j])
    out.release()
    filestart='./Quality/'+plottype+'/'+Box+'QualityMinSurface%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    img_array=[]
    img_array,size=CombineImages(img_array,filestart,Nz,size)
    videoname='./Quality/'+plottype+'/'+Box+'QualityMinSurface%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
      out.write(img_array[j])
    out.release()
    filestart='./Quality/'+plottype+'/'+Box+'QualityContour%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    img_array=[]
    img_array,size=CombineImages(img_array,filestart,Nz,size)
    videoname='./Quality/'+plottype+'/'+Box+'QualityContour%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
      out.write(img_array[j])
    out.release()
    filestart='./Quality/'+plottype+'/'+Box+'QualityPercentileContour%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    img_array=[]
    img_array,size=CombineImages(img_array,filestart,Nz,size)
    videoname='./Quality/'+plottype+'/'+Box+'QualityPercentileContour%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
      out.write(img_array[j])
    out.release()
    filestart='./Quality/'+plottype+'/'+Box+'QualityMinContour%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    img_array=[]
    img_array,size=CombineImages(img_array,filestart,Nz,size)
    videoname='./Quality/'+plottype+'/'+Box+'QualityMinContour%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
      out.write(img_array[j])
    out.release()
    # filestart='./Quality/'+plottype+'/'+Box+'QualityPercentileBothSurface%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    # img_array=[]
    # img_array,size=CombineImages(img_array,filestart,Nz,size)
    # videoname='./Quality/'+plottype+'/'+Box+'QualityPercentileBothSurface%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    # out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    # for j in range(len(img_array)):
      # out.write(img_array[j])
    # out.release()
    # # filestart='./Quality/'+plottype+'/'+Box+'QualityBothSurface%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    # img_array=[]
    # img_array,size=CombineImages(img_array,filestart,Nz,size)
    # videoname='./Quality/'+plottype+'/'+Box+'QualityBothSurface%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    # out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    # for j in range(len(img_array)):
      # out.write(img_array[j])
    # out.release()
    filestart='./Times/'+plottype+'/'+Box+'TimeSurface%03dto%03dNref%03d_z'%(Nra[0],Nra[-1],Nre)
    img_array=[]
    img_array,size=CombineImages(img_array,filestart,Nz,size)
    videoname='./Times/'+plottype+'/'+Box+'TimesSurface%03dto%03dNref%03d.avi'%(Nra[0],Nra[-1],Nre)
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
      out.write(img_array[j])
    out.release()
  return 1

if __name__=='__main__':
    #RayHeatMaps()
    TransmitterMovie()
    exit()



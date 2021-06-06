# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:37:56 2020

@author: Ai-fi
"""
import cv2
import numpy as np
import glob
#import os


def Diff_Cauls(preimg, curreimg):
      
      #resize im_diff_center with 256 by 256
      preimg = cv2.resize(preimg, size)
      curreimg = cv2.resize(curreimg, size)
      
      im_diff = preimg.astype(int) - curreimg.astype(int)
      im_diff = im_diff / im_diff.max() * 255
      #Cast to int32 to uint8
      #im_diff = im_diff.astype('uint8')
      #fit in range 0~255(center is 128)
    
      return im_diff


i = -1  
img = list()
path = './results/video/sunglass_cyclegan/test_130/images/A_0000_fake.png'
f_img = cv2.imread(path)

size = (256,256)
size_ = (256,256,3)
 
d_img = np.empty(size_ , dtype='u1')
n_img  = np.empty(size_ , dtype='u1')
f_img = cv2.resize(f_img ,size)
H, W, C = size_

diff_re = list()
di_im_ls = list()

for imagefile in glob.glob('./datasets/Mve_frame/testA/*.png'):
    img.append(cv2.imread(imagefile))
    i = i + 1
    if i ==0:
       di_im_ls.append(f_img)
    if i == 1:
      #一枚目はGANを読み込ませる
      print("read image")
      print(i)
      di_img = Diff_Cauls(img[i-1], img[i])
      
          
      for y in range(H):
         for x in range(W):
            for z in range(C):
              value = f_img[y][x][z] + di_img[y][x][z]
              if value < 256:
                 d_img[y][x][z] = value
              else:
                 d_img[y][x][z] = 255
              if value < 0:
                  d_img[y][x][z] = 0
                  
      d_img = d_img.astype('uint8')
      filename = './datasets/Mve_frame/n_im/d_img' + str(i).zfill(5) + '.png'
      cv2.imwrite(filename ,d_img)
      di_im_ls.append(d_img)
      
    if i > 1:
        
       di_img = Diff_Cauls(img[i-1], img[i])
      
       if di_img.mean() > 0:
          key_path = './results/video/sunglass_cyclegan/test_130/images/A_{0:04d}_fake.png'.format(i)
          print(str(i) + '回目')
          f_img = cv2.imread(key_path)
          f_img = d_img.astype(int)
       else:
           
          f_img = di_im_ls[i-1].astype(int)
          
       print(i)    
       for y in range(H):
         for x in range(W):
            for z in range(C):
                
              value = f_img[y][x][z] + di_img[y][x][z]
              
              if value < 256:
                 d_img[y][x][z] = value
              else:
                 d_img[y][x][z] = 255
              if value < 0:
                 d_img[y][x][z] = 0
      
       d_img = d_img.astype('uint8')
       di_im_ls.append(d_img)
       filename = './datasets/Mve_frame/n_im/d_img' + str(i).zfill(5) + '.png'
       cv2.imwrite(filename ,di_im_ls[i-1])

#-------save key frames ---------------------------------------------------------

for a in range(0, len(diff_re)):
    
    filenames = './datasets/Mve_frame/key/key_img' + str(a).zfill(5) + '.png'
    diff_im = diff_re[a]
    cv2.imwrite(filenames, diff_im) 
    
#-------------------------------------------------------------------------------     
#print(f_img.size)
#print(re_img.size)
    

          
      
          
      
 
    
        
    

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:42:50 2020

@author: Ai-fi
"""

import cv2
import numpy as np
import glob 




#変数の宣言-------

fake_path = './results/video/sunglass_cyclegan/test_130/images/*.png'
real_path = './datasets/Mve_frame/testA/*.png'

gan_path ='./results/video/sunglass_cyclegan/test_130/images/A_0000_fake.png'
re_path = './datasets/Mve_frame/testA/A_0000.png'

#サングラスの画像を格納する配列
r_im_ls = list()
f_im_ls = list()
i = 0
#GANの読み込み
fake_im = cv2.imread(gan_path)
#サングラスの最初の画像
re_fis_im = cv2.imread(re_path)

#リサイズ用
size = (256,256)
size_ = (256, 256, 3)
#for文用のサイズを取得
H,W,C = fake_im.shape
#空の画像用の配列を用意
d_img = np.empty(size_ , dtype='u1')
pos = np.empty(3, dtype='uint8')


#dictの宣言
color = {}
n = 0
#listの宣言
d_im_pose_ls = list()

#関数----------------------
def im_value(fake, real):
    #RGB 16進数
    H, W, C = fake.shape
    fake_str = ''
    real_str = ''
    
    for y in range(H):
        for x in range(W):
            fake_value = fake[y,x,:] # 画素値の取得
            real_value = real[y,x,:] 
            fake_str = ''
            real_str = ''
            for i in range(3):#6桁になるように０で埋める
                if len(str(format(fake_value[i], 'x'))) == 2:
                  fake_str = fake_str + str(format(fake_value[i], 'x'))
                else :
                  fake_str = fake_str + '0' +str(format(fake_value[i], 'x')) 
                if len(str(format(real_value[i], 'x'))) == 2:
                    real_str = real_str + str(format(real_value[i], 'x'))
                else :
                    real_str = real_str + '0' + str(format(real_value[i], 'x'))
            color[real_str] = fake_str 
    return  color
#-------
def dic_r(pos, f_im, y, x):#画素値を検索し変換する関数
    
   sungl_str = ''
   fake_str = ''
   
   for n in range(3):
     if len(str(format(pos[n], 'x'))) == 2 :
         sungl_str = sungl_str + str(format(pos[n], 'x'))
     else:
         sungl_str = sungl_str + '0' +str(format(pos[n], 'x'))
   if sungl_str in color:
      value = color[sungl_str]
   else :#6桁になるように０で埋める
      for z in range(3):
       f_im[y][x][z]
       if len(str(format(f_im[y][x][z], 'x'))) == 2:
         fake_str = fake_str + str(format(f_im[y][x][z], 'x'))
       else:
         fake_str = fake_str + '0' + str(format(f_im[y][x][z], 'x'))
      value = color.get(sungl_str, fake_str)
      color[sungl_str] = fake_str
      
   return value

def HexToTen(dict_pos):#変換した16進数を10進数に変える
    pos_str = dict_pos
    ret_pos = list()
    
    for i in range(3):#2文字ずつとって変換
       
       s = i*2
       pos = pos_str[s:s+2]
             
       pos = int(pos, 16)
       ret_pos.append(pos)
    
    return ret_pos

#処理の部分------------------------

#辞書の作成
im_value(fake_im, re_fis_im)


#対応する一枚目画像ペアをそれぞれ読み込む
for fake_img in glob.glob(fake_path):
    fake_img = cv2.imread(fake_img)
    f_im_ls.append(fake_img)
    
for real_img in glob.glob(real_path):#画像を読み込んでリサイズ後配列に格納
    real_img = cv2.imread(real_img)
    real_img = cv2.resize(real_img, size)
    r_im_ls.append(real_img)
    
    print(str(i)+'回目')
    r_im = r_im_ls[i]
    
    print(r_im_ls[i].mean())
    
    for y in range(H):    
       for x in range(W):
           for z in range(C):
               pos[z] =  r_im[y][x][z]
           value = dic_r(pos, f_im_ls[i], y, x) 
           d_im_pose_ls = HexToTen(value)
           
           for n in range(3):
               d_img[y][x][n] = d_im_pose_ls[n]
               
       
    print(d_img.mean())        
    #画像の保存 
    d_img = d_img.astype('uint8')
    filename = './datasets/Mve_frame/n_im/real_img' + str(i).zfill(5) + '.png'
    cv2.imwrite(filename, d_img)
    i = i + 1



# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:29:57 2020

@author: Ai-fi
"""

import cv2
import glob
import numpy as np

real_path = './datasets/Mve_frame/testA/*.png'
i =0 
kernel = np.array([[1,1,1],
                   [1,-8,1],
                   [1,1,1]])

for r_im in glob.glob(real_path):
    r_img = cv2.imread(r_im)
    r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
    out = cv2.Laplacian(r_img, cv2.CV_64F, kernel)
    
    
    filename = './datasets/Mve_frame/key/key_img' + str(i).zfill(5) + '.png'
    cv2.imwrite(filename, out)           
    i = i + 1
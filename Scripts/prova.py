# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:07:03 2019

@author: mzullich
"""

from CameraCalibration import *

K, dist, rvecs, tvecs, errStats, imgpoints, objpoints, picsList = calibrateLive((7,5), maxTime = 30, repErrThresh = .5, verbose = True)

plt.imshow(picsList[0])

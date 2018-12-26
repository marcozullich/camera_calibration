# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:14:52 2018

@author: mzullich
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#0. define import function for groups of imgs in same folder
def importImagesFolder(path, retList = None):
    '''
    import the whole content of a folder supposedly containing images
    ---
    * path -> path of the folder
    * retList -> a list to which images are appended to
    ---
    returns retList itself (in case None has been specified in its place)
    '''
    myImagesDirList = os.listdir(path)
    if retList == None:
        retList = []
        
    for img in myImagesDirList:
        retList.append(cv2.imread(path + '/' + img))
        
    return retList

def detectAndSaveCheckerboardInList(imgList, checkerboardSize, savePath = None):
    '''
    tries detection of a checkerboard of the specified size in a list of images
    The images are converted in grayscale and the checkerboard detection is 
    tried.
    If a savePath is specified, it'll draw the checkerboard and save the images
    there.
    ---
    * imgList -> the list of (BGR) images
    * checkerboardSize -> tuple containing the horizontal and vertical number
      of crossings within the checkerboard
    * savePath -> path where to save the images with the checkerboard drawn
    ---
    returns
    * a list of BGR images with the checkerboards drawn (where detected)
    * the list of object points
    ... expand?
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height*length,3), np.float32)
    objp[:,:2] = np.mgrid[0:height,0:length].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    imgList_gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in imgList]
    
    checkerBoardImages = []
    
    for i in range(len(imgList)):
        ret, corners = cv2.findChessboardCorners(imgList_gray[i],
                                                 checkerboardSize, None)
        
        if ret:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(imgList_gray[i] ,corners,
                                        checkerboardSize,(-1,-1),
                                        criteria)
            imgpoints.append(corners2)
            
            img = cv2.drawChessboardCorners(imgList[i], checkerboardSize, corners2 ,ret)
            
            checkerBoardImages.append(img)
            
            if savePath != None:
                cv2.imwrite(savePath + '/checker_' + str(i) + '.jpg', img)
                
    return img
    
def getCheckerboardSize(img, maxSizes):
    '''
    tries detection of checkerboard with a variable size within BGR image
    ---
    * img -> BGR image
    * maxSizes -> 2-sized tuple containing the maximum size of the checkerboard
      to be searched in the image. Both dimensions must be larger than 2
    ---
    returns a list of tuples containing the dimensions in which the
    checkerboard has been detected
    '''
    
    size_a, size_b = maxSizes
    if size_a<=2 or size_b<=2:
        raise ValueError("dimensions for checkerboard must be larger than 2" +
                         "for both axes")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    checkerboardDetectedIn = []
    
    for i in range(3, size_a):
        for j in range(3, size_b):
            ret, corners = cv2.findChessboardCorners(img_gray, (i,j), None)
            if ret:
                checkerboardDetectedIn.append((i,j))
    
    return checkerboardDetectedIn



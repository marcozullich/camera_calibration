# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 20:28:49 2018

@author: Marco Zullich and Domagoj Korais

add comments to functions
"""

import numpy as np
import cv2
import numbers

def draw_solid(solidType, solidCharacterization, origin,
               imgToDrawInto, checkerCorners, K, rvec, tvec, dist,
               baseColor = (0,0,255), pillarsColor = (255,0,0),
               ceilingColor = (0,255,0)):
    '''
    Draws a specified solid to a copy of an image passed by the user
    ---
    * solidType -> one of "Cube", "Pyramid"...
    * solidCharacterization -> list of quantities characterizing the solid:
        -Cube: a list of 1 value or a scalar [side]
        -Pyramid: a list of 3 values [sideWidth, sideHeight, height]
        -Cylinder: a list of 2 values [radius, height]
    * origin -> the (x,y) real world coordinates of the...
                (Cube/Pyramid) upper left corner of the base of the solid;
                (Cylinder) the center of the base circle.
    * imgToDrawInto -> image containing the checkerboard in which the solid
      will be drawn onto
    * checkerCorners -> array contaning the corners of the checkerboard
    * K -> intrinsics matrix
    * rvec -> rotation matrix from world to image for the given image
    * tvec -> translation vector from world to image for the given image
    * dist -> distortion coefficients for the perspective projection
    * baseColor -> the color of the base of the solid
    '''
    #sp = np.int32(solidPoints).reshape(-1,2)
    
    
    
    img = np.copy(imgToDrawInto)
    o_x, o_y = origin
    origin = [o_x, o_y, 0]

    if solidType == "Cube":
        side = 0
        if isinstance(solidCharacterization, numbers.Number):
            side = solidCharacterization
        elif isinstance(solidCharacterization[0], numbers.Number):
            side = solidCharacterization[0]
        else:
            raise ValueError("Expected a list or scalar as"\
                             " solidCharacterization")
        p = np.float32([origin, [o_x, o_y+side, 0], [o_x+side, o_y+side, 0], 
                        [o_x+side, o_y, 0], [o_x, o_y, -side],
                        [o_x, o_y+side, -side], [o_x+side, o_y+side, -side], 
                        [o_x+side, o_y, -side]])
        p_2d, _ = cv2.projectPoints(p, rvec, tvec, K, dist)
        p_2d = np.int32(p_2d).reshape(-1,2)
        print(p_2d)
        
        #create two layers for transparent overlay
        imgc = np.copy(img)
        imgd = np.copy(img)
        cv2.drawContours(imgc, [p_2d[:4]], -1, baseColor, -3)
        for i in range(4):
            cv2.line(img , tuple(p_2d[i]), tuple(p_2d[i+4]), pillarsColor, 5)
            cv2.line(imgc, tuple(p_2d[i]), tuple(p_2d[i+4]), pillarsColor, 5)
            cv2.line(imgd, tuple(p_2d[i]), tuple(p_2d[i+4]), pillarsColor, 5)
            #j=i+5
            #if j==8:
            #    j=4
            
            #v2.line(img, tuple(p_2d[i+4]), tuple(p_2d[j]), ceilingColor, 5)
        cv2.drawContours(imgd, [p_2d[4:8]], -1, ceilingColor, -3)
        cv2.addWeighted(imgc, .93, img, .07, 0, img)
        cv2.addWeighted(imgd, .75, img, .25, 0, img)
    elif solidType=="Pyramid":
        sx, sy, z = tuple(solidCharacterization)
        p = np.float32([origin, [o_x, o_y+sy, 0], [o_x+sx, o_y+sy, 0], 
                        [o_x+sx, o_y, 0], [(o_x+sx)/2, (o_y+sy)/2, -z]])
        p_2d, _ = cv2.projectPoints(p, rvec, tvec, K, dist)
        p_2d = np.int32(p_2d).reshape(-1,2)
        
        imgc = np.copy(img)
        cv2.drawContours(imgc, [p_2d[:4]], -1, baseColor, -3)
        
        for i in range(4):
            cv2.line(img , tuple(p_2d[i]), tuple(p_2d[4]), pillarsColor, 5)
            cv2.line(imgc, tuple(p_2d[i]), tuple(p_2d[4]), pillarsColor, 5)
        
        cv2.addWeighted(imgc, .75, img, .25, 0, img)
        
    elif solidType=="Cylinder":
        r, h = tuple(solidCharacterization)
        
        #print(solidCharacterization)
        
        totpts = 256
        step = 2*np.pi/totpts
        
        #print(step)
        
        crf_ground = []
        crf_ceil = []
        
        for i in range(totpts):
            crfx = r*np.cos(step*i)+o_x
            crfy = r*np.sin(step*i)+o_y
            crf_ground.append([crfx,crfy,0])
            crf_ceil.append([crfx,crfy,-h])

        #print(crf_ground)
        
        
        crf_ground = np.float32(crf_ground)
        #return crf_ground
    
        #print(crf_ground.shape)
        crf_ceil = np.float32(crf_ceil)
        
        crf_gr_2d, _ = cv2.projectPoints(crf_ground, rvec, tvec, K, dist)
        crf_gr_2d = np.int32(crf_gr_2d).reshape(-1,2)
        
        #return cfr_gr_2d
        crf_ce_2d, _ = cv2.projectPoints(crf_ceil, rvec, tvec, K, dist)
        crf_ce_2d = np.int32(crf_ce_2d).reshape(-1,2)
        
        imgc = np.copy(img)
        imgd = np.copy(img)
        
        cv2.drawContours(imgc, [crf_gr_2d], -1, baseColor, -3 )
        cv2.drawContours(imgd, [crf_ce_2d], -1, ceilingColor, -3 )
        
        cv2.addWeighted(imgc, .93, img, .07, 0, img)
        cv2.addWeighted(imgd, .75, img, .25, 0, img)
        
            
    else:
        raise ValueError("Figure not supported")
    return img
        


#from CameraCalibration import *
#
#imgs = importImagesFolder('../images_calib_custom/')
#checkerboardSize = (7,5)
#imgs_check, objpoints, imgpoints, flags = detectAndSaveCheckerboardInList(imgs, checkerboardSize)
#    
#ret, mtx, dist, rvecs, tvecs = calibrateCamera(
#    objpoints, imgpoints, imgs_check[0].shape[0:2], True)
#
#repError_dist, projPoints_dist = computeReprojectionError(imgpoints, objpoints, rvecs,
#                        tvecs, mtx, dist)
#
#img_ind = 4
#img = draw_solid('Cylinder', (1.5,1), (3,3), imgs[img_ind], imgpoints[img_ind], mtx, rvecs[img_ind], tvecs[img_ind], dist)
#plt.figure(figsize=(50, 20))
#imgplot = plt.imshow(img)
#

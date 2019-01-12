# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:14:52 2018

@authors: Marco Zullich and Domagoj Korais
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import threading
import time
import queue

def importImagesFolder(path, retList = None):
    '''
    import the whole content of a folder supposedly containing images
    into a new list, or appends it in case an already existing list is passed
    to as an input.
    ---
    * path -> path of the folder
    * retList -> a list to which images are appended to
    ---
    returns retList
    '''
    #strip whitespace and slash from string
    path = path.strip(' ').strip('/')
    
    #get list of files in given path
    myImagesDirList = os.listdir(path)
    #if no list was passed, create a new one
    if retList == None:
        retList = []
    
    #append each image in folder to list
    for img in myImagesDirList:
        #convert to RGB for correct matplolit representation
        img_BGR = cv2.imread(path + '/' + img)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        retList.append(img_RGB)
        
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
    * the list of image points
    * a list of booleans indicating whether the desired pattern (checkerboard
      of specified size) was found within the corresponding image in imgList
    '''
    
    if savePath != None:
        savePath = savePath.strip(' ').strip('/')
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points in 3D
    objp = np.zeros((checkerboardSize[0]*checkerboardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboardSize[0],0:checkerboardSize[1]].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    imgList_gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in imgList]
    
	#Prepare lists where return values will be stored into
    checkerBoardImages = [] #list of images with checkerboard drawn
    checkerboardFound = [] #list of bools, determining if pattern found in images
    
	#loop through all images
    for i in range(len(imgList)):
        ret, corners = cv2.findChessboardCorners(imgList_gray[i],
                                                 checkerboardSize, None)
        checkerboardFound.append(ret)
        
        if ret: #if checkerboard found within current image
            objpoints.append(objp) #append 3D points
            
			#refine intersections location within current image
			#TODO change window size
            corners2 = cv2.cornerSubPix(imgList_gray[i] ,corners,
                                        checkerboardSize,(-1,-1),
                                        criteria)
			#adds refined 2D coordinates to imgpoints list
            imgpoints.append(corners2)
            
			#copy image (not to overwrite it)
            img = np.copy(imgList[i])
            
			#draw chessboard corners on copied image
            cv2.drawChessboardCorners(img, checkerboardSize, corners2, ret)
            
			#if a savePath was specified, save image
            if savePath != None:
                cv2.imwrite(savePath + '/checker_' + str(i) + '.jpg', img)
            
            checkerBoardImages.append(img)
            
    return checkerBoardImages, objpoints, imgpoints, checkerboardFound
    
def getCheckerboardSize(img, maxSizes = (20,20)):
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
    
    for i in range(3, size_a+1):
        for j in range(i, size_b+1):
            ret, corners = cv2.findChessboardCorners(img_gray, (i,j), None)
            if ret:
                checkerboardDetectedIn.append((i,j))
    
    return checkerboardDetectedIn

def calibrateCamera(objpoints, imgpoints, imgSize, considerRadialDistortion, fisheye=False):
    '''
    wrapper over cv2.calibrateCamera to render it more comprehensible wrt the
    subjects which were studied during lectures; in particular, it simplifies
    all the complicated structure of flags which are used in cv2's method
    in order to limit it to a bool over radial distortion.
    ---
    * objpoints -> list of 3-D object points corresponding to the checkerboard
      intersections (one array for each image)
    * imgpoints -> list of corresponding 2-D within the image (one array for
      each image)
    * imgSize -> tuple containing the (common) size of all images
    * considerRadialDistortion -> boolean indicating whether radial distortion
      has to be considered during the calibration
          True means that a simplified model considering only the parameters
          k1, k2 will be employed
    ---
    returns the same objects returned by cv2's method:
    * ret -> error (average of average L-2 norm on reprojected vs image points)
    * mtx -> K (intrinsics matrix)
    * dist -> distortion coefficients
    * rvecs -> rotation matrices (extrinsics) - one for each calibration image
    * tvecs -> translation vector (extrinsics) - one for each calibration image
    '''
    distCoeffs = None
    flags= cv2.CALIB_ZERO_TANGENT_DIST 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    if not considerRadialDistortion:
        
        flags += cv2.CALIB_FIX_K1 + \
                 cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
        
    if fisheye:
        flags += cv2.CALIB_RATIONAL_MODEL
        #flags += cv2.CALIB_FIX_K3
        
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, imgSize, None,
                            distCoeffs = distCoeffs, flags = flags,criteria=criteria)
        
    return ret, mtx, dist, rvecs, tvecs
    



def computeReprojectionError(imgpoints, objpoints, rvecs, tvecs, K, dist):
    '''
    computes reprojection error given a 2-D image points, the corresponding 3-D
    object point related to the same checker-board intersections and the values
    of the intrinsics, extrinsics and the distortion coefficients
    ---
    * imgpoint -> list of array of points corresponding to the checkerboard inter-
      sections (2-dimensional) for one single calibration image
    * objpoint -> list of array of object points (3-dimensional)
    * rvec -> rotation matrices for the projections
    * tvec -> translation vectors for the projections
    * K -> intrinsics matrix of camera
    * dist -> vector containing distortion coefficients (k1, k2, p1, p2, k3)
    ---
    returns
    * errs -> list containig the total geometric residual for each calibration
      image
    * projPoints -> list containing the set of reprojected points (an array for
      each of the calibration images)
    '''
    errs = [None]*len(imgpoints)
    projPoints = [None]*len(imgpoints)
    
    for i in range(len(imgpoints)):
        
        reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                           K, dist)
        errs[i] = np.sum((imgpoints[i] - reprojected)**2)
        projPoints[i] = reprojected
    return errs, projPoints

def _shootFromActiveWebcam(cap):
    '''
    This function shoots a picture from a given active webcam caption and
    returns the frame
    ---
    *cap -> the webcam capture
    ---
    returns picture (ndarray) just shot
    '''
    ret, frame = cap.read()
    return frame
    
def _shootContinuouslyFromActiveWebcamAndAddToList(cap, picsList, tmInt,
                                                   status, verbose):
    '''
    This method lets us continuously shoot pictures from a given active webcam
    caption with a given time interval and appends them to a list which is
    passed on to this function
    ---
    * cap -> the webcam capture
    * picsList -> the images list
    * tmInt -> the time interval between one shot and the following
    * status -> queue for status (used for threading)
    * verbose -> boolean for verbosity
    '''
    i = 1
    if verbose:
        print("ST: initialized shooting thread and preparing to shoot pic {}".\
              format(i))
    while(status.empty() or (not status.get(False).startswith("stop"))):
        pic = _shootFromActiveWebcam(cap)
        picsList.append(pic)
        cv2.imshow('pic',pic)
        cv2.waitKey(1)
        
        if verbose:
            print("ST: shot pic {} and appended to list".format(i))
            
        i+=1
        time.sleep(tmInt)
        
        if verbose:
            print("ST: woke up and starting to shoot pic {}".format(i))
    
    cv2.destroyAllWindows()
    
def _detectCheckerAndCalibrate(imgList, imgpoints, objpoints, checkerSize,
                               startIndex = 0):
    '''
    This method lets us detect checkerboard in a given list of image
    (with the possibility to start at a given checkpoint within the list)
    and calibrate camera according to the detected correspondences plus
    previous correspondences
    ---
    * imgList -> list of calibration images
    * imgpoints -> list of 2d points corresponding to the checkerboard inter-
      sections
    * objpoints -> list of 3d points corresponding to the checkerboard inter-
      sections IRL
    * checkerSize -> 2d tuple representing the size of the checkerboard to be
      searched into the pictures
    * startIndex -> index relative to the imgList at which the algorithm has to
      start detection and append its results to img and objpoints
    ---
    returns:
    * total reprojection error
    * calibration matrix
    * distortion coefficients
    * rotation matrices
    * translation vectors
    * updated list of 2d points corresponding to checkerboard corners
    * updated list of 3d points corresponding to checkerboard corners
    '''
    #detect pattern in desired images
    _, objpoints2, imgpoints2, foundPattern =\
                detectAndSaveCheckerboardInList(imgList[startIndex:],
                                                checkerSize)
    #print(objpoints2)

    #append points in 2d and 3d to corresponding lists
    imgpoints.extend(imgpoints2)
    objpoints.extend(objpoints2)
    #print(objpoints)
    ret = None
    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    
    #start calibration
    if(len(objpoints)>=8):
        ret, mtx, dist, rvecs, tvecs = calibrateCamera(objpoints, imgpoints,
                                                       imgList[0].shape[0:2],
                                                       True)
    
    return ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints

def _calibrateContinuously(imgList, checkerSize, tmInt, repErrThresh, status,
                           verbose, retList):
    '''
    This methods lets us continuously calibrate our camera with a specified 
    time interval based upon a list of images which is supposed that to be
    fed as this algorithm keeps running. When a given reprojection error
    threshold is improved, the algorithm stops returning calibration infos
    and the error statistics wrt time.
    Note: calibration is based upon the wrapper provided by this library,
    so see docs for calibrateCamera for further info on the procedure
    ---
    * imgList -> list of images used for calibration
    * checkerSize -> the size of the checkerboard to be searched into the pics
    * timInt -> time interval from one calibration to the next one
    * repErrThresh -> threshold for total reprojection error under which the
      calibration stops and the methods return
    * status -> queue for status checking (used for threading)
    * verbose -> boolean for verbosity
    * retList -> a list (possibily empty) to which the return will be appended
    ---
    returns [appends to retList]
    * calibration matrix
    * distortion coefficients
    * rotation matrices
    * translation vectors
    * list containing statistics of total reprojection error and time
      elapsed
    * list containing 2d points corresponding to the checkerboard intersections
      (one array for each image captured)
    * list containing 3d points corresponding to the checkerboard intersections
      (one array for each image captured)
    '''    

    K = None
    dist = None
    rvecs = None
    tvecs = None
    errStats = []
    previousMark = 0
    imgpoints = []
    objpoints = []
    
    threshImproved = False
    
    i=1
    if verbose:
        print("CT: initialized calibrating thread and waiting for pictures"+\
            " list to be filled")
    
    while(status.empty() or (not status.get(False).startswith("stop"))):
        imgListSize = len(imgList)
        #need at least 8 pictures to start calibrating
        if(imgListSize>=8):
            
            if verbose:
                print("CT: image with check in list: {}; started calibration".\
                      format(imgListSize))
                
            ret, K, dist, rvecs, tvecs, imgpoints, objpoints =\
                        _detectCheckerAndCalibrate(imgList, imgpoints,
                                                   objpoints, checkerSize,
                                                   previousMark)
                        
            if verbose:
                print("CT: images in list: {}; finished calibration".\
                      format(imgListSize) + ". Checkerboard(s) detected: "+\
                      str(len(objpoints)) + ". Error rate:" + str(ret))
            
            if ret!= None:
                previousMark = imgListSize
                errStats.append(np.array([ret, 0])) #TODO add time
                if(ret<repErrThresh):
                    threshImproved = True
                    break
        else:
            if verbose:
                print("CT: not enough pics to start calibrating ({})".\
                      format(imgListSize))
        time.sleep(tmInt)
    
    retList.extend([K, dist, rvecs, tvecs, errStats, imgpoints, objpoints])
    
    reason = ""
    if(threshImproved):
        reason = "min error threshold improved"
    else:
        reason = "max time expired"
        
    
    if verbose:
        print("CT: calibration ended due to " + reason)

##TODO -> multiprocessing
def calibrateLive(checkerSize, webcamIndex = 0, tmInt = .25, maxTime = 30,
                  repErrThresh = 1.0, verbose = False):
    '''
    Calibrate a webcame of user's choice. The camera will shoot picture at a
    rate given by the users (tmInt) and will be calibrate as soon as enough
    picture with a checkerboard are detected by the algorithm. The algorithm
    will stop either
        a. when the total reprojection error as defined within the opencv lib
           will improve a given threshold
         or
        b. maxTime seconds will be passed
    ---
    * checkerSize -> tuple containing the 2d dimension of the checkerboard
      (horizontal intersections, vertical intersections)
    * webcamIndex -> index of the webcam to be calibrated
    * tmInt -> time interval (in seconds) between a shot and the following.
      Also, the rate at which calibration will be performed is tmInt*4
    * maxTime -> maximum number of seconds the algorithm will continuously
      perform the calibration
    * repErrThresh -> maximum reprojection error (as defined in the cv2 lib)
      allowed for the algorithm to stop
    * verbose -> verbosity of the method (use for testing sake)
    '''
    #this will store pics from shooting thread
    picsList = []
    #this will store the return from calibration trhead
    retList_calib = []

    cap = cv2.VideoCapture(0)
    #status vector
    status_shoot = queue.Queue()
    status_calibrate = queue.Queue()
    
    thread_shoot = threading.Thread(name ='Shoot',
                    target = _shootContinuouslyFromActiveWebcamAndAddToList,
                    args = (cap, picsList, tmInt, status_shoot, verbose))
    
    thread_calibrate = threading.Thread(name = 'Calibrate',
                                        target = _calibrateContinuously,
                                        args = (picsList, checkerSize, tmInt*4,
                                                repErrThresh, status_calibrate,
                                                verbose, retList_calib))
    
    if verbose:
        print("MAIN: all threads are set and ready to start")
    
    thread_shoot.start()
    thread_calibrate.start()
    
    t_start = time.time()
    t_end = t_start + maxTime
    
    
    while time.time() < t_end:
        if not (thread_calibrate.is_alive() and thread_shoot.is_alive()):
            t_end = time.time()
            break
    
    status_shoot.put("stop")
    status_calibrate.put("stop")

    thread_shoot.join()
    thread_calibrate.join()
    cap.release()
    
    if verbose:
        print("MAIN: all threads ended and method ready for return. Total "+
              "time elapsed: {}".format(t_end - t_start))
    
    K, dist, rvecs, tvecs, errStats, imgpoints, objpoints =\
        tuple(retList_calib)
    
    return K, dist, rvecs, tvecs, errStats, imgpoints, objpoints, picsList
    


#K, dist, rvecs, tvecs, errStats, imgpoints, objpoints, picsList = calibrateLive((7,5), maxTime = 30, verbose = True)


    

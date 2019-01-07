# CVPR_Exam
Computer Vision Exam Project, A.Y. 2018/'19

Domagoj Korais

Marco Zullich

# Exam project no. 1 → «CALIBRATING CAMERA AND SUPERIMPOSING AN OBJECT ON THE CALIBRATION PATTERN»

## Problem Statement
The problem may be split into two logical steps:
1.	<u>Camera calibration</u><br>
Based upon a set of pictures from the same camera depicting a common coplanar calibration pattern, compute the intrinsic and extrinsic parameters of the camera and its pose from the correspondences between 3D coordinates of the pattern’s points of interest and the corresponding 2D coordinates of such points in the image reference frame.
The task is ran twice, first without, then with compensation for radial distortion.
2.	<u>Superimposition of an object to the calibration plane</u><br>
Now that the parameters are known, they can be used to project points and solids of arbitrary 3D coordinate into any of the previous calibration images.

## Approach
Task 1 (camera calibration) is carried out using Zhang’s method for homography estimation. The calibration object is a white/black 2D checkerboard with 35 intersections (7 columns x 5 rows) stuck on a paperback cardboard.<br>
As far as radial distortion is concerned, after the estimation of the homography and calculation of the intrinsic ($K$) and extrinsic ($R,t$) parameters, we estimate distortion coefficients up to the 3rd order, we compensate for it refining the 3D-2D correspondences between checkerboard intersections in real world and within the images and re-running Zhang’s procedure iteratively.<br>
After obtaining $K,R,t$, we have a rule for transforming any point within the world’s reference frame (whose origin is fixed at the upper-left intersection of the checkerboard) into the 2D coordinates of any one of our calibration pictures, and hence we can superimpose any (virtual) object of our choice to our images, thus carrying out task 2.

## Implementation
We opted in favor of implement our program in Python, using mainly two libraries:
* OpenCV
* NumPy
The former in particular uses very efficient methods to quickly carry out some common computer vision and image processing tasks.
<br><br>
The code is contained within the folder "Scripts", which contains two Python libraries:
* CameraCalibration.py is a library that encapsulates a number of methods constructed mainly over built-in OpenCV routines. It's designed to carry out task 1. Though specifically thought for this project, they can be easily reproduced on similar settings.
* drawingHelper.py is a library that contains only one method (draw_solid) and is designed for task 2.
The libraries are put in display within the IPython notebook "KoraisZullich_proj2.ipynb".


# CVPR_Exam
Computer Vision Exam Project, A.Y. 2018/'19

Domagoj Korais

Marco Zullich

# Exam project no. 1 → «CALIBRATING CAMERA AND SUPERIMPOSING AN OBJECT ON THE CALIBRATION PATTERN»

## Problem Statement
The problem may be split into two logical steps:
1.	Camera calibration
Based upon a set of pictures from the same camera depicting a common coplanar calibration pattern, compute the intrinsic and extrinsic parameters of the camera and its pose from the correspondences between 3D coordinates of the pattern’s points of interest and the corresponding 2D coordinates of such points in the image reference frame.
The task is ran twice, first without, then with compensation for radial distortion.
2.	Superimposition of an object to the calibration plane
Now that the parameters are known, they can be used to project points and solids of arbitrary 3D coordinate into any of the previous calibration images.

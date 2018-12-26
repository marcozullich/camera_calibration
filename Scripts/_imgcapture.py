# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 18:44:41 2018

@author: MZULLICH
"""

import cv2
import os

def isInt(strNo):
    try:
        int(strNo)
        return True
    except Exception:
        return False


onlyfiles = [int(f[7:f.find(".")]) for f in os.listdir(".") if os.path.isfile(os.path.join(".", f)) and f.startswith("capture") and isInt(f[7:f.find(".")])]
onlyfiles.sort()
try:
    i = max(onlyfiles)
except Exception:
    i = 0

#start using camera
cap = cv2.VideoCapture(0)

print("Started recording. Press spacebar to shoot, 'q' to quit")

#do forever - until a special key is pressed
while(1):
    #get live feed from camera and show it
    _,frame = cap.read()
    cv2.imshow('Live recording',frame)
    
    c = cv2.waitKey(5)
    
    if 'q' == chr(c & 255):
        print("Quitting program")
        break
    if ' ' == chr(c & 255):
        print("Image obtained, press 's' to save it, 'q' to quit, 'r' to save and re-shoot, any other key to re-shoot it without saving")
        c2 = cv2.waitKey(0)
        i+=1
        if 's' == chr(c2 & 255) or 'r' == chr(c2 & 255):
            cv2.imwrite('capture'+str(i)+'.jpg',frame)
            print("Image saved as 'capture"+str(i)+".jpg' in the working directory")
            if 's' == chr(c2 & 255):
                break
        if 'q' == chr(c2 & 255):
            print("Quitting program")
            break
        else:
            print("Re-starting. You may try another shot!")
        
cv2.destroyAllWindows()
cap.release()

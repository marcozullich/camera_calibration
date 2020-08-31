import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("Scripts")
from Scripts import CameraCalibration as cc, drawingHelper as dh

@st.cache()
def calibrate():
    print("CAL")
    imgs = cc.importImagesFolder("/images_calib_custom")
    checkerboardSize = (7,5)
    imgs_check, objpoints, imgpoints, flags = cc.detectAndSaveCheckerboardInList(imgs, checkerboardSize)

    ret, mtx, dist, rvecs, tvecs = cc.calibrateCamera(objpoints, imgpoints, imgs_check[0].shape[0:2], True)

    repError_dist, projPoints_dist = cc.computeReprojectionError(imgpoints, objpoints, rvecs,
                        tvecs, mtx, dist)

    return imgs, imgpoints, mtx, dist, rvecs, tvecs

@st.cache()
def select_image(images, index=None):
    if index is None:
        index = np.random.choice(len(images))
    return index

def main():
    st.markdown("# Sovraimposizione solidi su immagini con fotocamera calibrata")

    imgs, imgpoints, mtx, dist, rvecs, tvecs = calibrate()

    img_ind = st.sidebar.text_input("ID immagine (vuoto per immagine casuale):")
    img_ind = int(img_ind) if not (img_ind is None or img_ind == "") else None

    # selezione casuale di un'immagine
    img_ind = select_image(imgs, img_ind)

    solidType = st.sidebar.selectbox("Tipo di solido da disegnare",
                            ["Cubo",
                             "Piramide",
                             "Cilindro"])
    
    o_x = st.sidebar.slider("Origine (asse x):", 0.0, 6.0, 2.0, step=.1)
    o_y = st.sidebar.slider("Origine (asse y):", 0.0, 4.0, 2.0, step=.1)

    # o_x = int(st.sidebar.text_input("Origine (asse x):", 2))
    # o_y = int(st.sidebar.text_input("Origine (asse y):", 2))

    if solidType == "Cubo": 
        solid_characterization = [st.sidebar.slider("Lato del cubo", .1, 4.0, 2.0, step=.1)]
    elif solidType == "Piramide":
        solid_characterization = [st.sidebar.slider("Larghezza base piramide", .1, 4.0, 2.0, step=.1),
                                  st.sidebar.slider("Profondità base piramide", .1, 4.0, 3.0, step=.1),
                                  st.sidebar.slider("Altezza piramide", .1, 5.0, 2.5, step=.1)]
    elif solidType == "Cilindro":
        solid_characterization = [st.sidebar.slider("Raggio cilindro", .1, 4.0, 2.0, step=.1),
                                  st.sidebar.slider("Altezza cilindro", .1, 5.0, 2.0, step=.1)]

    solidType_conversion = {
        "Cubo": "Cube",
        "Piramide": "Pyramid",
        "Cilindro": "Cylinder"
    }

    img = dh.draw_solid(
        solidType=solidType_conversion[solidType],
        solidCharacterization=[float(s) for s in solid_characterization],
        origin = (o_x, o_y),
        imgToDrawInto=imgs[img_ind],
        checkerCorners=imgpoints[img_ind],
        K=mtx,
        rvec=rvecs[img_ind],
        tvec=tvecs[img_ind],
        dist=dist
    )

    st.image(img)

if __name__=="__main__":
    main()
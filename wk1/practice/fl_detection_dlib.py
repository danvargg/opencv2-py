#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-12-11
Author: @danvargg
"""
import dlib
import cv2
import numpy as np
# from labs import renderFace

PREDICTOR_PATH = '/home/daniel/edu/opencv2-py/wk1/data/models/shape_predictor_5_face_landmarks.dat'

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

imageFileName = '/home/daniel/edu/opencv2-py/wk1/data/images/family.jpg'

im = cv2.imread(imageFileName)

faceRects = faceDetector(im, 0)

print(len(faceRects))

landmarksAll = []
for i in range(0, len(faceRects)):
    newRect = dlib.rectangle(
        int(faceRects[i].left()), int(faceRects[i].top()), int(faceRects[i].right()), int(faceRects[i].bottom())
    )
    # For every face rectangle, run landmarkDetector
    landmarks = landmarkDetector(im, newRect)
    # Print number of landmarks
    if i == 0:
        print("Number of landmarks", len(landmarks.parts()))

    landmarksAll.append(landmarks)

print(landmarksAll)

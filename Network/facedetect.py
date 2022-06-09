import cv2
import os
import numpy as np

cv2.ocl.setUseOpenCL(False)

# Simple script  facedetection using opencv cascade classifier
def find_faces(img, feature_file='Network/res/haarcascade_frontalface_default.xml'):
	DETECTION_PARAMS = feature_file

	detector = cv2.CascadeClassifier(DETECTION_PARAMS)

	found_faces = detector.detectMultiScale(img, 1.1, 4)

	faces = []

	for (x, y, w, h) in found_faces:
		image = img.copy()[y:y+h, x:x+w]
		faces.append(image.copy())

	if len(faces) == 0:
		return None

	return faces

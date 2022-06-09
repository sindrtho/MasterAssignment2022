import cv2
import numpy as np
import keras as k
from Network.facedetect import find_faces
import matplotlib.pyplot as plt
import open3d as opd

cap = cv2.VideoCapture(0)


IMAGES = ['020', '040', '050', '060', '080']
frames = [cv2.imread(f'Dataset/Disgust/F001/{image}.jpg') for image in IMAGES]
images = []
# while len(images) < N:
# 	_, frame = cap.read()
# 	faces = find_faces(frame)

# 	cv2.imshow('frame', frame)
	
# 	if faces:
# 		face = faces[0]
# 		cv2.imshow('face', face)
# 	else:
# 		cv2.destroyWindow('face')

# 	c = cv2.waitKey(1)
# 	if c == ord('c') and faces:
# 		images.append(faces[0])
# 	elif c == ord('q'):
# 		break

for frame in frames:
	faces = find_faces(frame)

	face = faces[0]
		
	images.append(faces[0])

batch = np.zeros((1, 5, 256, 256, 3), dtype=np.float32)

for i, face in enumerate(images):
	face = face.astype(np.float32)
	face = cv2.resize(face, (256, 256))
	cv2.normalize(face, batch[0, i], 0, 1, cv2.NORM_MINMAX)
	# cv2.imshow('face', batch[0, i])
	# cv2.waitKey(0)
	# cv2.normalize(face, batch[0, i], 0, 1, cv2.NORM_MINMAX)

cv2.destroyAllWindows()

# model = k.models.load_model('MODELS/8filters.h5')
model = k.models.load_model('results/16filters/model.h5')
# model = k.models.load_model('model.h5')
model.compile()

preds = model.predict(batch)[0]

pc = opd.geometry.PointCloud()
pc.points = opd.utility.Vector3dVector(preds)
opd.io.write_point_cloud('demo.ply', pc)
import cv2
import numpy as np
import keras as k
from Network.facedetect import find_faces
import matplotlib.pyplot as plt
import open3d as opd

cap = cv2.VideoCapture(0)

# Replace IMAGES_PATH with path to dataset used for demonstration.
# Takes in path to single subjects images, NOT dataset root directory.
IMAGES_PATH = 'Dataset/BU4DFE/'
IMAGES_PATH = [
	'/'.join([IMAGES_PATH, f]) if f.endswith('.jpg') for f in os.listdir(IMAGES_PATH)
	]

# Replace MODEL_PATH with path to saved model used for demo.
#If model uses more or less images, change N
MODEL_PATH = 'Path to model'
N = 8

IMAGES = [IMAGES_PATH[i]
	for i in np.random.choice(len(IMAGES_PATH-1), N, replace=False)]

frames = [cv2.imread(f'Dataset/Disgust/F001/{image}.jpg') for image in IMAGES]
input_images = []

for frame in frames:
	faces = find_faces(frame)

	face = faces[0]
		
	input_images.append(faces[0])

batch = np.zeros((1, N, 256, 256, 3), dtype=np.float32)

for i, face in enumerate(input_images):
	face = face.astype(np.float32)
	face = cv2.resize(face, (256, 256))
	cv2.normalize(face, batch[0, i], 0, 1, cv2.NORM_MINMAX)

	""" Uncomment 3 next lines to overview the selected images """
	# cv2.imshow('face', batch[0, i])
	# cv2.waitKey(0)
	# cv2.normalize(face, batch[0, i], 0, 1, cv2.NORM_MINMAX)

cv2.destroyAllWindows()

model = k.models.load_model(MODEL_PATH)
model.compile()

preds = model.predict(batch)[0]

pc = opd.geometry.PointCloud()
pc.points = opd.utility.Vector3dVector(preds)
opd.io.write_point_cloud('demo.ply', pc)